"""Tolerance-aware AnomalyDINO with configurable supervised residual projection heads.

The normal detector path intentionally delegates to anomalib's stock
``AnomalyDINOModel`` during training and in detector-only/parity mode.  This
keeps normal-memory construction and detector scoring identical to the stock
model while adding a separate residual classifier for acceptable/reject
appearances.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F

from anomalib.data import InferenceBatch
from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel

logger = logging.getLogger(__name__)

PATCH_GOOD = 0
PATCH_ACCEPT = 1
PATCH_REJECT = 2
PATCH_UNKNOWN = 3

# Persisted deployment-mode encoding.  The integer lives in ``state_dict`` so
# checkpoint reloads recover the requested inference/export path without an
# external mode argument.  It is intentionally synchronized to ordinary Python
# attributes outside ``forward()`` so ONNX sees a fixed graph and no mode input.
RESIDUAL_PROJECTION_MODE_TO_ID = {"direction": 0, "magnitude": 1, "dual": 2}
RESIDUAL_PROJECTION_ID_TO_MODE = {value: key for key, value in RESIDUAL_PROJECTION_MODE_TO_ID.items()}

# PT/ONNX parity hardening for ACCEPT-side decisions.  Hard threshold gates can
# turn tiny backend numeric differences into a zeroed-vs-hot patch or entire map.
# Use a narrow one-sided ramp: evidence must still cross the original ACCEPT
# boundary, then forgiveness reaches full legacy strength over this band.
# This is intentionally a code constant (not a runtime/export input or checkpoint
# parameter) so the deployment contract stays image-only.
ACCEPT_DECISION_TRANSITION_WIDTH = 5.0e-3
V8_DEFAULT_RESIDUAL_ANCHOR_K = 32
V81_RESIDUAL_ANCHOR_TEMPERATURE = 1.0e-2
V82_DEFAULT_RESIDUAL_ANCHOR_SEARCH_K = 128
V82_RESIDUAL_RERANK_CANDIDATE_CHUNK = 8


def _block_index_from_layer_name(layer_name: str) -> int:
    """Return a zero-based transformer block index from a timm layer name."""
    text = str(layer_name).strip()
    tail = text.rsplit(".", 1)[-1]
    try:
        return int(tail)
    except ValueError as exc:
        raise ValueError(
            f"Unable to determine transformer block index from layer name {layer_name!r}; "
            "expected a name ending in an integer, e.g. 'blocks.11'."
        ) from exc


def _resolve_final_layer_name(feature_encoder: object, encoder_name: str) -> tuple[str, int]:
    """Resolve the stock AnomalyDINO final transformer block without private APIs.

    Anomalib 2.3.3 uses its custom ``DinoVisionTransformer`` and exposes a
    ``blocks`` container directly. Prefer that authoritative depth. The older
    layer-name scan and architecture-name fallback remain for compatibility.
    """
    blocks = getattr(feature_encoder, "blocks", None)
    if blocks is not None and not bool(getattr(feature_encoder, "chunked_blocks", False)):
        try:
            depth = len(blocks)
        except TypeError:
            depth = 0
        if depth > 0:
            idx = depth - 1
            return f"blocks.{idx}", idx

    candidates: list[str] = []
    for attr in ("layers", "layer_names", "out_features"):
        value = getattr(feature_encoder, attr, None)
        if value is None:
            continue
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, dict):
            candidates.extend(str(v) for v in value.keys())
        else:
            try:
                candidates.extend(str(v) for v in value)
            except TypeError:
                pass

    parsed: list[tuple[int, str]] = []
    for layer_name in candidates:
        try:
            parsed.append((_block_index_from_layer_name(layer_name), layer_name))
        except ValueError:
            continue
    if parsed:
        idx, name = max(parsed, key=lambda item: item[0])
        return name, idx

    name = str(encoder_name).lower()
    architecture_depths = {
        "small": 12,
        "base": 12,
        "large": 24,
        "huge": 32,
        "giant": 40,
    }
    for architecture, depth in architecture_depths.items():
        if architecture in name:
            idx = depth - 1
            return f"blocks.{idx}", idx

    raise ValueError(
        "Unable to determine AnomalyDINO final transformer layer from the parent "
        f"feature extractor for encoder_name={encoder_name!r}. "
        "Set/use a supported DINO/DINOv2 ViT encoder."
    )


class TolerantAnomalyDINOModel(AnomalyDINOModel):
    """v8.3 Stock AnomalyDINO detector plus tolerance-aware residual banks.

    Important invariants:
    * Training always uses the stock AnomalyDINO ``forward`` implementation.
    * ``detector_only=True`` always uses stock AnomalyDINO inference.
    * With ``strict_stock_when_damping_one=True``, ``accept_damping=1`` and
      ``emit_patch_class=False``, inference also delegates to stock AnomalyDINO.

    This makes it possible to establish detector parity before evaluating the
    tolerance head.
    """


    def __init__(
        self,
        encoder_name: str = "dinov2_vit_small_14",
        num_neighbours: int = 1,
        masking: bool = False,
        scoring_mode: Literal["cosine", "pca"] = "cosine",
        pca_components: int = 32,
        coreset_subsampling: bool = False,
        sampling_ratio: float = 0.1,
        residual_creation_threshold: float = 0.1,
        residual_topk_per_image: int = 0,
        t_normal: float = 0.1,
        t_known: float = 0.5,
        t_accept_known: float | None = None,
        t_reject_known: float | None = None,
        margin: float = 0.05,
        accept_damping: float = 0.0,
        image_accept_enable: bool = False,
        image_accept_threshold: float = float("inf"),
        image_accept_damping: float = 0.0,
        image_reject_veto_enable: bool = True,
        image_reject_veto_threshold: float = 0.0,
        image_reject_boost_enable: bool = True,
        image_reject_boost_advantage_threshold: float = 0.0,
        image_reject_boost_lambda: float = 0.0,
        image_reject_boost_max: float = 1.0,
        # dual supervised residual projections (reference-only training)
        residual_projection_enable: bool = True,
        residual_projection_dim: int = 32,
        residual_projection_margin: float = 0.10,
        residual_projection_epochs: int = 40,
        residual_projection_steps_per_epoch: int = 64,
        residual_projection_batch_size: int = 128,
        residual_projection_lr: float = 2.0e-3,
        residual_projection_weight_decay: float = 1.0e-4,
        residual_projection_orthogonality_weight: float = 1.0e-3,
        residual_projection_use_magnitude: bool = True,
        residual_projection_magnitude_transform: Literal["log", "linear"] = "log",
        residual_projection_magnitude_scale: float = 1.0,
        residual_projection_magnitude_clip: float = 5.0,
        residual_projection_magnitude_eps: float = 1.0e-6,
        residual_projection_reject_type_weights: dict[str, float] | None = None,
        residual_projection_seed: int = 1337,
        residual_projection_fit_device: Literal["cpu", "model"] = "cpu",
        residual_projection_mode: Literal["direction", "magnitude", "dual"] | None = None,
        dual_projection_enable: bool = True,
        # Optional context channels for the learned projection only.
        # Relative XY uses normalized patch-center coordinates in [-1, 1]
        # relative to the cropped image grid. Intermediate layers are 0-based
        # transformer block indices (e.g. [8] for DINOv2-S/14).
        residual_projection_use_relative_xy: bool = False,
        residual_projection_xy_scale: float = 1.0,
        residual_projection_intermediate_layers: list[int] | tuple[int, ...] | None = None,
        residual_projection_intermediate_scale: float = 1.0,
        projected_reject_boost_enable: bool = True,
        projected_reject_boost_advantage_threshold: float = 0.0,
        projected_reject_boost_lambda: float = 0.0,
        projected_reject_boost_max: float = 0.75,
        direction_projected_reject_boost_advantage_threshold: float | None = None,
        direction_projected_reject_boost_lambda: float | None = None,
        magnitude_projected_reject_boost_advantage_threshold: float | None = None,
        magnitude_projected_reject_boost_lambda: float | None = None,
        detector_only: bool = False,
        strict_stock_when_damping_one: bool = True,
        emit_patch_class: bool = False,
        residual_bank_chunk_size: int = 65536,
        normal_bank_chunk_size: int = 65536,
        calibration_query_chunk_size: int = 256,
        calibration_seed: int = 1337,
        # v8/v8.1 compatibility: number of neighbours actually consumed by the
        # soft tolerance residual anchor. Appended args preserve old positional calls.
        residual_anchor_k: int = V8_DEFAULT_RESIDUAL_ANCHOR_K,
        # v8.2: wider fast-dtype retrieval pool, locally reranked in FP32 before
        # selecting the soft anchor neighbours. Must be >= residual_anchor_k.
        residual_anchor_search_k: int = V82_DEFAULT_RESIDUAL_ANCHOR_SEARCH_K,
        # v8.3 projected-space whole-image ACCEPT fallback. Appended to preserve
        # all existing positional-call meanings. In dual mode the evidence is
        # min(direction_advantage, magnitude_advantage), so both learned heads
        # must agree ACCEPT before this gate can fire.
        projected_image_accept_enable: bool = False,
        projected_image_accept_threshold: float = float("inf"),
        projected_image_accept_damping: float = 0.0,
    ) -> None:
        super().__init__(
            num_neighbours=num_neighbours,
            encoder_name=encoder_name,
            masking=masking,
            coreset_subsampling=coreset_subsampling,
            sampling_ratio=sampling_ratio,
        )

        if scoring_mode not in {"cosine", "pca"}:
            raise ValueError(f"Unsupported scoring_mode: {scoring_mode}")
        if not 0.0 <= accept_damping <= 1.0:
            raise ValueError("accept_damping must be in [0, 1]")
        if not 0.0 <= image_accept_damping <= 1.0:
            raise ValueError("image_accept_damping must be in [0, 1]")
        if not 0.0 <= projected_image_accept_damping <= 1.0:
            raise ValueError("projected_image_accept_damping must be in [0, 1]")
        if image_reject_boost_lambda < 0.0:
            raise ValueError("image_reject_boost_lambda must be >= 0")
        if image_reject_boost_max < 0.0:
            raise ValueError("image_reject_boost_max must be >= 0")

        self.scoring_mode = scoring_mode
        self.pca_components = pca_components
        self.residual_creation_threshold = residual_creation_threshold
        self.residual_topk_per_image = max(0, int(residual_topk_per_image))
        self.detector_only = detector_only
        self.strict_stock_when_damping_one = strict_stock_when_damping_one
        self.emit_patch_class = emit_patch_class
        self.residual_bank_chunk_size = max(1, int(residual_bank_chunk_size))
        self.normal_bank_chunk_size = max(1, int(normal_bank_chunk_size))
        # v8.2: use a wide fast-dtype candidate retrieval pool, then rerank only
        # those candidates with an explicit FP32 dot-product reduction. This keeps
        # the dominant full-bank search unchanged while making the final soft-anchor
        # membership far less sensitive to PT/ORT GEMM ranking noise.
        self.residual_anchor_k = max(1, int(residual_anchor_k))
        self.residual_anchor_search_k = max(
            self.residual_anchor_k, int(residual_anchor_search_k)
        )
        self.residual_anchor_temperature = float(V81_RESIDUAL_ANCHOR_TEMPERATURE)
        self.residual_anchor_rerank_candidate_chunk = int(V82_RESIDUAL_RERANK_CANDIDATE_CHUNK)
        self.calibration_query_chunk_size = max(1, int(calibration_query_chunk_size))
        self.calibration_seed = int(calibration_seed)

        # Residual direction + severity banks.
        self.register_buffer("accept_dir_bank", torch.empty(0))
        self.register_buffer("reject_dir_bank", torch.empty(0))
        self.register_buffer("accept_mag_bank", torch.empty(0))
        self.register_buffer("reject_mag_bank", torch.empty(0))

        # PCA parameters.  Means are required for mathematically correct PCA
        # reconstruction error; the previous implementation omitted them.
        self.register_buffer("accept_pca_basis", torch.empty(0))
        self.register_buffer("reject_pca_basis", torch.empty(0))
        self.register_buffer("accept_pca_mean", torch.empty(0))
        self.register_buffer("reject_pca_mean", torch.empty(0))

        # Tolerance thresholds.  Keep t_known for backward compatibility while
        # calibrating class-specific thresholds independently.
        accept_known = t_known if t_accept_known is None else t_accept_known
        reject_known = t_known if t_reject_known is None else t_reject_known
        self.register_buffer("t_normal", torch.tensor(float(t_normal)))
        self.register_buffer("t_known", torch.tensor(float(t_known)))
        self.register_buffer("t_accept_known", torch.tensor(float(accept_known)))
        self.register_buffer("t_reject_known", torch.tensor(float(reject_known)))
        self.register_buffer("margin", torch.tensor(float(margin)))
        self.register_buffer("accept_damping", torch.tensor(float(accept_damping)))
        self.image_accept_enable = bool(image_accept_enable)
        self.register_buffer("image_accept_threshold", torch.tensor(float(image_accept_threshold)))
        self.register_buffer("image_accept_damping", torch.tensor(float(image_accept_damping)))
        self.projected_image_accept_enable = bool(projected_image_accept_enable)
        self.register_buffer(
            "projected_image_accept_threshold",
            torch.tensor(float(projected_image_accept_threshold)),
        )
        self.register_buffer(
            "projected_image_accept_damping",
            torch.tensor(float(projected_image_accept_damping)),
        )
        self.image_reject_veto_enable = bool(image_reject_veto_enable)
        self.register_buffer("image_reject_veto_threshold", torch.tensor(float(image_reject_veto_threshold)))
        self.image_reject_boost_enable = bool(image_reject_boost_enable)
        self.register_buffer(
            "image_reject_boost_advantage_threshold",
            torch.tensor(float(image_reject_boost_advantage_threshold)),
        )
        self.register_buffer("image_reject_boost_lambda", torch.tensor(float(image_reject_boost_lambda)))
        self.register_buffer("image_reject_boost_max", torch.tensor(float(image_reject_boost_max)))

        # v6c projection state.  The learned matrix and projected banks are
        # buffers so they are exported with the detector; no optimizer is
        # needed at inference.  Training happens once from REFERENCE residuals.
        self.residual_projection_enable = bool(residual_projection_enable)
        self.residual_projection_dim = max(2, int(residual_projection_dim))
        self.residual_projection_margin = float(residual_projection_margin)
        self.residual_projection_epochs = max(1, int(residual_projection_epochs))
        self.residual_projection_steps_per_epoch = max(1, int(residual_projection_steps_per_epoch))
        self.residual_projection_batch_size = max(8, int(residual_projection_batch_size))
        self.residual_projection_lr = float(residual_projection_lr)
        self.residual_projection_weight_decay = float(residual_projection_weight_decay)
        self.residual_projection_orthogonality_weight = float(residual_projection_orthogonality_weight)
        self.residual_projection_use_magnitude = bool(residual_projection_use_magnitude)
        if residual_projection_magnitude_transform not in {"log", "linear"}:
            raise ValueError("residual_projection_magnitude_transform must be 'log' or 'linear'")
        self.residual_projection_magnitude_transform = str(residual_projection_magnitude_transform)
        self.residual_projection_magnitude_scale = float(residual_projection_magnitude_scale)
        if self.residual_projection_magnitude_scale < 0.0:
            raise ValueError("residual_projection_magnitude_scale must be >= 0")
        self.residual_projection_magnitude_clip = float(residual_projection_magnitude_clip)
        if self.residual_projection_magnitude_clip < 0.0:
            raise ValueError("residual_projection_magnitude_clip must be >= 0")
        self.residual_projection_magnitude_eps = max(float(residual_projection_magnitude_eps), 1.0e-12)
        # Reference-only normalization statistics for the optional magnitude channel.
        # Stored as buffers so exported inference uses exactly the training transform.
        self.register_buffer("residual_projection_magnitude_mean", torch.tensor(0.0))
        self.register_buffer("residual_projection_magnitude_std", torch.tensor(1.0))
        weights = residual_projection_reject_type_weights or {}
        self.residual_projection_reject_type_weights = {
            str(name): max(0.0, float(weight)) for name, weight in weights.items()
        }
        self.residual_projection_seed = int(residual_projection_seed)
        fit_device = str(residual_projection_fit_device).strip().lower()
        if fit_device not in {"cpu", "model"}:
            raise ValueError("residual_projection_fit_device must be 'cpu' or 'model'")
        self.residual_projection_fit_device = fit_device

        # Canonical projection-mode selector.  The legacy booleans remain accepted
        # only when residual_projection_mode is omitted, preserving the behavior
        # of the previous dual package (dual=True + use_magnitude=True -> dual;
        # every other legacy combination -> direction).
        if residual_projection_mode is None:
            if bool(dual_projection_enable) and bool(residual_projection_use_magnitude):
                resolved_projection_mode = "dual"
            else:
                resolved_projection_mode = "direction"
        else:
            resolved_projection_mode = str(residual_projection_mode).strip().lower()
        if resolved_projection_mode not in {"direction", "magnitude", "dual"}:
            raise ValueError(
                "residual_projection_mode must be one of: 'direction', 'magnitude', 'dual'"
            )
        # Runtime/export mode only. Both heads are fitted and calibrated regardless
        # of this selector so switching direction/magnitude/dual never retrains a
        # different projection.  Persist the selector as a state_dict buffer so
        # checkpoint reloads automatically recover the deployment mode.  Forward
        # still branches on Python attributes, not this tensor, which keeps ONNX
        # opset-14 export image-only and statically specialized to the saved mode.
        self.register_buffer(
            "residual_projection_mode_id",
            torch.tensor(
                RESIDUAL_PROJECTION_MODE_TO_ID[resolved_projection_mode],
                dtype=torch.int64,
            ),
        )
        self.residual_projection_mode = resolved_projection_mode
        self.direction_projection_enabled = resolved_projection_mode in {"direction", "dual"}
        self.magnitude_projection_enabled = resolved_projection_mode in {"magnitude", "dual"}
        # Derived legacy attributes retained for callers/metadata that still read them.
        self.dual_projection_enable = resolved_projection_mode == "dual"
        self.residual_projection_use_magnitude = self.magnitude_projection_enabled

        # Optional context channels shared by whichever projection head(s) are active.
        self.residual_projection_use_relative_xy = bool(
            self.residual_projection_enable and residual_projection_use_relative_xy
        )
        self.residual_projection_xy_scale = float(residual_projection_xy_scale)
        if self.residual_projection_xy_scale < 0.0:
            raise ValueError("residual_projection_xy_scale must be >= 0")

        intermediate_layers = tuple(sorted(set(int(v) for v in (residual_projection_intermediate_layers or ()))))
        self.residual_projection_final_layer_name, last_block = _resolve_final_layer_name(
            self.feature_encoder, encoder_name
        )
        invalid_layers = [v for v in intermediate_layers if v < 0 or v >= last_block]
        if invalid_layers:
            raise ValueError(
                "residual_projection_intermediate_layers must be 0-based transformer "
                f"block indices before the final block {last_block}; got {invalid_layers}"
            )
        self.residual_projection_intermediate_layers = intermediate_layers
        self.residual_projection_intermediate_layer_names = tuple(
            f"blocks.{idx}" for idx in self.residual_projection_intermediate_layers
        )
        self.residual_projection_intermediate_scale = float(residual_projection_intermediate_scale)
        if self.residual_projection_intermediate_scale < 0.0:
            raise ValueError("residual_projection_intermediate_scale must be >= 0")
        self.residual_projection_intermediate_enabled = bool(
            self.residual_projection_enable and self.residual_projection_intermediate_layer_names
        )
        if self.residual_projection_intermediate_enabled and coreset_subsampling:
            raise ValueError(
                "Intermediate residual features require coreset_subsampling=false so the "
                "parallel intermediate normal bank remains index-aligned with the stock normal bank."
            )

        # Anomalib 2.3.3 compatibility:
        # --------------------------------
        # AnomalyDINO 2.3.3 uses the custom DinoVisionTransformer returned by
        # DinoV2Loader. Its public get_intermediate_layers() method accepts an
        # explicit Sequence[int], applies the same final norm used by stock
        # AnomalyDINO, and removes CLS/register tokens before returning patch
        # tokens. Use that API directly rather than replacing feature_encoder
        # with TimmFeatureExtractor (whose transformer NLC/output_fmt API only
        # appears in newer Anomalib releases). This preserves one backbone pass
        # and stock final-feature semantics.
        #
        # TODO(anomalib-newer): if upgrading from 2.3.3, revisit this adapter.
        # Anomalib 2.5.1+ migrated DINO models to timm and exposes richer public
        # transformer feature-extraction options that may make this path simpler.
        self.residual_projection_final_block_index = int(last_block)

        self._cached_final_features: torch.Tensor | None = None
        self._cached_intermediate_features: torch.Tensor | None = None
        self.register_buffer("intermediate_memory_bank", torch.empty(0))
        self.intermediate_embedding_store: list[torch.Tensor] = []

        # Cross-backend stability constants.  These do not materially change
        # anomaly distances; they only impose a deterministic lexicographic
        # preference when PT and ORT see equal/nearly-equal kNN/top-evidence
        # scores.  This matters because the selected normal patch defines the
        # residual direction, so an arbitrary TopK tie can otherwise flip an
        # image-level ACCEPT decision and suppress/restore the whole anomaly map.
        self.register_buffer("normal_knn_tie_break_epsilon", torch.tensor(1.0e-6))
        self.register_buffer("image_topk_tie_break_epsilon", torch.tensor(1.0e-6))

        # Direction-only (v6b) and direction+magnitude (v6c) heads are kept as
        # independent exported buffers.  Both consume the same residuals and
        # reference banks, so inference adds only a second small 32-D projection
        # and projected-bank lookup; DINO and normal-bank work is shared.
        self.register_buffer("residual_projection_direction_weight", torch.empty(0))
        self.register_buffer("residual_projection_magnitude_weight", torch.empty(0))
        self.register_buffer("accept_direction_projected_bank", torch.empty(0))
        self.register_buffer("reject_direction_projected_bank", torch.empty(0))
        self.register_buffer("accept_magnitude_projected_bank", torch.empty(0))
        self.register_buffer("reject_magnitude_projected_bank", torch.empty(0))

        # Backward-compatible aliases for single-head inspection/export tooling.
        # In dual mode these mirror the magnitude-aware head.
        self.register_buffer("residual_projection_weight", torch.empty(0))
        self.register_buffer("accept_projected_bank", torch.empty(0))
        self.register_buffer("reject_projected_bank", torch.empty(0))

        self.projected_reject_boost_enable = bool(projected_reject_boost_enable)
        direction_a0 = projected_reject_boost_advantage_threshold if direction_projected_reject_boost_advantage_threshold is None else direction_projected_reject_boost_advantage_threshold
        direction_lam = projected_reject_boost_lambda if direction_projected_reject_boost_lambda is None else direction_projected_reject_boost_lambda
        magnitude_a0 = projected_reject_boost_advantage_threshold if magnitude_projected_reject_boost_advantage_threshold is None else magnitude_projected_reject_boost_advantage_threshold
        magnitude_lam = projected_reject_boost_lambda if magnitude_projected_reject_boost_lambda is None else magnitude_projected_reject_boost_lambda
        self.register_buffer("direction_projected_reject_boost_advantage_threshold", torch.tensor(float(direction_a0)))
        self.register_buffer("direction_projected_reject_boost_lambda", torch.tensor(float(direction_lam)))
        self.register_buffer("magnitude_projected_reject_boost_advantage_threshold", torch.tensor(float(magnitude_a0)))
        self.register_buffer("magnitude_projected_reject_boost_lambda", torch.tensor(float(magnitude_lam)))
        # Dual-fusion safety masks are separate from each head's standalone
        # calibrated lambda. A dual fallback can disable one branch without
        # destroying the exact single-head calibration.
        self.register_buffer("dual_projection_direction_fusion_scale", torch.tensor(1.0))
        self.register_buffer("dual_projection_magnitude_fusion_scale", torch.tensor(1.0))
        # Legacy projected_* fields mirror magnitude head parameters.
        self.register_buffer("projected_reject_boost_advantage_threshold", torch.tensor(float(magnitude_a0)))
        self.register_buffer("projected_reject_boost_lambda", torch.tensor(float(magnitude_lam)))
        self.register_buffer("projected_reject_boost_max", torch.tensor(float(projected_reject_boost_max)))

        self.accept_residual_store: list[torch.Tensor] = []
        self.reject_residual_store: list[torch.Tensor] = []
        self.accept_magnitude_store: list[torch.Tensor] = []
        self.reject_magnitude_store: list[torch.Tensor] = []
        self.accept_xy_store: list[torch.Tensor] = []
        self.reject_xy_store: list[torch.Tensor] = []
        self.accept_intermediate_store: list[torch.Tensor] = []
        self.reject_intermediate_store: list[torch.Tensor] = []
        self.register_buffer("accept_xy_bank", torch.empty(0))
        self.register_buffer("reject_xy_bank", torch.empty(0))
        self.register_buffer("accept_intermediate_dir_bank", torch.empty(0))
        self.register_buffer("reject_intermediate_dir_bank", torch.empty(0))
        # Python-only grouping used to balance projection training by reject type.
        # It is discarded after fitting and is never consulted at inference.
        self.reject_projection_store: dict[str, list[torch.Tensor]] = {}
        self.reject_projection_magnitude_store: dict[str, list[torch.Tensor]] = {}
        self.reject_projection_xy_store: dict[str, list[torch.Tensor]] = {}
        self.reject_projection_intermediate_store: dict[str, list[torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Normal detector helpers
    # ------------------------------------------------------------------

    def extract_features_with_context(
        self, image_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract final and optional intermediate DINO patch tokens in one pass.

        On Anomalib 2.3.3 the stock detector uses
        ``feature_encoder.get_intermediate_layers(image, n=1)[0]``. When
        intermediate context is enabled, request the configured block indices
        plus the final block in a single call. ``norm=True`` and
        ``return_class_token=False`` match stock patch-token semantics.
        """
        if not self.residual_projection_intermediate_enabled:
            return super().extract_features(image_tensor), None

        get_layers = getattr(self.feature_encoder, "get_intermediate_layers", None)
        if get_layers is None or not callable(get_layers):
            raise RuntimeError(
                "Intermediate residual context is configured for the Anomalib 2.3.3 "
                "DinoVisionTransformer API, but feature_encoder has no callable "
                "get_intermediate_layers(). If Anomalib was upgraded, use the newer "
                "public transformer feature-extraction API or disable "
                "residual_projection_intermediate_layers."
            )

        requested = [
            *self.residual_projection_intermediate_layers,
            self.residual_projection_final_block_index,
        ]
        # intermediate_layers is sorted and validated to be before the final
        # block, so get_intermediate_layers returns outputs in this same order.
        # The DINO backbone is frozen. no_grad() avoids retaining transformer
        # activations during Lightning training but, unlike inference_mode(),
        # returns ordinary tensors that torch.onnx.export can trace correctly.
        with torch.no_grad():
            outputs = get_layers(
                image_tensor,
                n=requested,
                reshape=False,
                return_class_token=False,
                norm=True,
            )
        if len(outputs) != len(requested):
            raise RuntimeError(
                "DINO intermediate extraction returned an unexpected number of layers: "
                f"requested={requested} returned={len(outputs)}"
            )

        final = outputs[-1]
        if final.ndim != 3:
            raise RuntimeError(
                f"Expected final DINO patch tokens [B,N,D], got {tuple(final.shape)}"
            )
        intermediate_outputs = list(outputs[:-1])
        for layer_idx, tensor in zip(
            self.residual_projection_intermediate_layers,
            intermediate_outputs,
            strict=True,
        ):
            if tensor.ndim != 3 or tensor.shape[:2] != final.shape[:2]:
                raise RuntimeError(
                    "Intermediate/final DINO token mismatch for block "
                    f"{layer_idx}: intermediate={tuple(tensor.shape)} "
                    f"final={tuple(final.shape)}"
                )

        # [B,N,L,D]. Keep layer identity explicit so each residual is normalized
        # independently before concatenation into the learned projection input.
        intermediate = torch.stack(intermediate_outputs, dim=2)
        return final, intermediate

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Stock-compatible final feature API with training-time context cache."""
        final, intermediate = self.extract_features_with_context(image_tensor)
        self._cached_final_features = final
        self._cached_intermediate_features = intermediate
        return final

    def _grid_size_from_input(self, input_tensor: torch.Tensor) -> tuple[int, int]:
        patch_size = int(self.feature_encoder.patch_size)
        h, w = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
        return ((h - (h % patch_size)) // patch_size, (w - (w % patch_size)) // patch_size)

    @staticmethod
    def _relative_xy_from_patch_indices(
        patch_idx: torch.Tensor,
        grid_size: tuple[int, int],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Patch-center XY normalized relative to the cropped image grid.

        Returns ``[Q,2]`` in ``[-1,1]`` ordered as ``(x,y)``.  This is image/ROI
        relative rather than an absolute pixel coordinate, so the feature is
        stable across input resizing.
        """
        gh, gw = int(grid_size[0]), int(grid_size[1])
        if gh <= 0 or gw <= 0:
            raise ValueError(f"Invalid grid_size={grid_size}")
        idx = patch_idx.to(torch.long)
        y = torch.div(idx, gw, rounding_mode="floor").to(dtype)
        x = (idx % gw).to(dtype)
        x = ((x + 0.5) / float(gw)) * 2.0 - 1.0
        y = ((y + 0.5) / float(gh)) * 2.0 - 1.0
        return torch.stack((x, y), dim=1)

    def _collect_training_intermediate(self, original_input: torch.Tensor) -> None:
        """Collect normal intermediate features aligned to stock memory-bank rows."""
        if not self.residual_projection_intermediate_enabled:
            return
        final = self._cached_final_features
        intermediate = self._cached_intermediate_features
        if final is None or intermediate is None:
            raise RuntimeError("Intermediate feature cache missing during normal-bank training")

        grid_size = self._grid_size_from_input(original_input)
        if grid_size[0] * grid_size[1] != final.shape[1]:
            raise RuntimeError(
                f"Intermediate grid mismatch: grid={grid_size} tokens={final.shape[1]}"
            )
        if self.masking:
            masks_np = self.compute_background_masks(final.detach().cpu().numpy(), grid_size)
            masks = torch.from_numpy(masks_np).to(final.device)
        else:
            masks = torch.ones(final.shape[:2], dtype=torch.bool, device=final.device)

        selected = intermediate[masks]
        selected = F.normalize(selected, p=2, dim=-1)
        self.intermediate_embedding_store.append(selected.detach())

    def fit(self) -> None:
        """Finalize stock normal bank and its optional aligned intermediate bank."""
        super().fit()
        if not self.residual_projection_intermediate_enabled:
            self.intermediate_memory_bank = torch.empty(
                0, device=self.memory_bank.device, dtype=self.memory_bank.dtype
            )
            return
        if not self.intermediate_embedding_store:
            raise RuntimeError(
                "Intermediate projection layers are enabled but no normal intermediate embeddings were collected"
            )
        bank = torch.cat(self.intermediate_embedding_store, dim=0).to(
            device=self.memory_bank.device, dtype=self.memory_bank.dtype
        )
        self.intermediate_embedding_store.clear()
        if bank.shape[0] != self.memory_bank.shape[0]:
            raise RuntimeError(
                "Normal/intermediate memory-bank alignment failure: "
                f"final={self.memory_bank.shape[0]} intermediate={bank.shape[0]}"
            )
        self.intermediate_memory_bank = bank
        logger.info(
            "Intermediate normal bank: layers=%s shape=%s dtype=%s",
            self.residual_projection_intermediate_layers,
            tuple(bank.shape),
            bank.dtype,
        )

    def _normal_knn(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return v8.2 soft normal residual anchor plus stock-equivalent distance.

        The full normal-bank search remains in the detector/memory-bank dtype.
        ``d_n`` is still computed directly from that search using exactly
        ``num_neighbours`` entries, preserving stock AnomalyDINO scoring semantics.

        Tolerance residual anchoring is deliberately decoupled from ``d_n``:
        1. retain a wider fast-dtype candidate pool (128 by default),
        2. gather only those candidate normal vectors,
        3. recompute query-candidate cosine distances explicitly in FP32,
        4. deterministically select the best soft-anchor neighbours (32 by default),
        5. build an FP32 softmax-weighted centroid.

        The FP32 rerank is candidate-chunked so training/diagnostics never need to
        materialize a huge [Q,128,D] tensor. The same selected indices/weights are
        reused for intermediate-layer anchors.
        """
        if self.memory_bank.numel() == 0:
            raise RuntimeError("Memory bank is empty. Fit the normal bank first.")

        # Preserve the highest-precision query available for local reranking before
        # matching the full-bank search dtype. In exported float32 models these are
        # identical; during FP16 training this avoids needlessly quantizing rerank input.
        features_fp32 = features.float()
        if features.dtype != self.memory_bank.dtype:
            features = features.to(self.memory_bank.dtype)

        bank_rows = int(self.memory_bank.shape[0])
        score_k = min(max(1, int(self.num_neighbours)), bank_rows)
        anchor_k = min(max(1, int(self.residual_anchor_k)), bank_rows)
        # Backward compatibility for full-module checkpoints created before v8.2.
        # Plain Python attributes are not restored by __init__ when an entire module
        # object is unpickled, so older v8/v8.1 objects may not have
        # residual_anchor_search_k. Falling back to anchor_k preserves their original
        # candidate-pool semantics while fresh v8.2 instances still use the configured
        # wider search pool (128 by default). This is a static Python attribute lookup,
        # so it does not introduce tensor-dependent control flow into ONNX export.
        residual_anchor_search_k = int(
            getattr(self, "residual_anchor_search_k", self.residual_anchor_k)
        )
        candidate_k = min(
            max(score_k, anchor_k, residual_anchor_search_k),
            bank_rows,
        )

        best_dist = torch.full(
            (features.shape[0], candidate_k),
            float("inf"),
            device=features.device,
            dtype=features.dtype,
        )
        best_rank = torch.full(
            (features.shape[0], candidate_k),
            float("inf"),
            device=features.device,
            dtype=torch.float32,
        )
        best_idx = torch.full(
            (features.shape[0], candidate_k),
            -1,
            device=features.device,
            dtype=torch.long,
        )

        bank_size = max(bank_rows, 1)
        tie_eps = self.normal_knn_tie_break_epsilon.to(device=features.device, dtype=torch.float32)

        # Stage 1: unchanged full-bank search math in the detector's existing dtype.
        for start in range(0, bank_rows, self.normal_bank_chunk_size):
            bank_chunk = self.memory_bank[start : start + self.normal_bank_chunk_size]
            sim = torch.matmul(features, bank_chunk.T)
            dist = (1.0 - sim).clamp(0.0, 2.0)
            local_k = min(candidate_k, int(bank_chunk.shape[0]))

            global_idx = torch.arange(
                start, start + bank_chunk.shape[0], device=features.device, dtype=torch.long
            )
            tie = global_idx.to(torch.float32) / float(bank_size) * tie_eps
            rank_dist = dist.float() + tie.unsqueeze(0)
            local_rank, local_pos = torch.topk(rank_dist, k=local_k, dim=1, largest=False)
            local_dist = torch.gather(dist, 1, local_pos)
            local_idx = global_idx.index_select(0, local_pos.reshape(-1)).reshape_as(local_pos)

            merged_rank = torch.cat((best_rank, local_rank), dim=1)
            merged_dist = torch.cat((best_dist, local_dist), dim=1)
            merged_idx = torch.cat((best_idx, local_idx), dim=1)
            keep = torch.topk(merged_rank, k=candidate_k, dim=1, largest=False).indices
            best_rank = torch.gather(merged_rank, 1, keep)
            best_dist = torch.gather(merged_dist, 1, keep)
            best_idx = torch.gather(merged_idx, 1, keep)

        # Preserve stock detector score from the original fast-dtype search.
        stock_dist = best_dist[:, :score_k]
        d_n = stock_dist.mean(dim=1) if score_k > 1 else stock_dist.squeeze(1)

        # Stage 2: local FP32 reranking over only the retained candidate pool.
        # Candidate-dimension chunking keeps peak memory bounded and is independent
        # of runtime query count/batch size, which is friendlier to ONNX tracing.
        rerank_dist_chunks: list[torch.Tensor] = []
        rerank_chunk = max(1, int(self.residual_anchor_rerank_candidate_chunk))
        for c_start in range(0, candidate_k, rerank_chunk):
            c_end = min(c_start + rerank_chunk, candidate_k)
            idx_chunk = best_idx[:, c_start:c_end]
            flat_idx = idx_chunk.reshape(-1)
            candidate_features = self.memory_bank.index_select(0, flat_idx).reshape(
                features.shape[0], c_end - c_start, self.memory_bank.shape[1]
            ).float()
            # Explicit elementwise multiply+reduce avoids relying on another large
            # GEMM kernel for the decision-critical local rerank.
            sim32 = (features_fp32.unsqueeze(1) * candidate_features).sum(dim=2)
            rerank_dist_chunks.append((1.0 - sim32).clamp(0.0, 2.0))

        rerank_dist = torch.cat(rerank_dist_chunks, dim=1)
        rerank_tie = best_idx.to(torch.float32) / float(bank_size) * tie_eps
        rerank_rank = rerank_dist + rerank_tie
        anchor_pos = torch.topk(rerank_rank, k=anchor_k, dim=1, largest=False).indices
        anchor_indices = torch.gather(best_idx, 1, anchor_pos)
        anchor_dist = torch.gather(rerank_dist, 1, anchor_pos)

        anchor_relative_dist = anchor_dist - anchor_dist.min(dim=1, keepdim=True).values
        anchor_temperature = max(float(self.residual_anchor_temperature), 1.0e-6)
        anchor_weights = torch.softmax(-anchor_relative_dist / anchor_temperature, dim=1)

        flat_anchor_idx = anchor_indices.reshape(-1)
        anchor_features = self.memory_bank.index_select(0, flat_anchor_idx).reshape(
            features.shape[0], anchor_k, self.memory_bank.shape[1]
        )
        anchor = (anchor_features.float() * anchor_weights.unsqueeze(-1)).sum(dim=1)
        anchor = F.normalize(anchor, p=2, dim=1)
        return anchor, d_n, anchor_indices, anchor_weights

    def sample_normal_self_distances(self, max_samples: int = 4096) -> torch.Tensor:
        """Leave-one-out normal kNN distances using the detector's same k.

        The old implementation built a potentially multi-GB 4096 x full-bank
        similarity matrix and calibrated 1-NN even when inference used k>1.
        This version is chunked and masks the exact sampled bank index.
        """
        bank = self.memory_bank
        n = bank.shape[0]
        if n < 2:
            return torch.empty(0, device=bank.device, dtype=torch.float32)

        count = min(int(max_samples), n)
        generator = torch.Generator(device=bank.device)
        generator.manual_seed(self.calibration_seed)
        sample_idx = torch.randperm(n, generator=generator, device=bank.device)[:count]
        k = min(max(1, int(self.num_neighbours)), n - 1)
        results: list[torch.Tensor] = []

        for start in range(0, count, self.calibration_query_chunk_size):
            idx = sample_idx[start : start + self.calibration_query_chunk_size]
            query = bank[idx]
            sim = torch.matmul(query, bank.T)
            # Mask the exact self entry rather than whichever duplicate happens
            # to be returned by argmax.
            rows = torch.arange(idx.numel(), device=bank.device)
            sim[rows, idx] = -2.0
            dists = (1.0 - sim).clamp(0.0, 2.0)
            vals = torch.topk(dists, k=k, dim=1, largest=False).values
            results.append(vals.mean(dim=1).float())

        return torch.cat(results, dim=0)

    # ------------------------------------------------------------------
    # Residual-bank construction
    # ------------------------------------------------------------------

    def add_defect_features(
        self,
        features: torch.Tensor,
        label: Literal["accept", "reject"],
        reject_types: list[str] | tuple[str, ...] | None = None,
        intermediate_features: torch.Tensor | None = None,
        grid_size: tuple[int, int] | None = None,
    ) -> None:
        """Add defect residuals and optional projection context.

        ``features`` are final-layer DINO tokens ``[B,N,D]``.  When configured,
        ``intermediate_features`` must be ``[B,N,L,D]`` from the same single
        encoder pass. v8.2 intermediate residuals use the SAME soft-anchor normal
        neighborhood and weights selected in final-layer space, avoiding a second
        expensive normal-bank search while removing single-anchor instability.
        """
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B,N,D] or [N,D], got {tuple(features.shape)}")
        if self.residual_projection_intermediate_enabled:
            if intermediate_features is None:
                raise ValueError("Intermediate projection layers are enabled but intermediate_features were not provided")
            if intermediate_features.ndim == 3:
                intermediate_features = intermediate_features.unsqueeze(0)
            if intermediate_features.ndim != 4:
                raise ValueError(
                    f"Expected intermediate_features [B,N,L,D], got {tuple(intermediate_features.shape)}"
                )
            if intermediate_features.shape[:2] != features.shape[:2]:
                raise ValueError(
                    "Final/intermediate defect feature shape mismatch: "
                    f"final={tuple(features.shape)} intermediate={tuple(intermediate_features.shape)}"
                )

        dir_store = self.accept_residual_store if label == "accept" else self.reject_residual_store
        mag_store = self.accept_magnitude_store if label == "accept" else self.reject_magnitude_store
        xy_store = self.accept_xy_store if label == "accept" else self.reject_xy_store
        intermediate_store = self.accept_intermediate_store if label == "accept" else self.reject_intermediate_store

        for image_i, image_features in enumerate(features):
            # Keep the original feature precision for v8.2 local FP32 reranking;
            # _normal_knn() casts a separate copy for the full-bank detector search.
            image_features = F.normalize(
                image_features.to(self.memory_bank.device),
                p=2,
                dim=1,
            )
            normal_anchor, d_n, anchor_indices, anchor_weights = self._normal_knn(image_features)
            candidate_idx = torch.nonzero(d_n > self.residual_creation_threshold, as_tuple=False).squeeze(1)
            if candidate_idx.numel() == 0:
                continue

            if self.residual_topk_per_image > 0 and candidate_idx.numel() > self.residual_topk_per_image:
                local_scores = d_n[candidate_idx]
                keep = torch.topk(
                    local_scores,
                    k=self.residual_topk_per_image,
                    largest=True,
                ).indices
                candidate_idx = candidate_idx[keep]

            raw_residual = image_features[candidate_idx].float() - normal_anchor[candidate_idx]
            magnitude = raw_residual.norm(dim=1)
            valid = magnitude > torch.finfo(raw_residual.dtype).eps
            if not valid.any():
                continue

            selected_idx = candidate_idx[valid]
            direction = F.normalize(raw_residual[valid], p=2, dim=1)
            selected_magnitude = magnitude[valid]
            dir_store.append(direction.detach())
            mag_store.append(selected_magnitude.detach())

            selected_xy: torch.Tensor | None = None
            if self.residual_projection_use_relative_xy:
                if grid_size is None:
                    side = int(round(math.sqrt(image_features.shape[0])))
                    if side * side != image_features.shape[0]:
                        raise ValueError(
                            "grid_size is required for relative XY when patch tokens are not square"
                        )
                    active_grid = (side, side)
                else:
                    active_grid = grid_size
                selected_xy = self._relative_xy_from_patch_indices(
                    selected_idx, active_grid, dtype=torch.float32
                ).to(self.memory_bank.device)
                xy_store.append(selected_xy.detach())

            selected_intermediate: torch.Tensor | None = None
            if self.residual_projection_intermediate_enabled:
                if self.intermediate_memory_bank.numel() == 0:
                    raise RuntimeError("Intermediate normal memory bank is empty")
                assert intermediate_features is not None
                image_intermediate = intermediate_features[image_i].to(
                    self.memory_bank.device, self.memory_bank.dtype
                )
                image_intermediate = F.normalize(image_intermediate, p=2, dim=-1)
                selected_anchor_idx = anchor_indices.index_select(0, selected_idx)
                selected_anchor_weights = anchor_weights.index_select(0, selected_idx)
                anchor_k = selected_anchor_idx.shape[1]
                flat_anchor_idx = selected_anchor_idx.reshape(-1)
                normal_intermediate_neighbors = self.intermediate_memory_bank.index_select(
                    0, flat_anchor_idx
                ).reshape(
                    selected_idx.shape[0],
                    anchor_k,
                    self.intermediate_memory_bank.shape[1],
                    self.intermediate_memory_bank.shape[2],
                )
                intermediate_anchor = (
                    normal_intermediate_neighbors.float()
                    * selected_anchor_weights[:, :, None, None]
                ).sum(dim=1)
                intermediate_anchor = F.normalize(intermediate_anchor, p=2, dim=-1)
                raw_intermediate = image_intermediate[selected_idx].float() - intermediate_anchor
                # Normalize each layer residual independently so one layer cannot
                # dominate merely because of activation scale, then concatenate.
                selected_intermediate = F.normalize(
                    raw_intermediate, p=2, dim=-1
                ).flatten(start_dim=1)
                intermediate_store.append(selected_intermediate.detach())

            if label == "reject" and self.residual_projection_enable:
                reject_type = ""
                if reject_types is not None and image_i < len(reject_types):
                    reject_type = str(reject_types[image_i])
                self.reject_projection_store.setdefault(reject_type, []).append(direction.detach())
                self.reject_projection_magnitude_store.setdefault(reject_type, []).append(selected_magnitude.detach())
                if selected_xy is not None:
                    self.reject_projection_xy_store.setdefault(reject_type, []).append(selected_xy.detach())
                if selected_intermediate is not None:
                    self.reject_projection_intermediate_store.setdefault(reject_type, []).append(
                        selected_intermediate.detach()
                    )

    def finalize_residual_banks(self) -> None:
        """Finalize residual banks and PCA parameters."""
        target_device = self.memory_bank.device
        target_dtype = self.memory_bank.dtype

        if self.accept_residual_store:
            self.accept_dir_bank = torch.vstack(self.accept_residual_store).to(target_device, target_dtype)
            self.accept_mag_bank = torch.cat(self.accept_magnitude_store).to(target_device, target_dtype)
            self.accept_residual_store.clear()
            self.accept_magnitude_store.clear()
            logger.info("Accept residual bank: %s", tuple(self.accept_dir_bank.shape))
        else:
            logger.warning("No acceptable-defect residuals collected.")

        if self.reject_residual_store:
            self.reject_dir_bank = torch.vstack(self.reject_residual_store).to(target_device, target_dtype)
            self.reject_mag_bank = torch.cat(self.reject_magnitude_store).to(target_device, target_dtype)
            self.reject_residual_store.clear()
            self.reject_magnitude_store.clear()
            logger.info("Reject residual bank: %s", tuple(self.reject_dir_bank.shape))
        else:
            logger.warning("No reject-defect residuals collected.")

        def _finalize_optional_store(
            store: list[torch.Tensor],
            expected_rows: int,
            width: int,
            name: str,
        ) -> torch.Tensor:
            if not store:
                return torch.empty((0, width), device=target_device, dtype=torch.float32)
            bank = torch.cat(store, dim=0).to(target_device, torch.float32)
            store.clear()
            if bank.shape[0] != expected_rows:
                raise RuntimeError(
                    f"{name} row alignment mismatch: context={bank.shape[0]} residual={expected_rows}"
                )
            return bank

        if self.residual_projection_use_relative_xy:
            self.accept_xy_bank = _finalize_optional_store(
                self.accept_xy_store, int(self.accept_dir_bank.shape[0]), 2, "accept_xy_bank"
            )
            self.reject_xy_bank = _finalize_optional_store(
                self.reject_xy_store, int(self.reject_dir_bank.shape[0]), 2, "reject_xy_bank"
            )
        else:
            self.accept_xy_store.clear()
            self.reject_xy_store.clear()
            self.accept_xy_bank = torch.empty((0, 2), device=target_device, dtype=torch.float32)
            self.reject_xy_bank = torch.empty((0, 2), device=target_device, dtype=torch.float32)

        if self.residual_projection_intermediate_enabled:
            inter_width = (
                int(self.intermediate_memory_bank.shape[1] * self.intermediate_memory_bank.shape[2])
                if self.intermediate_memory_bank.ndim == 3 else 0
            )
            self.accept_intermediate_dir_bank = _finalize_optional_store(
                self.accept_intermediate_store, int(self.accept_dir_bank.shape[0]), inter_width,
                "accept_intermediate_dir_bank",
            )
            self.reject_intermediate_dir_bank = _finalize_optional_store(
                self.reject_intermediate_store, int(self.reject_dir_bank.shape[0]), inter_width,
                "reject_intermediate_dir_bank",
            )
        else:
            self.accept_intermediate_store.clear()
            self.reject_intermediate_store.clear()
            self.accept_intermediate_dir_bank = torch.empty((0, 0), device=target_device, dtype=torch.float32)
            self.reject_intermediate_dir_bank = torch.empty((0, 0), device=target_device, dtype=torch.float32)

        if self.scoring_mode == "pca":
            self.accept_pca_basis, self.accept_pca_mean = self._fit_pca(
                self.accept_dir_bank,
                self.pca_components,
            )
            self.reject_pca_basis, self.reject_pca_mean = self._fit_pca(
                self.reject_dir_bank,
                self.pca_components,
            )

    @staticmethod
    def _fit_pca(bank: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor]:
        if bank.numel() == 0:
            empty = torch.empty(0, device=bank.device, dtype=bank.dtype)
            return empty, empty

        # SVD in float32 is both more stable and works when the model itself is FP16.
        bank_f = bank.float()
        mean = bank_f.mean(dim=0, keepdim=True)
        centered = bank_f - mean
        k = min(max(1, int(n_components)), bank.shape[0], bank.shape[1])
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:k]
        return basis.to(bank.dtype), mean.to(bank.dtype)

    # ------------------------------------------------------------------
    # v6c supervised residual projection
    # ------------------------------------------------------------------

    def _transform_projection_magnitude(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """Transform raw residual magnitudes before reference-only standardization."""
        m = magnitudes.float().reshape(-1)
        if self.residual_projection_magnitude_transform == "log":
            m = torch.log(m.clamp_min(self.residual_projection_magnitude_eps))
        return m

    def _apply_residual_projection_mode(self, mode: str, *, persist: bool) -> None:
        """Apply a deployment mode to Python execution flags.

        ``persist=True`` also updates the small state_dict buffer.  This helper is
        never called from ``forward``; inference/export therefore sees a fixed
        Python branch and the mode is not exposed as an ONNX runtime input.
        """
        resolved = str(mode).strip().lower()
        if resolved not in RESIDUAL_PROJECTION_MODE_TO_ID:
            raise ValueError("residual projection mode must be direction, magnitude, or dual")
        self.residual_projection_mode = resolved
        self.direction_projection_enabled = resolved in {"direction", "dual"}
        self.magnitude_projection_enabled = resolved in {"magnitude", "dual"}
        self.dual_projection_enable = resolved == "dual"
        self.residual_projection_use_magnitude = self.magnitude_projection_enabled
        if persist and hasattr(self, "residual_projection_mode_id"):
            with torch.no_grad():
                self.residual_projection_mode_id.fill_(
                    RESIDUAL_PROJECTION_MODE_TO_ID[resolved]
                )

    def _restore_residual_projection_mode_from_state(self) -> None:
        """Restore Python execution flags from the persisted mode buffer."""
        mode_id = int(self.residual_projection_mode_id.detach().cpu().item())
        mode = RESIDUAL_PROJECTION_ID_TO_MODE.get(mode_id)
        if mode is None:
            raise RuntimeError(
                f"Invalid persisted residual_projection_mode_id={mode_id}; expected 0, 1, or 2"
            )
        self._apply_residual_projection_mode(mode, persist=False)

    def set_residual_projection_mode(self, mode: str) -> None:
        """Set and persist the inference/export mode without refitting heads.

        This is a one-time artifact configuration operation, not an inference
        argument.  Saving the model/checkpoint afterward preserves the selection.
        """
        self._apply_residual_projection_mode(mode, persist=True)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load buffers then restore the persisted deployment mode.

        Older v7 checkpoints predate ``residual_projection_mode_id``.  They remain
        loadable: when that key is absent, the constructor/YAML mode is retained.
        New checkpoints restore the saved mode automatically.
        """
        mode_key = prefix + "residual_projection_mode_id"
        has_persisted_mode = mode_key in state_dict
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if not has_persisted_mode:
            # ``super`` records the new buffer as missing in strict mode. Remove
            # only this compatibility key; all other strict-load diagnostics stay.
            try:
                missing_keys.remove(mode_key)
            except ValueError:
                pass
            logger.warning(
                "Checkpoint has no residual_projection_mode_id; retaining constructor mode=%s",
                self.residual_projection_mode,
            )
            self._apply_residual_projection_mode(
                self.residual_projection_mode, persist=True
            )
            return
        self._restore_residual_projection_mode_from_state()

    def _fit_projection_magnitude_stats(self) -> None:
        """Fit reference-only magnitude normalization deterministically."""
        fit_device = (
            torch.device("cpu")
            if self.residual_projection_fit_device == "cpu"
            else self.memory_bank.device
        )
        parts: list[torch.Tensor] = []
        if self.accept_mag_bank.numel() > 0:
            parts.append(self.accept_mag_bank.detach().to(fit_device, torch.float32))
        if self.reject_mag_bank.numel() > 0:
            parts.append(self.reject_mag_bank.detach().to(fit_device, torch.float32))
        if not parts:
            self.residual_projection_magnitude_mean = torch.tensor(
                0.0, device=self.memory_bank.device, dtype=torch.float32
            )
            self.residual_projection_magnitude_std = torch.tensor(
                1.0, device=self.memory_bank.device, dtype=torch.float32
            )
            return
        values = self._transform_projection_magnitude(torch.cat(parts, dim=0))
        mean = values.mean()
        std = values.std(unbiased=False).clamp_min(1.0e-6)
        self.residual_projection_magnitude_mean = mean.detach().to(
            device=self.memory_bank.device, dtype=torch.float32
        )
        self.residual_projection_magnitude_std = std.detach().to(
            device=self.memory_bank.device, dtype=torch.float32
        )
        logger.info(
            "Residual projection magnitude stats: transform=%s mean=%.6f std=%.6f scale=%.3f clip=%.3f fit_device=%s",
            self.residual_projection_magnitude_transform,
            float(mean.detach().cpu()),
            float(std.detach().cpu()),
            self.residual_projection_magnitude_scale,
            self.residual_projection_magnitude_clip,
            fit_device,
        )

    def _projection_features(
        self,
        residual_dirs: torch.Tensor,
        residual_magnitudes: torch.Tensor | None,
        relative_xy: torch.Tensor | None = None,
        intermediate_dirs: torch.Tensor | None = None,
        *,
        include_magnitude: bool | None = None,
    ) -> torch.Tensor:
        """Build learned-projection input from residual direction + optional context.

        Context channels are deliberately appended rather than altering the stock
        anomaly metric:
        * intermediate residual directions from selected DINO blocks,
        * image-relative patch-center ``(x,y)``,
        * optional standardized final-residual magnitude (magnitude head only).
        """
        direction = F.normalize(residual_dirs.float(), p=2, dim=1)
        parts: list[torch.Tensor] = [direction]

        if self.residual_projection_intermediate_enabled:
            if intermediate_dirs is None:
                raise ValueError("Intermediate projection context is enabled but intermediate_dirs are missing")
            if intermediate_dirs.shape[0] != direction.shape[0]:
                raise ValueError(
                    f"Intermediate row mismatch: {intermediate_dirs.shape[0]} vs {direction.shape[0]}"
                )
            parts.append(
                intermediate_dirs.to(direction.device, torch.float32)
                * self.residual_projection_intermediate_scale
            )

        if self.residual_projection_use_relative_xy:
            if relative_xy is None:
                raise ValueError("Relative XY projection context is enabled but relative_xy is missing")
            if relative_xy.shape != (direction.shape[0], 2):
                raise ValueError(
                    f"Expected relative_xy [{direction.shape[0]},2], got {tuple(relative_xy.shape)}"
                )
            parts.append(
                relative_xy.to(direction.device, torch.float32)
                * self.residual_projection_xy_scale
            )

        use_magnitude = self.residual_projection_use_magnitude if include_magnitude is None else bool(include_magnitude)
        if use_magnitude:
            if residual_magnitudes is None:
                raise ValueError("Magnitude-aware projection requires residual magnitudes")
            magnitude = self._transform_projection_magnitude(residual_magnitudes)
            mean = self.residual_projection_magnitude_mean.to(device=direction.device, dtype=torch.float32)
            std = self.residual_projection_magnitude_std.to(device=direction.device, dtype=torch.float32).clamp_min(1.0e-6)
            magnitude = (magnitude.to(direction.device) - mean) / std
            clip = self.residual_projection_magnitude_clip
            if clip > 0.0:
                magnitude = torch.clamp(magnitude, min=-clip, max=clip)
            magnitude = magnitude * self.residual_projection_magnitude_scale
            parts.append(magnitude[:, None])

        return torch.cat(parts, dim=1)

    def _projection_encode_features(
        self,
        features: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if weight.numel() == 0:
            return torch.empty((features.shape[0], 0), device=features.device, dtype=features.dtype)
        w = weight.to(device=features.device, dtype=torch.float32)
        z = features.float() @ w.T
        return F.normalize(z, p=2, dim=1)

    def _projection_encode_head(
        self,
        residual_dirs: torch.Tensor,
        residual_magnitudes: torch.Tensor | None,
        relative_xy: torch.Tensor | None = None,
        intermediate_dirs: torch.Tensor | None = None,
        *,
        include_magnitude: bool,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        features = self._projection_features(
            residual_dirs,
            residual_magnitudes,
            relative_xy=relative_xy,
            intermediate_dirs=intermediate_dirs,
            include_magnitude=include_magnitude,
        )
        return self._projection_encode_features(features, weight)

    def _refresh_projected_banks(self) -> None:
        device = self.memory_bank.device
        with torch.no_grad():
            if self.residual_projection_direction_weight.numel() > 0:
                self.accept_direction_projected_bank = self._projection_encode_head(
                    self.accept_dir_bank, self.accept_mag_bank,
                    relative_xy=self.accept_xy_bank,
                    intermediate_dirs=self.accept_intermediate_dir_bank,
                    include_magnitude=False,
                    weight=self.residual_projection_direction_weight,
                )
                self.reject_direction_projected_bank = self._projection_encode_head(
                    self.reject_dir_bank, self.reject_mag_bank,
                    relative_xy=self.reject_xy_bank,
                    intermediate_dirs=self.reject_intermediate_dir_bank,
                    include_magnitude=False,
                    weight=self.residual_projection_direction_weight,
                )
            else:
                self.accept_direction_projected_bank = torch.empty(0, device=device, dtype=torch.float32)
                self.reject_direction_projected_bank = torch.empty(0, device=device, dtype=torch.float32)

            if self.residual_projection_magnitude_weight.numel() > 0:
                self.accept_magnitude_projected_bank = self._projection_encode_head(
                    self.accept_dir_bank, self.accept_mag_bank,
                    relative_xy=self.accept_xy_bank,
                    intermediate_dirs=self.accept_intermediate_dir_bank,
                    include_magnitude=True,
                    weight=self.residual_projection_magnitude_weight,
                )
                self.reject_magnitude_projected_bank = self._projection_encode_head(
                    self.reject_dir_bank, self.reject_mag_bank,
                    relative_xy=self.reject_xy_bank,
                    intermediate_dirs=self.reject_intermediate_dir_bank,
                    include_magnitude=True,
                    weight=self.residual_projection_magnitude_weight,
                )
            else:
                self.accept_magnitude_projected_bank = torch.empty(0, device=device, dtype=torch.float32)
                self.reject_magnitude_projected_bank = torch.empty(0, device=device, dtype=torch.float32)

            # Backward-compatible aliases mirror the sole active head in single-head
            # modes.  In dual mode they retain the historical magnitude-head alias.
            if self.residual_projection_mode in {"magnitude", "dual"} and self.residual_projection_magnitude_weight.numel() > 0:
                self.residual_projection_weight = self.residual_projection_magnitude_weight
                self.accept_projected_bank = self.accept_magnitude_projected_bank
                self.reject_projected_bank = self.reject_magnitude_projected_bank
            else:
                self.residual_projection_weight = self.residual_projection_direction_weight
                self.accept_projected_bank = self.accept_direction_projected_bank
                self.reject_projected_bank = self.reject_direction_projected_bank

    def fit_residual_projection(self) -> dict[str, object]:
        """Fit the configured supervised residual metric head(s).

        Lightning finalizes memory-bank models from ``on_validation_start`` under
        inference mode.  Leave inference mode explicitly while fitting the tiny
        linear heads, then return to the normal evaluation lifecycle.
        """
        with torch.inference_mode(False):
            with torch.enable_grad():
                return self._fit_residual_projection_with_grad()

    def _fit_single_projection_head(
        self,
        *,
        include_magnitude: bool,
        seed: int,
        head_name: str,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        model_device = self.accept_dir_bank.device
        device = (
            torch.device("cpu")
            if self.residual_projection_fit_device == "cpu"
            else model_device
        )
        accept_dirs = self.accept_dir_bank.detach().to(device=device, dtype=torch.float32).clone()
        reject_dirs = self.reject_dir_bank.detach().to(device=device, dtype=torch.float32).clone()
        accept_mags = self.accept_mag_bank.detach().to(device=device, dtype=torch.float32).clone()
        reject_mags = self.reject_mag_bank.detach().to(device=device, dtype=torch.float32).clone()
        accept_xy = self.accept_xy_bank.detach().to(device=device, dtype=torch.float32).clone()
        reject_xy = self.reject_xy_bank.detach().to(device=device, dtype=torch.float32).clone()
        accept_intermediate = self.accept_intermediate_dir_bank.detach().to(device=device, dtype=torch.float32).clone()
        reject_intermediate = self.reject_intermediate_dir_bank.detach().to(device=device, dtype=torch.float32).clone()
        accept_source = self._projection_features(
            accept_dirs, accept_mags, relative_xy=accept_xy, intermediate_dirs=accept_intermediate,
            include_magnitude=include_magnitude
        )
        reject_source = self._projection_features(
            reject_dirs, reject_mags, relative_xy=reject_xy, intermediate_dirs=reject_intermediate,
            include_magnitude=include_magnitude
        )
        d = int(accept_source.shape[1])
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        combined = torch.cat((accept_source, reject_source), dim=0)
        # PCA/SVD initialization cannot produce more orthogonal rows than the
        # number of available reference residuals. Real datasets are much larger,
        # but this also makes tiny smoke tests and sparse classes robust.
        p = min(self.residual_projection_dim, d, int(combined.shape[0]))
        max_init = min(8192, combined.shape[0])
        if combined.shape[0] > max_init:
            idx = torch.randperm(combined.shape[0], generator=generator, device=device)[:max_init]
            combined = combined[idx]
        centered = combined - combined.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            init = vh[:p].contiguous()
        except RuntimeError:
            init = torch.randn((p, d), device=device, dtype=torch.float32, generator=generator)
            init = torch.linalg.qr(init.T, mode="reduced").Q.T.contiguous()

        weight = torch.nn.Parameter(init.clone())
        optimizer = torch.optim.AdamW(
            [weight],
            lr=self.residual_projection_lr,
            weight_decay=self.residual_projection_weight_decay,
        )

        reject_groups: dict[str, torch.Tensor] = {}
        for name, chunks in self.reject_projection_store.items():
            mag_chunks = self.reject_projection_magnitude_store.get(name, [])
            if chunks and mag_chunks:
                dirs = torch.cat(chunks, dim=0).detach().to(device=device, dtype=torch.float32).clone()
                mags = torch.cat(mag_chunks, dim=0).detach().to(device=device, dtype=torch.float32).clone()
                if dirs.shape[0] != mags.shape[0]:
                    raise RuntimeError(
                        f"Projection direction/magnitude count mismatch for reject type {name!r}: "
                        f"{dirs.shape[0]} vs {mags.shape[0]}"
                    )

                xy: torch.Tensor | None = None
                if self.residual_projection_use_relative_xy:
                    xy_chunks = self.reject_projection_xy_store.get(name, [])
                    if not xy_chunks:
                        raise RuntimeError(f"Missing XY context for reject type {name!r}")
                    xy = torch.cat(xy_chunks, dim=0).detach().to(device=device, dtype=torch.float32).clone()
                    if xy.shape[0] != dirs.shape[0]:
                        raise RuntimeError(
                            f"Projection direction/XY count mismatch for reject type {name!r}: "
                            f"{dirs.shape[0]} vs {xy.shape[0]}"
                        )

                intermediate: torch.Tensor | None = None
                if self.residual_projection_intermediate_enabled:
                    inter_chunks = self.reject_projection_intermediate_store.get(name, [])
                    if not inter_chunks:
                        raise RuntimeError(f"Missing intermediate context for reject type {name!r}")
                    intermediate = torch.cat(inter_chunks, dim=0).detach().to(
                        device=device, dtype=torch.float32
                    ).clone()
                    if intermediate.shape[0] != dirs.shape[0]:
                        raise RuntimeError(
                            f"Projection direction/intermediate count mismatch for reject type {name!r}: "
                            f"{dirs.shape[0]} vs {intermediate.shape[0]}"
                        )

                reject_groups[name] = self._projection_features(
                    dirs, mags, relative_xy=xy, intermediate_dirs=intermediate,
                    include_magnitude=include_magnitude
                )
        if not reject_groups:
            reject_groups = {"": reject_source}
        group_names = sorted(reject_groups)
        configured = set(self.residual_projection_reject_type_weights)
        missing = sorted(configured - set(group_names))
        if missing:
            logger.warning(
                "%s projection weights configured for missing REFERENCE folders: %s",
                head_name, missing,
            )
        active_type_weights = {
            name: float(self.residual_projection_reject_type_weights.get(name, 1.0))
            for name in group_names
        }

        half = max(4, self.residual_projection_batch_size // 2)
        last_loss = 0.0
        last_rank = 0.0
        accept = accept_source
        for _epoch in range(self.residual_projection_epochs):
            for _ in range(self.residual_projection_steps_per_epoch):
                a_idx = torch.randint(accept.shape[0], (half,), generator=generator, device=device)
                a = accept[a_idx]

                per_group = max(1, math.ceil(half / max(1, len(group_names))))
                r_parts: list[torch.Tensor] = []
                r_names: list[str] = []
                for name in group_names:
                    bank = reject_groups[name]
                    idx = torch.randint(bank.shape[0], (per_group,), generator=generator, device=device)
                    r_parts.append(bank[idx])
                    r_names.extend([name] * per_group)
                r = torch.cat(r_parts, dim=0)[:half]
                r_names = r_names[: r.shape[0]]

                x = torch.cat((a, r), dim=0)
                labels = torch.cat((
                    torch.zeros(a.shape[0], device=device, dtype=torch.long),
                    torch.ones(r.shape[0], device=device, dtype=torch.long),
                ))
                anchor_weights = torch.ones(x.shape[0], device=device, dtype=torch.float32)
                for j, name in enumerate(r_names):
                    anchor_weights[a.shape[0] + j] = active_type_weights.get(name, 1.0)

                z = F.normalize(x @ weight.T, p=2, dim=1)
                distance = (1.0 - z @ z.T).clamp(0.0, 2.0)
                n = x.shape[0]
                eye = torch.eye(n, device=device, dtype=torch.bool)
                same = labels[:, None] == labels[None, :]
                pos_mask = same & ~eye
                neg_mask = ~same
                d_pos = distance.masked_fill(~pos_mask, float("inf")).min(dim=1).values
                d_neg = distance.masked_fill(~neg_mask, float("inf")).min(dim=1).values
                valid = torch.isfinite(d_pos) & torch.isfinite(d_neg)
                rank = F.relu(d_pos + self.residual_projection_margin - d_neg)
                rank_loss = (
                    (rank[valid] * anchor_weights[valid]).sum()
                    / anchor_weights[valid].sum().clamp_min(1.0)
                )

                gram = weight @ weight.T
                ident = torch.eye(p, device=device, dtype=torch.float32)
                ortho = ((gram - ident) ** 2).mean()
                loss = rank_loss + self.residual_projection_orthogonality_weight * ortho

                optimizer.zero_grad(set_to_none=True)
                if not loss.requires_grad:
                    raise RuntimeError(
                        f"{head_name} residual projection loss has no autograd graph. "
                        f"grad_enabled={torch.is_grad_enabled()} "
                        f"inference_mode={torch.is_inference_mode_enabled()}"
                    )
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())
                last_rank = float(rank_loss.detach().cpu())

        fitted = weight.detach().to(device=model_device, dtype=torch.float32)
        info = {
            "input_dim": d,
            "projection_dim": p,
            "include_magnitude": bool(include_magnitude),
            "use_relative_xy": bool(self.residual_projection_use_relative_xy),
            "intermediate_layers": list(self.residual_projection_intermediate_layers),
            "seed": int(seed),
            "fit_device": str(device),
            "reject_type_weights": active_type_weights,
            "final_loss": last_loss,
            "final_rank_loss": last_rank,
        }
        logger.info(
            "%s residual projection fitted: input_dim=%d dim=%d magnitude=%s final_rank_loss=%.6f",
            head_name, d, p, include_magnitude, last_rank,
        )
        return fitted, info

    def _fit_residual_projection_with_grad(self) -> dict[str, object]:
        if not self.residual_projection_enable:
            return {"status": "disabled"}
        if self.accept_dir_bank.numel() == 0 or self.reject_dir_bank.numel() == 0:
            logger.warning("Residual projection skipped because ACCEPT/REJECT banks are empty.")
            return {"status": "missing_banks"}

        self._fit_projection_magnitude_stats()

        # Mode-invariant fitting: both heads are always trained from identical
        # reference banks. Each head owns a fixed RNG stream, so execution order
        # and runtime mode cannot alter its sample schedule or initialization.
        direction_seed = self.residual_projection_seed
        magnitude_seed = self.residual_projection_seed
        direction_weight, direction_info = self._fit_single_projection_head(
            include_magnitude=False, seed=direction_seed, head_name="direction"
        )
        magnitude_weight, magnitude_info = self._fit_single_projection_head(
            include_magnitude=True, seed=magnitude_seed, head_name="magnitude"
        )
        self.residual_projection_direction_weight = direction_weight
        self.residual_projection_magnitude_weight = magnitude_weight

        with torch.no_grad():
            self._refresh_projected_banks()

        self.reject_projection_store.clear()
        self.reject_projection_magnitude_store.clear()
        self.reject_projection_xy_store.clear()
        self.reject_projection_intermediate_store.clear()
        return {
            "status": "ok",
            "residual_projection_mode": self.residual_projection_mode,
            "residual_projection_mode_id": int(self.residual_projection_mode_id.detach().cpu().item()),
            "projection_fit_policy": "both_heads_mode_invariant",
            "projection_fit_device": self.residual_projection_fit_device,
            "dual_projection_enable": self.dual_projection_enable,
            "direction_projection_enabled": self.direction_projection_enabled,
            "magnitude_projection_enabled": self.magnitude_projection_enabled,
            "use_relative_xy": self.residual_projection_use_relative_xy,
            "xy_scale": self.residual_projection_xy_scale,
            "intermediate_layers": list(self.residual_projection_intermediate_layers),
            "intermediate_scale": self.residual_projection_intermediate_scale,
            "magnitude_transform": self.residual_projection_magnitude_transform,
            "magnitude_mean": float(self.residual_projection_magnitude_mean.detach().float().cpu()),
            "magnitude_std": float(self.residual_projection_magnitude_std.detach().float().cpu()),
            "direction_head": direction_info,
            "magnitude_head": magnitude_info,
        }

    def _projected_residual_distances(
        self,
        residual_dirs: torch.Tensor,
        residual_magnitudes: torch.Tensor | None = None,
        relative_xy: torch.Tensor | None = None,
        intermediate_dirs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return direction-head and magnitude-head ACCEPT/REJECT distances."""
        q = residual_dirs.shape[0]
        inf = torch.full((q,), float("inf"), device=residual_dirs.device, dtype=torch.float32)

        if self.direction_projection_enabled and self.residual_projection_direction_weight.numel() > 0:
            direction_query = self._projection_encode_head(
                residual_dirs,
                residual_magnitudes,
                relative_xy=relative_xy,
                intermediate_dirs=intermediate_dirs,
                include_magnitude=False,
                weight=self.residual_projection_direction_weight,
            )
            direction_d_a = self._cosine_knn_distance(
                direction_query, self.accept_direction_projected_bank
            ).float()
            direction_d_r = self._cosine_knn_distance(
                direction_query, self.reject_direction_projected_bank
            ).float()
        else:
            direction_d_a = inf.clone()
            direction_d_r = inf.clone()

        if self.magnitude_projection_enabled and self.residual_projection_magnitude_weight.numel() > 0:
            magnitude_query = self._projection_encode_head(
                residual_dirs,
                residual_magnitudes,
                relative_xy=relative_xy,
                intermediate_dirs=intermediate_dirs,
                include_magnitude=True,
                weight=self.residual_projection_magnitude_weight,
            )
            magnitude_d_a = self._cosine_knn_distance(
                magnitude_query, self.accept_magnitude_projected_bank
            ).float()
            magnitude_d_r = self._cosine_knn_distance(
                magnitude_query, self.reject_magnitude_projected_bank
            ).float()
        else:
            magnitude_d_a = inf.clone()
            magnitude_d_r = inf.clone()

        return direction_d_a, direction_d_r, magnitude_d_a, magnitude_d_r

    # ------------------------------------------------------------------
    # Residual scoring / calibration
    # ------------------------------------------------------------------

    def _cosine_knn_distance(self, query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """Mean cosine distance to k nearest bank entries, streamed in FP32.

        The residual banks are tiny compared with the normal bank.  Keeping this
        decision-critical path FP32 greatly reduces PT/ORT threshold jitter with
        negligible cost relative to DINO + normal-bank kNN.
        """
        query32 = query.float()
        if bank.numel() == 0:
            return torch.full(
                (query32.shape[0],),
                float("inf"),
                device=query32.device,
                dtype=torch.float32,
            )

        k = min(max(1, int(self.num_neighbours)), bank.shape[0])
        best = torch.full(
            (query32.shape[0], k), -float("inf"), device=query32.device, dtype=torch.float32
        )

        for start in range(0, bank.shape[0], self.residual_bank_chunk_size):
            bank_chunk = bank[start : start + self.residual_bank_chunk_size].to(
                device=query32.device, dtype=torch.float32
            )
            sim = torch.matmul(query32, bank_chunk.T)
            best = torch.topk(torch.cat((best, sim), dim=1), k=k, dim=1, largest=True).values

        return (1.0 - best.mean(dim=1)).clamp(0.0, 2.0)

    def _bank_distance(
        self,
        query: torch.Tensor,
        dir_bank: torch.Tensor,
        pca_basis: torch.Tensor,
        pca_mean: torch.Tensor,
    ) -> torch.Tensor:
        query32 = query.float()
        if dir_bank.numel() == 0:
            return torch.full(
                (query32.shape[0],),
                float("inf"),
                device=query32.device,
                dtype=torch.float32,
            )

        if self.scoring_mode == "pca" and pca_basis.numel() > 0:
            mean32 = pca_mean.to(device=query32.device, dtype=torch.float32)
            basis32 = pca_basis.to(device=query32.device, dtype=torch.float32)
            centered = query32 - mean32
            projection = centered @ basis32.T @ basis32
            return (centered - projection).norm(dim=1)

        return self._cosine_knn_distance(query32, dir_bank)

    def _residual_distances(self, residual_dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d_a = self._bank_distance(
            residual_dirs, self.accept_dir_bank, self.accept_pca_basis, self.accept_pca_mean
        )
        d_d = self._bank_distance(
            residual_dirs, self.reject_dir_bank, self.reject_pca_basis, self.reject_pca_mean
        )
        return d_a.float(), d_d.float()

    def _leave_one_out_cosine_distances(
        self,
        bank: torch.Tensor,
        max_samples: int = 4096,
        *,
        seed_offset: int = 0,
    ) -> torch.Tensor:
        n = bank.shape[0]
        if n < 2:
            return torch.empty(0, device=bank.device, dtype=torch.float32)

        count = min(int(max_samples), n)
        generator = torch.Generator(device=bank.device)
        generator.manual_seed(self.calibration_seed + int(seed_offset))
        idx_all = torch.randperm(n, generator=generator, device=bank.device)[:count]
        k = min(max(1, int(self.num_neighbours)), n - 1)
        results: list[torch.Tensor] = []

        bank32 = bank.float()
        for start in range(0, count, self.calibration_query_chunk_size):
            idx = idx_all[start : start + self.calibration_query_chunk_size]
            query = bank32.index_select(0, idx)
            # Match runtime residual scoring: the decision-critical ACCEPT/REJECT
            # metric is FP32 even if stored reference banks are FP16.
            sim = torch.matmul(query, bank32.T)
            rows = torch.arange(idx.numel(), device=bank.device)
            sim[rows, idx] = -2.0
            vals = torch.topk(sim, k=k, dim=1, largest=True).values
            results.append((1.0 - vals.mean(dim=1)).clamp(0.0, 2.0))

        return torch.cat(results, dim=0)

    def _known_calibration_distances(
        self,
        bank: torch.Tensor,
        basis: torch.Tensor,
        mean: torch.Tensor,
        *,
        seed_offset: int = 0,
    ) -> torch.Tensor:
        if bank.numel() == 0:
            return torch.empty(0, device=self.memory_bank.device, dtype=torch.float32)
        if self.scoring_mode == "pca" and basis.numel() > 0:
            bank32 = bank.float()
            mean32 = mean.to(device=bank.device, dtype=torch.float32)
            basis32 = basis.to(device=bank.device, dtype=torch.float32)
            centered = bank32 - mean32
            projection = centered @ basis32.T @ basis32
            return (centered - projection).norm(dim=1)
        return self._leave_one_out_cosine_distances(bank, seed_offset=seed_offset)

    def calibrate_thresholds(
        self,
        val_d_n: torch.Tensor,
        t_normal_percentile: float = 95.0,
        t_known_percentile: float = 95.0,
    ) -> None:
        """Calibrate normal and class-specific residual thresholds."""
        if val_d_n.numel() > 0:
            value = torch.quantile(val_d_n.float(), t_normal_percentile / 100.0).item()
            self.t_normal.fill_(value)
            logger.info("Calibrated T_normal=%.6f (p%.1f)", value, t_normal_percentile)

        accept_dist = self._known_calibration_distances(
            self.accept_dir_bank,
            self.accept_pca_basis,
            self.accept_pca_mean,
            seed_offset=1,
        )
        reject_dist = self._known_calibration_distances(
            self.reject_dir_bank,
            self.reject_pca_basis,
            self.reject_pca_mean,
            seed_offset=2,
        )

        if accept_dist.numel() > 0:
            value = torch.quantile(accept_dist.float(), t_known_percentile / 100.0).item()
            self.t_accept_known.fill_(value)
            logger.info("Calibrated T_accept_known=%.6f (p%.1f)", value, t_known_percentile)
        if reject_dist.numel() > 0:
            value = torch.quantile(reject_dist.float(), t_known_percentile / 100.0).item()
            self.t_reject_known.fill_(value)
            logger.info("Calibrated T_reject_known=%.6f (p%.1f)", value, t_known_percentile)

        # Keep the compatibility threshold meaningful for external inspection.
        available = []
        if accept_dist.numel() > 0:
            available.append(self.t_accept_known.float())
        if reject_dist.numel() > 0:
            available.append(self.t_reject_known.float())
        if available:
            self.t_known.fill_(torch.stack(available).max().item())

    # ------------------------------------------------------------------
    # Forward / diagnostics
    # ------------------------------------------------------------------

    def _tolerance_forward(
        self,
        input_tensor: torch.Tensor,
        return_diagnostics: bool = False,
        return_patch_map: bool = False,
        apply_image_accept: bool = True,
        apply_reject_boost: bool = True,
        apply_projected_reject_boost: bool = True,
    ) -> InferenceBatch | tuple:
        """Run the tolerance path and optionally return per-image diagnostics.

        ``base_pred_score`` is the undamped AnomalyDINO score computed from the
        same patch distances used by the tolerance path. ``pred_score`` is the
        final score after acceptable-patch damping.
        """
        if self.memory_bank.numel() == 0:
            raise RuntimeError("Memory bank is empty. Fit the model before inference.")

        input_tensor = input_tensor.type(self.memory_bank.dtype)
        b, _, h, w = input_tensor.shape
        patch_size = self.feature_encoder.patch_size
        crop_h = h % patch_size
        crop_w = w % patch_size
        pad_top = crop_h // 2
        pad_bottom = crop_h - pad_top
        pad_left = crop_w // 2
        pad_right = crop_w - pad_left
        cropped_h = h - crop_h
        cropped_w = w - crop_w

        if crop_h > 0 or crop_w > 0:
            input_tensor = input_tensor[:, :, pad_top : h - pad_bottom, pad_left : w - pad_right]

        grid_size = (cropped_h // patch_size, cropped_w // patch_size)
        device = input_tensor.device
        features_grid, intermediate_grid = self.extract_features_with_context(input_tensor)

        if self.masking:
            features_np = features_grid.detach().cpu().numpy()
            masks_np = self.compute_background_masks(features_np, grid_size)
            masks = torch.from_numpy(masks_np).to(device)
        else:
            masks = torch.ones(features_grid.shape[:2], dtype=torch.bool, device=device)

        # Flattening a boolean [B,N] mask follows the same row-major order as
        # torch.nonzero(masks), so patch_idx stays aligned with ``features``.
        batch_idx, patch_idx = torch.nonzero(masks, as_tuple=True)
        features = features_grid[masks]
        # Preserve query precision for the v8.2 candidate rerank. _normal_knn()
        # internally casts only the full-bank-search copy to memory_bank.dtype.
        features = F.normalize(features, p=2, dim=1)

        intermediate_features: torch.Tensor | None = None
        if self.residual_projection_intermediate_enabled:
            if intermediate_grid is None or self.intermediate_memory_bank.numel() == 0:
                raise RuntimeError("Intermediate projection enabled but intermediate features/bank are unavailable")
            intermediate_features = intermediate_grid[masks].to(device=device, dtype=self.memory_bank.dtype)
            intermediate_features = F.normalize(intermediate_features, p=2, dim=-1)

        relative_xy_all: torch.Tensor | None = None
        if self.residual_projection_use_relative_xy:
            relative_xy_all = self._relative_xy_from_patch_indices(
                patch_idx, grid_size, dtype=torch.float32
            ).to(device)

        normal_anchor, d_n, anchor_indices, anchor_weights = self._normal_knn(features)

        q = features.shape[0]
        is_anom = d_n > self.t_normal
        anom_idx = torch.nonzero(is_anom, as_tuple=False).squeeze(1)

        # Avoid tensor-dependent Python control flow here. ``anom_idx`` may be
        # empty and all downstream tensor ops remain valid for zero-row inputs.
        if self.accept_dir_bank.numel() > 0 or self.reject_dir_bank.numel() > 0:
            raw_residual = (
                features.index_select(0, anom_idx).float()
                - normal_anchor.index_select(0, anom_idx)
            )
            residual_mag = raw_residual.norm(dim=1)
            residual_dirs = F.normalize(raw_residual, p=2, dim=1)
            d_a, d_d = self._residual_distances(residual_dirs)

            projected_xy: torch.Tensor | None = None
            if relative_xy_all is not None:
                projected_xy = relative_xy_all.index_select(0, anom_idx)

            projected_intermediate: torch.Tensor | None = None
            if intermediate_features is not None:
                query_intermediate = intermediate_features.index_select(0, anom_idx)
                selected_anchor_idx = anchor_indices.index_select(0, anom_idx)
                selected_anchor_weights = anchor_weights.index_select(0, anom_idx)
                anchor_k = selected_anchor_idx.shape[1]
                flat_anchor_idx = selected_anchor_idx.reshape(-1)
                normal_intermediate_neighbors = self.intermediate_memory_bank.index_select(
                    0, flat_anchor_idx
                ).reshape(
                    anom_idx.shape[0],
                    anchor_k,
                    self.intermediate_memory_bank.shape[1],
                    self.intermediate_memory_bank.shape[2],
                )
                intermediate_anchor = (
                    normal_intermediate_neighbors.float()
                    * selected_anchor_weights[:, :, None, None]
                ).sum(dim=1)
                intermediate_anchor = F.normalize(intermediate_anchor, p=2, dim=-1)
                raw_intermediate = query_intermediate.float() - intermediate_anchor
                projected_intermediate = F.normalize(
                    raw_intermediate, p=2, dim=-1
                ).flatten(start_dim=1)

            (
                direction_projected_d_a,
                direction_projected_d_d,
                magnitude_projected_d_a,
                magnitude_projected_d_d,
            ) = self._projected_residual_distances(
                residual_dirs,
                residual_mag,
                relative_xy=projected_xy,
                intermediate_dirs=projected_intermediate,
            )

            accept_known = d_a <= self.t_accept_known
            reject_known = d_d <= self.t_reject_known
            local_accept = accept_known & ((~reject_known) | ((d_a + self.margin) < d_d))
            local_reject = reject_known & (~local_accept)
            local_unknown = ~(local_accept | local_reject)

            # Smooth only the ACCEPT *damping strength* near the hard class
            # boundaries.  Class labels/diagnostics retain the calibrated legacy
            # boolean semantics.  The ramp is one-sided/conservative: a patch gets
            # no forgiveness until it is already on the legacy ACCEPT side.
            transition_width = torch.as_tensor(
                ACCEPT_DECISION_TRANSITION_WIDTH, device=device, dtype=torch.float32
            )
            da32 = d_a.float()
            dd32 = d_d.float()
            accept_threshold32 = self.t_accept_known.to(device=device, dtype=torch.float32)
            reject_threshold32 = self.t_reject_known.to(device=device, dtype=torch.float32)
            margin32 = self.margin.to(device=device, dtype=torch.float32)

            accept_known_strength = torch.clamp(
                (accept_threshold32 - da32) / transition_width, min=0.0, max=1.0
            )
            reject_not_known_strength = torch.clamp(
                (dd32 - reject_threshold32) / transition_width, min=0.0, max=1.0
            )
            prefer_accept_strength = torch.clamp(
                (dd32 - da32 - margin32) / transition_width, min=0.0, max=1.0
            )
            accept_competition_strength = torch.maximum(
                reject_not_known_strength, prefer_accept_strength
            )
            local_accept_strength = accept_known_strength * accept_competition_strength
            # Never grant soft forgiveness outside the legacy ACCEPT class.
            local_accept_strength = torch.where(
                local_accept, local_accept_strength, torch.zeros_like(local_accept_strength)
            )

            # Functional scatter avoids in-place index_put_ in the export path.
            is_accept = torch.zeros(q, dtype=torch.bool, device=device).scatter(0, anom_idx, local_accept)
            is_reject = torch.zeros(q, dtype=torch.bool, device=device).scatter(0, anom_idx, local_reject)
            is_unknown = is_anom.scatter(0, anom_idx, local_unknown)
            # d_a/d_d are intentionally FP32 even when the base detector/memory
            # bank uses FP16. Keep the bookkeeping tensors FP32 as well; scatter
            # requires source and destination dtypes to match exactly.
            d_a_all = torch.full((q,), float("inf"), device=device, dtype=torch.float32).scatter(
                0, anom_idx, d_a.to(torch.float32)
            )
            d_d_all = torch.full((q,), float("inf"), device=device, dtype=torch.float32).scatter(
                0, anom_idx, d_d.to(torch.float32)
            )
            direction_projected_d_a_all = torch.full(
                (q,), float("inf"), device=device, dtype=torch.float32
            ).scatter(0, anom_idx, direction_projected_d_a)
            direction_projected_d_d_all = torch.full(
                (q,), float("inf"), device=device, dtype=torch.float32
            ).scatter(0, anom_idx, direction_projected_d_d)
            magnitude_projected_d_a_all = torch.full(
                (q,), float("inf"), device=device, dtype=torch.float32
            ).scatter(0, anom_idx, magnitude_projected_d_a)
            magnitude_projected_d_d_all = torch.full(
                (q,), float("inf"), device=device, dtype=torch.float32
            ).scatter(0, anom_idx, magnitude_projected_d_d)
            residual_mag_all = torch.zeros(q, device=device, dtype=d_n.dtype).scatter(
                0, anom_idx, residual_mag.to(d_n.dtype)
            )
            accept_strength_all = torch.zeros(q, device=device, dtype=torch.float32).scatter(
                0, anom_idx, local_accept_strength
            )
        else:
            # Static trained-model fallback; determined by reference-bank buffers.
            is_accept = torch.zeros(q, dtype=torch.bool, device=device)
            is_reject = torch.zeros(q, dtype=torch.bool, device=device)
            is_unknown = is_anom
            d_a_all = torch.full((q,), float("inf"), device=device, dtype=torch.float32)
            d_d_all = torch.full((q,), float("inf"), device=device, dtype=torch.float32)
            direction_projected_d_a_all = torch.full((q,), float("inf"), device=device, dtype=torch.float32)
            direction_projected_d_d_all = torch.full((q,), float("inf"), device=device, dtype=torch.float32)
            magnitude_projected_d_a_all = torch.full((q,), float("inf"), device=device, dtype=torch.float32)
            magnitude_projected_d_d_all = torch.full((q,), float("inf"), device=device, dtype=torch.float32)
            residual_mag_all = torch.zeros(q, device=device, dtype=d_n.dtype)
            accept_strength_all = torch.zeros(q, device=device, dtype=torch.float32)

        # Continuous local ACCEPT damping.  Far inside the ACCEPT region this is
        # exactly the legacy damping; near a boundary it changes smoothly, so a
        # tiny PT/ORT residual-distance difference cannot erase/restore a patch.
        damping32 = self.accept_damping.to(device=device, dtype=torch.float32)
        effective_scale32 = 1.0 - accept_strength_all * (1.0 - damping32)
        effective_d = d_n * effective_scale32.to(d_n.dtype)

        n_grid = grid_size[0] * grid_size[1]
        flat_pos = batch_idx * n_grid + patch_idx
        flat_size = b * n_grid
        base_full = torch.zeros((flat_size,), device=device, dtype=d_n.dtype).scatter(
            0, flat_pos, d_n
        ).view(b, n_grid)
        effective_full = torch.zeros((flat_size,), device=device, dtype=effective_d.dtype).scatter(
            0, flat_pos, effective_d
        ).view(b, n_grid)

        local_class = torch.full((q,), PATCH_GOOD, dtype=torch.uint8, device=device)
        local_class = torch.where(is_accept, torch.full_like(local_class, PATCH_ACCEPT), local_class)
        local_class = torch.where(is_reject, torch.full_like(local_class, PATCH_REJECT), local_class)
        local_class = torch.where(is_unknown, torch.full_like(local_class, PATCH_UNKNOWN), local_class)
        class_full = torch.full((flat_size,), PATCH_GOOD, dtype=torch.uint8, device=device).scatter(
            0, flat_pos, local_class
        ).view(b, n_grid)

        da_full = torch.full((flat_size,), float("inf"), device=device, dtype=torch.float32).scatter(
            0, flat_pos, d_a_all
        ).view(b, n_grid)
        dd_full = torch.full((flat_size,), float("inf"), device=device, dtype=torch.float32).scatter(
            0, flat_pos, d_d_all
        ).view(b, n_grid)
        direction_projected_da_full = torch.full(
            (flat_size,), float("inf"), device=device, dtype=torch.float32
        ).scatter(0, flat_pos, direction_projected_d_a_all).view(b, n_grid)
        direction_projected_dd_full = torch.full(
            (flat_size,), float("inf"), device=device, dtype=torch.float32
        ).scatter(0, flat_pos, direction_projected_d_d_all).view(b, n_grid)
        magnitude_projected_da_full = torch.full(
            (flat_size,), float("inf"), device=device, dtype=torch.float32
        ).scatter(0, flat_pos, magnitude_projected_d_a_all).view(b, n_grid)
        magnitude_projected_dd_full = torch.full(
            (flat_size,), float("inf"), device=device, dtype=torch.float32
        ).scatter(0, flat_pos, magnitude_projected_d_d_all).view(b, n_grid)
        mag_full = torch.zeros((flat_size,), device=device, dtype=d_n.dtype).scatter(
            0, flat_pos, residual_mag_all
        ).view(b, n_grid)
        accept_strength_full = torch.zeros(
            (flat_size,), device=device, dtype=torch.float32
        ).scatter(0, flat_pos, accept_strength_all).view(b, n_grid)

        # Image-level evidence is computed from the same top 1% *base* anomaly
        # patches used by AnomalyDINO's image score.  This is deliberately
        # continuous: positive advantage means those high-anomaly patches are
        # closer to the acceptable residual bank than the reject bank.
        base_score = self.mean_top1p(base_full).reshape(b)
        patch_score = self.mean_top1p(effective_full).reshape(b)

        num_top = max(int(n_grid * 0.01), 1)
        # Use a tiny deterministic patch-index preference for evidence selection.
        # The actual base/effective values are untouched; this only stabilizes
        # which near-tied patches feed the hard image-level tolerance decision.
        patch_order = torch.arange(n_grid, device=device, dtype=torch.float32)
        top_tie = (patch_order / float(max(n_grid, 1))) * self.image_topk_tie_break_epsilon.to(
            device=device, dtype=torch.float32
        )
        top_rank = base_full.float() - top_tie.unsqueeze(0)  # lower patch index wins ties
        top_idx = torch.topk(top_rank, k=num_top, dim=1, largest=True).indices
        top_class = torch.gather(class_full.long(), 1, top_idx)
        top_da = torch.gather(da_full, 1, top_idx)
        top_dd = torch.gather(dd_full, 1, top_idx)
        top_direction_projected_da = torch.gather(direction_projected_da_full, 1, top_idx)
        top_direction_projected_dd = torch.gather(direction_projected_dd_full, 1, top_idx)
        top_magnitude_projected_da = torch.gather(magnitude_projected_da_full, 1, top_idx)
        top_magnitude_projected_dd = torch.gather(magnitude_projected_dd_full, 1, top_idx)
        top_mag = torch.gather(mag_full, 1, top_idx)

        # Diagnostic view of the exact scalar magnitude feature consumed by v6c.
        # Only patches with projected residual distances are considered valid.
        if self.residual_projection_use_magnitude:
            transformed_mag = self._transform_projection_magnitude(top_mag.reshape(-1)).reshape_as(top_mag)
            mag_mean = self.residual_projection_magnitude_mean.to(device=device, dtype=torch.float32)
            mag_std = self.residual_projection_magnitude_std.to(device=device, dtype=torch.float32).clamp_min(1.0e-6)
            projected_mag_feature = (transformed_mag - mag_mean) / mag_std
            if self.residual_projection_magnitude_clip > 0.0:
                projected_mag_feature = torch.clamp(
                    projected_mag_feature,
                    min=-self.residual_projection_magnitude_clip,
                    max=self.residual_projection_magnitude_clip,
                )
            projected_mag_feature = projected_mag_feature * self.residual_projection_magnitude_scale
            projected_mag_valid = torch.isfinite(top_magnitude_projected_da) | torch.isfinite(top_magnitude_projected_dd)
            projected_mag_feature = torch.where(
                projected_mag_valid,
                projected_mag_feature,
                torch.full_like(projected_mag_feature, float("nan")),
            )
        else:
            projected_mag_feature = torch.full_like(top_mag.float(), float("nan"))

        top_da = top_da.float()
        top_dd = top_dd.float()
        top_direction_projected_da = top_direction_projected_da.float()
        top_direction_projected_dd = top_direction_projected_dd.float()
        top_magnitude_projected_da = top_magnitude_projected_da.float()
        top_magnitude_projected_dd = top_magnitude_projected_dd.float()
        finite_da = torch.where(torch.isfinite(top_da), top_da, torch.full_like(top_da, float("nan")))
        finite_dd = torch.where(torch.isfinite(top_dd), top_dd, torch.full_like(top_dd, float("nan")))
        accept_advantage = top_dd - top_da  # positive => closer to acceptable bank
        accept_advantage = torch.where(
            torch.isfinite(accept_advantage),
            accept_advantage,
            torch.full_like(accept_advantage, float("nan")),
        )
        direction_projected_accept_advantage = top_direction_projected_dd - top_direction_projected_da
        direction_projected_accept_advantage = torch.where(
            torch.isfinite(direction_projected_accept_advantage),
            direction_projected_accept_advantage,
            torch.full_like(direction_projected_accept_advantage, float("nan")),
        )
        magnitude_projected_accept_advantage = top_magnitude_projected_dd - top_magnitude_projected_da
        magnitude_projected_accept_advantage = torch.where(
            torch.isfinite(magnitude_projected_accept_advantage),
            magnitude_projected_accept_advantage,
            torch.full_like(magnitude_projected_accept_advantage, float("nan")),
        )

        def _nanmean(x: torch.Tensor) -> torch.Tensor:
            valid = torch.isfinite(x)
            count = valid.sum(dim=1).clamp_min(1)
            out = torch.where(valid, x, torch.zeros_like(x)).sum(dim=1) / count
            return torch.where(valid.any(dim=1), out, torch.full_like(out, float("nan")))

        def _nanmin(x: torch.Tensor) -> torch.Tensor:
            filled = torch.where(torch.isfinite(x), x, torch.full_like(x, float("inf")))
            out = filled.min(dim=1).values
            return torch.where(torch.isfinite(out), out, torch.full_like(out, float("nan")))

        top_mean_advantage = _nanmean(accept_advantage)
        top_mean_direction_projected_advantage = _nanmean(direction_projected_accept_advantage)
        top_mean_magnitude_projected_advantage = _nanmean(magnitude_projected_accept_advantage)

        # Aggregate reject evidence vetoes local patch forgiveness.  A negative
        # advantage means the top anomaly patches are, on average, closer to the
        # reject bank.  In that case restore the stock/base anomaly field before
        # considering image-level ACCEPT.  This prevents a few locally ACCEPT-like
        # patches from suppressing an image that is globally reject-like.
        image_reject_veto = torch.zeros((b,), dtype=torch.bool, device=device)
        if self.image_reject_veto_enable:
            veto_threshold = self.image_reject_veto_threshold.to(
                device=device, dtype=top_mean_advantage.dtype
            )
            image_reject_veto = torch.isfinite(top_mean_advantage) & (top_mean_advantage <= veto_threshold)
            effective_full = torch.where(
                image_reject_veto[:, None],
                base_full,
                effective_full,
            )

        veto_score = self.mean_top1p(effective_full).reshape(b)

        # Whole-image ACCEPT can come from either the legacy raw residual evidence
        # or the learned projected evidence. Raw ACCEPT remains subordinate to the
        # raw reject veto. The projected gate is deliberately independent of that
        # veto: its purpose is to rescue acceptable appearances for which raw
        # residual space is reject-like but BOTH projected heads agree ACCEPT.
        raw_image_accept_strength = torch.zeros((b,), device=device, dtype=torch.float32)
        projected_image_accept_strength = torch.zeros((b,), device=device, dtype=torch.float32)
        projected_image_accept_evidence = torch.full(
            (b,), float("nan"), device=device, dtype=torch.float32
        )

        if apply_image_accept and self.image_accept_enable:
            advantage32 = top_mean_advantage.float()
            threshold32 = self.image_accept_threshold.to(device=device, dtype=torch.float32)
            finite_advantage = torch.isfinite(advantage32)
            transition_width = torch.as_tensor(
                ACCEPT_DECISION_TRANSITION_WIDTH, device=device, dtype=torch.float32
            )
            raw_image_accept_strength = torch.where(
                finite_advantage,
                torch.clamp(
                    (advantage32 - threshold32) / transition_width,
                    min=0.0,
                    max=1.0,
                ),
                torch.zeros_like(advantage32),
            )
            # Legacy/raw ACCEPT never overrides explicit raw reject-like evidence.
            raw_image_accept_strength = torch.where(
                image_reject_veto,
                torch.zeros_like(raw_image_accept_strength),
                raw_image_accept_strength,
            )

        projected_image_accept_enabled = bool(
            getattr(self, "projected_image_accept_enable", False)
        )
        if apply_image_accept and projected_image_accept_enabled:
            direction_adv32 = top_mean_direction_projected_advantage.float()
            magnitude_adv32 = top_mean_magnitude_projected_advantage.float()

            # Runtime projection mode is a static Python attribute and therefore
            # specializes cleanly during ONNX export (no extra model input).
            if self.residual_projection_mode == "direction":
                projected_image_accept_evidence = direction_adv32
            elif self.residual_projection_mode == "magnitude":
                projected_image_accept_evidence = magnitude_adv32
            else:
                # Conservative dual fusion: require both heads to be finite and
                # acceptable-like. min() means the weaker head controls the gate.
                both_finite = torch.isfinite(direction_adv32) & torch.isfinite(magnitude_adv32)
                projected_image_accept_evidence = torch.where(
                    both_finite,
                    torch.minimum(direction_adv32, magnitude_adv32),
                    torch.full_like(direction_adv32, float("nan")),
                )

            # Backward-compatible fallbacks only matter for an older full-module
            # checkpoint that is manually opted into the new gate after loading.
            projected_threshold = getattr(
                self, "projected_image_accept_threshold", self.image_accept_threshold
            ).to(device=device, dtype=torch.float32)
            projected_finite = torch.isfinite(projected_image_accept_evidence)
            transition_width = torch.as_tensor(
                ACCEPT_DECISION_TRANSITION_WIDTH, device=device, dtype=torch.float32
            )
            projected_image_accept_strength = torch.where(
                projected_finite,
                torch.clamp(
                    (projected_image_accept_evidence - projected_threshold) / transition_width,
                    min=0.0,
                    max=1.0,
                ),
                torch.zeros_like(projected_image_accept_evidence),
            )
            # Intentionally DO NOT mask this by image_reject_veto. The projected
            # evidence is the independent fallback that may override the raw veto.

        raw_image_accept = raw_image_accept_strength > 0.0
        projected_image_accept = projected_image_accept_strength > 0.0
        image_accept_strength = torch.maximum(
            raw_image_accept_strength, projected_image_accept_strength
        )
        image_accept = image_accept_strength > 0.0

        if apply_image_accept:
            raw_damping32 = self.image_accept_damping.to(device=device, dtype=torch.float32)
            projected_damping = getattr(
                self, "projected_image_accept_damping", self.image_accept_damping
            )
            projected_damping32 = projected_damping.to(device=device, dtype=torch.float32)
            raw_scale32 = 1.0 - raw_image_accept_strength * (1.0 - raw_damping32)
            projected_scale32 = 1.0 - projected_image_accept_strength * (1.0 - projected_damping32)
            # Apply ACCEPT damping once. If both gates fire, the stronger
            # suppression wins instead of multiplying two damping operations.
            image_scale32 = torch.minimum(raw_scale32, projected_scale32)
            effective_full = effective_full * image_scale32.to(effective_full.dtype)[:, None]

        # v4 score after all ACCEPT-side processing and before the v5 REJECT boost.
        accept_score = self.mean_top1p(effective_full).reshape(b)

        # v5: continuous positive REJECT evidence.  Negative accept-advantage
        # means the top anomaly patches are closer to the reject residual bank.
        # Convert the amount by which the image falls below the calibrated
        # advantage threshold into an additive anomaly-score boost.  The boost
        # is written only onto the same top base-anomaly patches used for the
        # image score, preserving spatial localization rather than lifting the
        # entire anomaly map.
        reject_boost_evidence = torch.zeros((b,), device=device, dtype=top_mean_advantage.dtype)
        reject_boost_amount = torch.zeros((b,), device=device, dtype=effective_full.dtype)
        reject_boost_applied = torch.zeros((b,), dtype=torch.bool, device=device)
        if apply_reject_boost and self.image_reject_boost_enable:
            boost_threshold = self.image_reject_boost_advantage_threshold.to(
                device=device, dtype=top_mean_advantage.dtype
            )
            finite_advantage = torch.isfinite(top_mean_advantage)
            reject_boost_evidence = torch.where(
                finite_advantage,
                torch.clamp(boost_threshold - top_mean_advantage, min=0.0),
                torch.zeros_like(top_mean_advantage),
            )
            reject_boost_amount = reject_boost_evidence.to(effective_full.dtype) * self.image_reject_boost_lambda.to(
                device=device, dtype=effective_full.dtype
            )
            max_boost = self.image_reject_boost_max.to(device=device, dtype=effective_full.dtype)
            if float(max_boost.detach().float().cpu()) > 0.0:
                reject_boost_amount = torch.clamp(reject_boost_amount, max=max_boost)

            # ACCEPT wins only when the aggregate evidence itself was ACCEPT-like.
            # Explicitly keep the two image-level actions mutually exclusive for
            # unusual custom threshold configurations.
            reject_boost_amount = reject_boost_amount * (
                1.0 - image_accept_strength.to(reject_boost_amount.dtype)
            )
            reject_boost_applied = reject_boost_amount > 0

            # ONNX opset-14 friendly: no tensor-dependent Python branch and no
            # in-place scatter_. A zero boost naturally produces a zero field.
            boost_values = reject_boost_amount[:, None].expand(-1, num_top)
            boost_field = torch.zeros_like(effective_full).scatter(1, top_idx, boost_values)
            effective_full = effective_full + boost_field

        # v5 score before the learned projection evidence.
        v5_score = self.mean_top1p(effective_full).reshape(b)

        # Projected evidence is mode-specific. In single-head modes the inactive
        # head has zero boost, so the same max-fusion code reduces exactly to the
        # active head. In dual mode both calibrated boosts are computed from the
        # same frozen v5 score and max-fused to avoid double-counting evidence.
        direction_projected_boost_evidence = torch.zeros(
            (b,), device=device, dtype=top_mean_direction_projected_advantage.dtype
        )
        magnitude_projected_boost_evidence = torch.zeros(
            (b,), device=device, dtype=top_mean_magnitude_projected_advantage.dtype
        )
        direction_projected_boost_amount = torch.zeros((b,), device=device, dtype=effective_full.dtype)
        magnitude_projected_boost_amount = torch.zeros((b,), device=device, dtype=effective_full.dtype)
        direction_projected_boost_applied = torch.zeros((b,), dtype=torch.bool, device=device)
        magnitude_projected_boost_applied = torch.zeros((b,), dtype=torch.bool, device=device)

        if apply_projected_reject_boost and self.projected_reject_boost_enable:
            direction_threshold = self.direction_projected_reject_boost_advantage_threshold.to(
                device=device, dtype=top_mean_direction_projected_advantage.dtype
            )
            direction_finite = torch.isfinite(top_mean_direction_projected_advantage)
            direction_projected_boost_evidence = torch.where(
                direction_finite,
                torch.clamp(
                    direction_threshold - top_mean_direction_projected_advantage,
                    min=0.0,
                ),
                torch.zeros_like(top_mean_direction_projected_advantage),
            )
            direction_projected_boost_amount = (
                direction_projected_boost_evidence.to(effective_full.dtype)
                * self.direction_projected_reject_boost_lambda.to(
                    device=device, dtype=effective_full.dtype
                )
            )

            magnitude_threshold = self.magnitude_projected_reject_boost_advantage_threshold.to(
                device=device, dtype=top_mean_magnitude_projected_advantage.dtype
            )
            magnitude_finite = torch.isfinite(top_mean_magnitude_projected_advantage)
            magnitude_projected_boost_evidence = torch.where(
                magnitude_finite,
                torch.clamp(
                    magnitude_threshold - top_mean_magnitude_projected_advantage,
                    min=0.0,
                ),
                torch.zeros_like(top_mean_magnitude_projected_advantage),
            )
            magnitude_projected_boost_amount = (
                magnitude_projected_boost_evidence.to(effective_full.dtype)
                * self.magnitude_projected_reject_boost_lambda.to(
                    device=device, dtype=effective_full.dtype
                )
            )

            proj_max = self.projected_reject_boost_max.to(device=device, dtype=effective_full.dtype)
            if float(proj_max.detach().float().cpu()) > 0.0:
                direction_projected_boost_amount = torch.clamp(
                    direction_projected_boost_amount, max=proj_max
                )
                magnitude_projected_boost_amount = torch.clamp(
                    magnitude_projected_boost_amount, max=proj_max
                )

            remaining_reject_strength = 1.0 - image_accept_strength.to(
                direction_projected_boost_amount.dtype
            )
            direction_projected_boost_amount = (
                direction_projected_boost_amount * remaining_reject_strength
            )
            magnitude_projected_boost_amount = (
                magnitude_projected_boost_amount
                * remaining_reject_strength.to(magnitude_projected_boost_amount.dtype)
            )
            direction_projected_boost_applied = direction_projected_boost_amount > 0
            magnitude_projected_boost_applied = magnitude_projected_boost_amount > 0

        # Runtime mode controls execution/fusion only; fitted weights and per-head
        # calibrations are mode-invariant. Dual safety fallback uses separate
        # scales instead of zeroing a head's standalone lambda.
        if self.residual_projection_mode == "direction":
            fused_direction_amount = direction_projected_boost_amount
            fused_magnitude_amount = torch.zeros_like(magnitude_projected_boost_amount)
        elif self.residual_projection_mode == "magnitude":
            fused_direction_amount = torch.zeros_like(direction_projected_boost_amount)
            fused_magnitude_amount = magnitude_projected_boost_amount
        else:
            fused_direction_amount = direction_projected_boost_amount * self.dual_projection_direction_fusion_scale.to(
                device=device, dtype=direction_projected_boost_amount.dtype
            )
            fused_magnitude_amount = magnitude_projected_boost_amount * self.dual_projection_magnitude_fusion_scale.to(
                device=device, dtype=magnitude_projected_boost_amount.dtype
            )

        projected_boost_amount = torch.maximum(fused_direction_amount, fused_magnitude_amount)
        projected_boost_applied = projected_boost_amount > 0
        projected_boost_evidence = torch.where(
            fused_direction_amount >= fused_magnitude_amount,
            direction_projected_boost_evidence.to(torch.float32),
            magnitude_projected_boost_evidence.to(torch.float32),
        )

        # Per-head scores are useful for exact within-run ablation and make it
        # possible to confirm that fused final == max(direction, magnitude).
        direction_projected_pred_score = v5_score + direction_projected_boost_amount.reshape(b)
        magnitude_projected_pred_score = v5_score + magnitude_projected_boost_amount.reshape(b)

        # ONNX opset-14 friendly out-of-place scatter.
        proj_boost_values = projected_boost_amount[:, None].expand(-1, num_top)
        proj_boost_field = torch.zeros_like(effective_full).scatter(1, top_idx, proj_boost_values)
        effective_full = effective_full + proj_boost_field

        image_score = self.mean_top1p(effective_full).reshape(b)
        anomaly_map = effective_full.view(b, 1, *grid_size)
        anomaly_map = self.anomaly_map_generator(anomaly_map, (cropped_h, cropped_w))
        if crop_h > 0 or crop_w > 0:
            anomaly_map = F.pad(
                anomaly_map,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="replicate",
            )

        predictions = InferenceBatch(pred_score=image_score, anomaly_map=anomaly_map)
        if return_patch_map:
            patch_class_map = class_full.view(b, 1, *grid_size).float()
            patch_class_map = F.interpolate(
                patch_class_map,
                size=(cropped_h, cropped_w),
                mode="nearest",
            )
            if crop_h > 0 or crop_w > 0:
                patch_class_map = F.pad(
                    patch_class_map,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                )
            return predictions, patch_class_map

        if not return_diagnostics:
            return predictions

        diagnostics = {
            "base_pred_score": base_score.float(),
            "patch_pred_score": patch_score.float(),
            "veto_pred_score": veto_score.float(),
            "accept_pred_score": accept_score.float(),
            "v5_pred_score": v5_score.float(),
            "direction_projected_pred_score": direction_projected_pred_score.float(),
            "magnitude_projected_pred_score": magnitude_projected_pred_score.float(),
            "final_pred_score": image_score.float(),
            "patch_score_reduction": (base_score - patch_score).float(),
            "veto_score_restoration": (veto_score - patch_score).float(),
            "image_score_reduction": (veto_score - accept_score).float(),
            "reject_boost_score_increase": (v5_score - accept_score).float(),
            "projected_boost_score_increase": (image_score - v5_score).float(),
            "image_reject_veto_applied": image_reject_veto.float(),
            "score_reduction": (base_score - image_score).float(),
            "image_accept_applied": image_accept.float(),
            "image_accept_strength": image_accept_strength.float(),
            "raw_image_accept_applied": raw_image_accept.float(),
            "raw_image_accept_strength": raw_image_accept_strength.float(),
            "projected_image_accept_applied": projected_image_accept.float(),
            "projected_image_accept_strength": projected_image_accept_strength.float(),
            "projected_image_accept_evidence": projected_image_accept_evidence.float(),
            "image_accept_threshold": torch.full((b,), float(self.image_accept_threshold.detach().float().cpu()), device=device),
            "projected_image_accept_threshold": torch.full(
                (b,),
                float(
                    getattr(
                        self, "projected_image_accept_threshold", self.image_accept_threshold
                    ).detach().float().cpu()
                ),
                device=device,
            ),
            "accept_decision_transition_width": torch.full(
                (b,), ACCEPT_DECISION_TRANSITION_WIDTH, device=device, dtype=torch.float32
            ),
            "top_mean_local_accept_strength": torch.gather(
                accept_strength_full, 1, top_idx
            ).mean(dim=1).float(),
            "image_reject_boost_applied": reject_boost_applied.float(),
            "image_reject_boost_evidence": reject_boost_evidence.float(),
            "image_reject_boost_amount": reject_boost_amount.float(),
            "image_reject_boost_advantage_threshold": torch.full(
                (b,),
                float(self.image_reject_boost_advantage_threshold.detach().float().cpu()),
                device=device,
            ),
            "image_reject_boost_lambda": torch.full(
                (b,), float(self.image_reject_boost_lambda.detach().float().cpu()), device=device
            ),
            "projected_reject_boost_applied": projected_boost_applied.float(),
            "projected_reject_boost_evidence": projected_boost_evidence.float(),
            "projected_reject_boost_amount": projected_boost_amount.float(),
            "direction_projected_reject_boost_applied": direction_projected_boost_applied.float(),
            "direction_projected_reject_boost_evidence": direction_projected_boost_evidence.float(),
            "direction_projected_reject_boost_amount": direction_projected_boost_amount.float(),
            "direction_projected_reject_boost_advantage_threshold": torch.full(
                (b,), float(self.direction_projected_reject_boost_advantage_threshold.detach().float().cpu()), device=device
            ),
            "direction_projected_reject_boost_lambda": torch.full(
                (b,), float(self.direction_projected_reject_boost_lambda.detach().float().cpu()), device=device
            ),
            "magnitude_projected_reject_boost_applied": magnitude_projected_boost_applied.float(),
            "magnitude_projected_reject_boost_evidence": magnitude_projected_boost_evidence.float(),
            "magnitude_projected_reject_boost_amount": magnitude_projected_boost_amount.float(),
            "magnitude_projected_reject_boost_advantage_threshold": torch.full(
                (b,), float(self.magnitude_projected_reject_boost_advantage_threshold.detach().float().cpu()), device=device
            ),
            "magnitude_projected_reject_boost_lambda": torch.full(
                (b,), float(self.magnitude_projected_reject_boost_lambda.detach().float().cpu()), device=device
            ),
            # Legacy projected_* parameters mirror the magnitude head.
            "projected_reject_boost_advantage_threshold": torch.full(
                (b,), float(self.magnitude_projected_reject_boost_advantage_threshold.detach().float().cpu()), device=device
            ),
            "projected_reject_boost_lambda": torch.full(
                (b,), float(self.magnitude_projected_reject_boost_lambda.detach().float().cpu()), device=device
            ),
            "residual_projection_use_magnitude": torch.full(
                (b,), 1.0 if self.residual_projection_use_magnitude else 0.0, device=device
            ),
            "dual_projection_direction_fusion_scale": torch.full(
                (b,), float(self.dual_projection_direction_fusion_scale.detach().float().cpu()), device=device
            ),
            "dual_projection_magnitude_fusion_scale": torch.full(
                (b,), float(self.dual_projection_magnitude_fusion_scale.detach().float().cpu()), device=device
            ),
            "residual_projection_use_relative_xy": torch.full(
                (b,), 1.0 if self.residual_projection_use_relative_xy else 0.0, device=device
            ),
            "residual_projection_intermediate_layer_count": torch.full(
                (b,), float(len(self.residual_projection_intermediate_layers)), device=device
            ),
            "residual_projection_magnitude_mean": torch.full(
                (b,), float(self.residual_projection_magnitude_mean.detach().float().cpu()), device=device
            ),
            "residual_projection_magnitude_std": torch.full(
                (b,), float(self.residual_projection_magnitude_std.detach().float().cpu()), device=device
            ),
            "num_candidate_patches": (class_full != PATCH_GOOD).sum(dim=1).float(),
            "num_accept_patches": (class_full == PATCH_ACCEPT).sum(dim=1).float(),
            "num_reject_patches": (class_full == PATCH_REJECT).sum(dim=1).float(),
            "num_unknown_patches": (class_full == PATCH_UNKNOWN).sum(dim=1).float(),
            "top1p_patch_count": torch.full((b,), float(num_top), device=device),
            "top_accept_patches": (top_class == PATCH_ACCEPT).sum(dim=1).float(),
            "top_reject_patches": (top_class == PATCH_REJECT).sum(dim=1).float(),
            "top_unknown_patches": (top_class == PATCH_UNKNOWN).sum(dim=1).float(),
            "top_min_d_accept": _nanmin(finite_da).float(),
            "top_min_d_reject": _nanmin(finite_dd).float(),
            "top_mean_d_accept": _nanmean(finite_da).float(),
            "top_mean_d_reject": _nanmean(finite_dd).float(),
            "top_mean_accept_advantage": top_mean_advantage.float(),
            "top_mean_direction_projected_d_accept": _nanmean(torch.where(torch.isfinite(top_direction_projected_da), top_direction_projected_da, torch.full_like(top_direction_projected_da, float("nan")))).float(),
            "top_mean_direction_projected_d_reject": _nanmean(torch.where(torch.isfinite(top_direction_projected_dd), top_direction_projected_dd, torch.full_like(top_direction_projected_dd, float("nan")))).float(),
            "top_mean_direction_projected_accept_advantage": top_mean_direction_projected_advantage.float(),
            "top_mean_magnitude_projected_d_accept": _nanmean(torch.where(torch.isfinite(top_magnitude_projected_da), top_magnitude_projected_da, torch.full_like(top_magnitude_projected_da, float("nan")))).float(),
            "top_mean_magnitude_projected_d_reject": _nanmean(torch.where(torch.isfinite(top_magnitude_projected_dd), top_magnitude_projected_dd, torch.full_like(top_magnitude_projected_dd, float("nan")))).float(),
            "top_mean_magnitude_projected_accept_advantage": top_mean_magnitude_projected_advantage.float(),
            # Legacy projected diagnostics mirror the magnitude-aware head.
            "top_mean_projected_d_accept": _nanmean(torch.where(torch.isfinite(top_magnitude_projected_da), top_magnitude_projected_da, torch.full_like(top_magnitude_projected_da, float("nan")))).float(),
            "top_mean_projected_d_reject": _nanmean(torch.where(torch.isfinite(top_magnitude_projected_dd), top_magnitude_projected_dd, torch.full_like(top_magnitude_projected_dd, float("nan")))).float(),
            "top_mean_projected_accept_advantage": top_mean_magnitude_projected_advantage.float(),
            "top_mean_residual_magnitude": top_mag.mean(dim=1).float(),
            "top_mean_projection_magnitude_feature": _nanmean(projected_mag_feature).float(),
        }
        return predictions, diagnostics

    def forward_with_diagnostics(
        self,
        input_tensor: torch.Tensor,
        apply_image_accept: bool = True,
        apply_reject_boost: bool = True,
        apply_projected_reject_boost: bool = True,
    ) -> tuple[InferenceBatch, dict[str, torch.Tensor]]:
        """Force the tolerance path and return diagnostics even when damping=1.

        ``apply_image_accept=False`` disables both raw and projected whole-image
        ACCEPT gates while collecting calibration evidence.
        ``apply_reject_boost=False`` is used while calibrating the v5 REJECT
        boost so that candidate boost parameters can be evaluated off-line on
        the frozen pre-boost score.
        """
        if self.training:
            raise RuntimeError("forward_with_diagnostics is inference-only")
        result = self._tolerance_forward(
            input_tensor,
            return_diagnostics=True,
            apply_image_accept=apply_image_accept,
            apply_reject_boost=apply_reject_boost,
            apply_projected_reject_boost=apply_projected_reject_boost,
        )
        assert isinstance(result, tuple)
        return result

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch | tuple:
        if self.training:
            result = super().forward(input_tensor)
            if self.residual_projection_intermediate_enabled:
                self._collect_training_intermediate(input_tensor)
            return result

        if self.detector_only or (
            self.strict_stock_when_damping_one
            and not self.emit_patch_class
            and not self.image_accept_enable
            and not bool(getattr(self, "projected_image_accept_enable", False))
            and (
                not self.image_reject_boost_enable
                or float(self.image_reject_boost_lambda.detach().float().cpu()) == 0.0
            )
            and float(self.accept_damping.detach().float().cpu()) == 1.0
        ):
            return super().forward(input_tensor)

        if self.emit_patch_class:
            return self._tolerance_forward(
                input_tensor,
                return_diagnostics=False,
                return_patch_map=True,
            )

        predictions = self._tolerance_forward(input_tensor, return_diagnostics=False)
        assert isinstance(predictions, InferenceBatch)
        return predictions
