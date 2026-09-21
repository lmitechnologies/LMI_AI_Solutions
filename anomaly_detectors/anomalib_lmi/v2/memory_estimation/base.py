from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from .common import (
    FeatureProfile,
    MemoryBudget,
    MemoryEstimate,
    TileConfig,
    bytes_to_mib,
    dtype_from_precision,
    dtype_nbytes,
)

_MISSING = object()


class BaseAnomalibMemoryEstimator(ABC):
    """Base estimator for anomalib training memory.

    Anomalib models fall into distinct training-memory categories:

    - memory-bank (peak scales with the number of training images): PatchCore,
      PaDiM. ``max_train_images`` is the meaningful output.
    - gradient-trained / deep (peak driven by params + optimizer state + batch
      activations, independent of dataset size): EfficientAd, Dinomaly, Ganomaly,
      AnomalyDino. Image count does not constrain storage; see
      ``GradientTrainedMemoryEstimator``.
    """

    model_type: str = "base"

    def __init__(
        self,
        model,
        tile_config: TileConfig,
        *,
        precision: str | torch.dtype = "float32",
        profiling_device: str = "cuda",
        profiling_dtype: torch.dtype | None = None,
    ):
        self.model = model
        self.tile_config = tile_config
        self.profiling_device = profiling_device

        self.activation_dtype = dtype_from_precision(precision)
        self.profiling_dtype = profiling_dtype or self.activation_dtype
        self.activation_dtype_bytes = self.dtype_nbytes(self.activation_dtype)

    dtype_nbytes = staticmethod(dtype_nbytes)

    @staticmethod
    def dtype_from_string(
        dtype_str: str,
        fallback: torch.dtype = torch.float32,
    ) -> torch.dtype:
        return dtype_from_precision(dtype_str, fallback=fallback)

    def unwrap_inner_model(self):
        return getattr(self.model, "model", self.model)

    def _infer_hparam(self, name: str, default, cast=None) -> Any:
        """Walk model / model.model / hparams candidates for an hparam value."""
        candidates = [
            self.model,
            getattr(self.model, "model", None),
            getattr(self.model, "hparams", None),
            getattr(getattr(self.model, "model", None), "hparams", None),
        ]

        for obj in candidates:
            if obj is None:
                continue

            value = getattr(obj, name, _MISSING)
            if value is _MISSING and hasattr(obj, "get"):
                value = obj.get(name, _MISSING)

            if value is not _MISSING and value is not None:
                return cast(value) if cast is not None else value

        return cast(default) if cast is not None else default

    def get_feature_extractor(self):
        inner = self.unwrap_inner_model()

        if hasattr(inner, "feature_extractor"):
            return inner.feature_extractor

        if hasattr(self.model, "feature_extractor"):
            return self.model.feature_extractor

        raise AttributeError(f"Could not find feature_extractor on {type(self.model)}")

    def profile_features(self) -> FeatureProfile:
        cfg = self.tile_config
        inner = self.unwrap_inner_model()
        extractor = self.get_feature_extractor().to(self.profiling_device).eval()

        Th, Tw = cfg.effective_tile_size

        dummy = torch.zeros(
            1,
            cfg.channels,
            Th,
            Tw,
            device=self.profiling_device,
            dtype=self.profiling_dtype,
        )

        use_autocast = self.profiling_device.startswith("cuda") and self.profiling_dtype in {torch.float16, torch.bfloat16}

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=self.profiling_dtype,
                enabled=use_autocast,
            ):
                out = extractor(dummy)

                if isinstance(out, dict):
                    features = out
                elif isinstance(out, (list, tuple)):
                    features = {f"layer_{i}": x for i, x in enumerate(out)}
                else:
                    features = {"features": out}

                feature_shapes = {name: tuple(feat.shape) for name, feat in features.items()}

                feature_dtypes = {name: str(feat.dtype) for name, feat in features.items()}

                embedding = None
                if hasattr(inner, "generate_embedding"):
                    try:
                        embedding = inner.generate_embedding(features)
                    except Exception:
                        embedding = None

        if embedding is not None:
            embedding_shape = tuple(embedding.shape)
            embedding_dtype = str(embedding.dtype)
        else:
            max_h = max(shape[-2] for shape in feature_shapes.values())
            max_w = max(shape[-1] for shape in feature_shapes.values())
            total_c = sum(shape[1] for shape in feature_shapes.values())

            embedding_shape = (1, total_c, max_h, max_w)
            embedding_dtype = next(iter(feature_dtypes.values()))

        _, emb_c, emb_h, emb_w = embedding_shape
        H, W = cfg.image_size

        stitched_h = int(round(H * emb_h / Th))
        stitched_w = int(round(W * emb_w / Tw))

        return FeatureProfile(
            feature_shapes=feature_shapes,
            feature_dtypes=feature_dtypes,
            embedding_shape_per_tile=embedding_shape,
            embedding_dtype=embedding_dtype,
            stitched_embedding_shape=(
                cfg.batch_size,
                emb_c,
                stitched_h,
                stitched_w,
            ),
        )

    def input_tile_batch_mib(self) -> float:
        cfg = self.tile_config
        Th, Tw = cfg.effective_tile_size

        n_bytes = cfg.effective_tile_batch * cfg.channels * Th * Tw * self.activation_dtype_bytes

        return bytes_to_mib(n_bytes)

    def feature_batch_mib(self, profile: FeatureProfile) -> float:
        cfg = self.tile_config
        total_bytes = 0

        for shape in profile.feature_shapes.values():
            _, C, H, W = shape
            total_bytes += cfg.effective_tile_batch * C * H * W * self.activation_dtype_bytes

        return bytes_to_mib(total_bytes)

    def embedding_batch_mib(self, profile: FeatureProfile) -> float:
        cfg = self.tile_config
        _, C, H, W = profile.embedding_shape_per_tile

        n_bytes = cfg.effective_tile_batch * C * H * W * self.activation_dtype_bytes

        return bytes_to_mib(n_bytes)

    def activation_peak_mib(self, profile: FeatureProfile) -> float:
        return self.input_tile_batch_mib() + self.feature_batch_mib(profile) + self.embedding_batch_mib(profile)

    def gross_available_mib(
        self,
        memory_budget: MemoryBudget,
        activation_mib: float,
        extra_fixed_mib: float = 0.0,
    ) -> float:
        return (
            memory_budget.memory_limit_mib - memory_budget.reserve_mib - memory_budget.fixed_overhead_mib - activation_mib - extra_fixed_mib
        )

    def effective_budget_mib(
        self,
        memory_budget: MemoryBudget,
        activation_mib: float,
        extra_fixed_mib: float = 0.0,
    ) -> float:
        return max(
            0.0,
            self.gross_available_mib(memory_budget, activation_mib, extra_fixed_mib) * memory_budget.safety_fraction,
        )

    def common_budget_dict(
        self,
        memory_budget: MemoryBudget,
        *,
        activation_mib: float,
        effective_budget_mib: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        budget = {
            "memory_limit_mib": memory_budget.memory_limit_mib,
            "safety_fraction": memory_budget.safety_fraction,
            "reserve_mib": memory_budget.reserve_mib,
            "fixed_overhead_mib": memory_budget.fixed_overhead_mib,
            "activation_peak_mib": activation_mib,
            "usable_mib": memory_budget.usable_mib,
            "effective_budget_mib": effective_budget_mib,
        }
        if extra:
            budget.update(extra)
        return budget

    def common_tiling_dict(self, profile: FeatureProfile) -> dict[str, Any]:
        return {
            "image_size": self.tile_config.image_size,
            "uses_tiling": self.tile_config.uses_tiling,
            "tile_size": self.tile_config.tile_size,
            "stride": self.tile_config.stride,
            "effective_tile_size": self.tile_config.effective_tile_size,
            "effective_stride": self.tile_config.effective_stride,
            "batch_size": self.tile_config.batch_size,
            "n_tiles_h": self.tile_config.n_tiles_h,
            "n_tiles_w": self.tile_config.n_tiles_w,
            "tiles_per_image": self.tile_config.tiles_per_image,
            "effective_tile_batch": self.tile_config.effective_tile_batch,
            "input_tile_batch_mib": self.input_tile_batch_mib(),
            "feature_batch_mib": self.feature_batch_mib(profile),
            "embedding_batch_mib": self.embedding_batch_mib(profile),
        }

    def common_model_dtype_dict(self, profile: FeatureProfile) -> dict[str, Any]:
        return {
            "activation_dtype": str(self.activation_dtype),
            "activation_dtype_bytes": self.activation_dtype_bytes,
            "profiling_dtype": str(self.profiling_dtype),
            "profile_embedding_dtype": profile.embedding_dtype,
            "profile_feature_dtypes": profile.feature_dtypes,
        }

    def estimate(self, *, memory_budget: MemoryBudget, **kwargs) -> MemoryEstimate:
        profile = kwargs.pop("profile", None)
        if profile is None:
            profile = self.profile_features()

        return self.estimate_from_profile(
            profile=profile,
            memory_budget=memory_budget,
            **kwargs,
        )

    @abstractmethod
    def estimate_from_profile(
        self,
        *,
        profile: FeatureProfile,
        memory_budget: MemoryBudget,
        **kwargs,
    ) -> MemoryEstimate:
        raise NotImplementedError

    def calibrate_workspace_factor(
        self,
        *,
        observed_peak_mib: float,
        n_train_images: int,
        profile: FeatureProfile | None = None,
        fixed_overhead_mib: float = 0.0,
    ) -> float:
        """Recover this estimator's workspace factor from an observed training peak.

        Memory-bank estimators (PatchCore, PaDiM) override this to solve for their
        per-image workspace overhead. For estimators whose peak does not scale with
        the training set (e.g. gradient-trained models) calibration is a no-op and
        returns ``0.0`` so generic calibration pipelines can run unconditionally.
        """
        return 0.0
