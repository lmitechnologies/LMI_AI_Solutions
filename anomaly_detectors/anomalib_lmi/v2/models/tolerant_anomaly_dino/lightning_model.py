"""Lightning wrapper for tolerance-aware AnomalyDINO with configurable residual projection heads."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize

from anomalib import LearningType, PrecisionType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import TolerantAnomalyDINOModel

logger = logging.getLogger(__name__)


class TolerantAnomalyDINO(MemoryBankMixin, AnomalibModule):
    """Stock AnomalyDINO plus conservative acceptable/reject residual references.

    The normal detector remains stock-compatible.  Optional held-out diagnostics
    are deliberately independent of anomalib's validation/test split so that
    acceptable and reject reference images can be kept out of evaluation.
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
        residual_topk_per_image: int = 10,
        t_normal: float = 0.1,
        t_known: float = 0.5,
        t_accept_known: float | None = None,
        t_reject_known: float | None = None,
        margin: float = 0.05,
        accept_damping: float = 0.0,
        # calibrated image-level continuous acceptance
        image_accept_enable: bool = True,
        image_accept_threshold: float = float("inf"),
        image_accept_damping: float = 0.0,
        image_reject_veto_enable: bool = True,
        image_reject_veto_threshold: float = 0.0,
        image_accept_calibrate: bool = True,
        image_accept_max_reject_false_accept_rate: float = 0.0,
        image_accept_safety_margin: float = 0.01,
        image_accept_min_threshold: float = 0.0,
        # v8.3: calibrated projected-space whole-image ACCEPT fallback.
        # In dual mode the weaker of direction/magnitude evidence controls the gate.
        projected_image_accept_enable: bool = False,
        projected_image_accept_threshold: float = float("inf"),
        projected_image_accept_damping: float = 0.0,
        projected_image_accept_calibrate: bool = True,
        projected_image_accept_max_reject_false_accept_rate: float = 0.05,
        projected_image_accept_safety_margin: float = 0.01,
        projected_image_accept_min_threshold: float = 0.0,
        # calibrated continuous REJECT evidence (v5)
        image_reject_boost_enable: bool = True,
        image_reject_boost_advantage_threshold: float = 0.0,
        image_reject_boost_lambda: float = 0.0,
        image_reject_boost_max: float = 1.0,
        image_reject_boost_calibrate: bool = True,
        image_reject_boost_target_precision: float = 0.99,
        image_reject_boost_advantage_candidates: tuple[float, ...] | list[float] = (
            -0.30, -0.25, -0.20, -0.15, -0.10, -0.075, -0.05, -0.025, 0.0
        ),
        image_reject_boost_lambda_candidates: tuple[float, ...] | list[float] = (
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0
        ),
        # v6c supervised residual projection (trained on REFERENCE only)
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
        residual_projection_reject_class_depth: int = 1,
        residual_projection_mode: Literal["direction", "magnitude", "dual"] | None = None,
        dual_projection_enable: bool = True,
        residual_projection_use_relative_xy: bool = False,
        residual_projection_xy_scale: float = 1.0,
        residual_projection_intermediate_layers: list[int] | tuple[int, ...] | None = None,
        residual_projection_intermediate_scale: float = 1.0,
        # calibrated projected-space REJECT boosts; direction and magnitude
        # heads are calibrated independently against the same frozen v5 score
        # and fused with max() at inference.
        projected_reject_boost_enable: bool = True,
        projected_reject_boost_advantage_threshold: float = 0.0,
        projected_reject_boost_lambda: float = 0.0,
        projected_reject_boost_max: float = 0.75,
        projected_reject_boost_calibrate: bool = True,
        projected_reject_boost_target_precision: float = 0.999,
        projected_reject_boost_reject_type_weights: dict[str, float] | None = None,
        projected_reject_boost_require_non_decreasing_overall_recall: bool = True,
        projected_reject_boost_advantage_candidates: tuple[float, ...] | list[float] = (
            -0.40, -0.30, -0.25, -0.20, -0.15, -0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05
        ),
        projected_reject_boost_lambda_candidates: tuple[float, ...] | list[float] = (
            0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        ),
        detector_only: bool = False,
        strict_stock_when_damping_one: bool = True,
        emit_patch_class: bool = False,
        residual_bank_chunk_size: int = 65536,
        normal_bank_chunk_size: int = 65536,
        calibration_query_chunk_size: int = 256,
        # calibration
        calibrate: bool = True,
        calibrate_t_normal_percentile: float = 95.0,
        calibrate_t_known_percentile: float = 95.0,
        calibration_normal_samples: int = 4096,
        calibration_seed: int = 1337,
        # tolerance reference data
        normal_reference_dir: str | Path | None = None,
        acceptable_dir: str | Path | None = None,
        reject_dir: str | Path | None = None,
        defect_batch_size: int = 8,
        defect_num_workers: int = 4,
        # image-level calibration data (never added to residual banks)
        image_calibration_good_dir: str | Path | None = None,
        image_calibration_acceptable_dir: str | Path | None = None,
        image_calibration_reject_dir: str | Path | None = None,
        image_calibration_output_dir: str | Path | None = None,
        image_calibration_batch_size: int = 4,
        image_calibration_num_workers: int = 4,
        # held-out diagnostics / final test
        diagnostic_enable: bool = False,
        diagnostic_good_dir: str | Path | None = None,
        diagnostic_acceptable_dir: str | Path | None = None,
        diagnostic_reject_dir: str | Path | None = None,
        diagnostic_output_dir: str | Path | None = None,
        diagnostic_batch_size: int = 4,
        diagnostic_num_workers: int = 4,
        diagnostic_target_precisions: tuple[float, ...] | list[float] = (0.95, 0.99),
        diagnostic_report_threshold: float | None = None,
        diagnostic_fail_on_overlap: bool = True,
        diagnostic_reject_class_depth: int = 1,
        precision: str | PrecisionType = PrecisionType.FLOAT32,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.normal_reference_dir = Path(normal_reference_dir) if normal_reference_dir else None
        self.acceptable_dir = Path(acceptable_dir) if acceptable_dir else None
        self.reject_dir = Path(reject_dir) if reject_dir else None
        self.defect_batch_size = int(defect_batch_size)
        self.defect_num_workers = int(defect_num_workers)
        self.calibrate = bool(calibrate)
        self.calibrate_t_normal_percentile = float(calibrate_t_normal_percentile)
        self.calibrate_t_known_percentile = float(calibrate_t_known_percentile)
        self.calibration_normal_samples = int(calibration_normal_samples)
        self.calibration_seed = int(calibration_seed)

        self.image_accept_calibrate = bool(image_accept_calibrate)
        self.image_accept_max_reject_false_accept_rate = float(image_accept_max_reject_false_accept_rate)
        if not 0.0 <= self.image_accept_max_reject_false_accept_rate <= 1.0:
            raise ValueError("image_accept_max_reject_false_accept_rate must be in [0, 1]")
        self.image_accept_safety_margin = float(image_accept_safety_margin)
        self.image_accept_min_threshold = float(image_accept_min_threshold)

        self.projected_image_accept_calibrate = bool(projected_image_accept_calibrate)
        self.projected_image_accept_max_reject_false_accept_rate = float(
            projected_image_accept_max_reject_false_accept_rate
        )
        if not 0.0 <= self.projected_image_accept_max_reject_false_accept_rate <= 1.0:
            raise ValueError("projected_image_accept_max_reject_false_accept_rate must be in [0, 1]")
        self.projected_image_accept_safety_margin = float(projected_image_accept_safety_margin)
        if self.projected_image_accept_safety_margin < 0.0:
            raise ValueError("projected_image_accept_safety_margin must be >= 0")
        self.projected_image_accept_min_threshold = float(projected_image_accept_min_threshold)

        self.image_reject_boost_calibrate = bool(image_reject_boost_calibrate)
        self.image_reject_boost_target_precision = float(image_reject_boost_target_precision)
        if not 0.0 < self.image_reject_boost_target_precision <= 1.0:
            raise ValueError("image_reject_boost_target_precision must be in (0, 1]")
        self.image_reject_boost_advantage_candidates = tuple(
            sorted(set(float(v) for v in image_reject_boost_advantage_candidates))
        )
        self.image_reject_boost_lambda_candidates = tuple(
            sorted(set(float(v) for v in image_reject_boost_lambda_candidates))
        )
        if not self.image_reject_boost_advantage_candidates:
            raise ValueError("image_reject_boost_advantage_candidates must not be empty")
        if not self.image_reject_boost_lambda_candidates:
            raise ValueError("image_reject_boost_lambda_candidates must not be empty")
        if any(v < 0.0 for v in self.image_reject_boost_lambda_candidates):
            raise ValueError("image_reject_boost_lambda_candidates must be >= 0")
        # Always retain v4/no-boost as a calibration candidate.
        if 0.0 not in self.image_reject_boost_lambda_candidates:
            self.image_reject_boost_lambda_candidates = (0.0,) + self.image_reject_boost_lambda_candidates

        self.residual_projection_reject_class_depth = int(residual_projection_reject_class_depth)
        self.projected_reject_boost_calibrate = bool(projected_reject_boost_calibrate)
        self.projected_reject_boost_target_precision = float(projected_reject_boost_target_precision)
        if not 0.0 < self.projected_reject_boost_target_precision <= 1.0:
            raise ValueError("projected_reject_boost_target_precision must be in (0, 1]")
        type_weights = projected_reject_boost_reject_type_weights or {}
        self.projected_reject_boost_reject_type_weights = {
            str(name): max(0.0, float(weight)) for name, weight in type_weights.items()
        }
        self.projected_reject_boost_require_non_decreasing_overall_recall = bool(
            projected_reject_boost_require_non_decreasing_overall_recall
        )
        self.projected_reject_boost_advantage_candidates = tuple(
            sorted(set(float(v) for v in projected_reject_boost_advantage_candidates))
        )
        self.projected_reject_boost_lambda_candidates = tuple(
            sorted(set(float(v) for v in projected_reject_boost_lambda_candidates))
        )
        if not self.projected_reject_boost_advantage_candidates:
            raise ValueError("projected_reject_boost_advantage_candidates must not be empty")
        if not self.projected_reject_boost_lambda_candidates:
            raise ValueError("projected_reject_boost_lambda_candidates must not be empty")
        if any(v < 0.0 for v in self.projected_reject_boost_lambda_candidates):
            raise ValueError("projected_reject_boost_lambda_candidates must be >= 0")
        if 0.0 not in self.projected_reject_boost_lambda_candidates:
            self.projected_reject_boost_lambda_candidates = (0.0,) + self.projected_reject_boost_lambda_candidates

        self.image_calibration_good_dir = Path(image_calibration_good_dir) if image_calibration_good_dir else None
        self.image_calibration_acceptable_dir = (
            Path(image_calibration_acceptable_dir) if image_calibration_acceptable_dir else None
        )
        self.image_calibration_reject_dir = (
            Path(image_calibration_reject_dir) if image_calibration_reject_dir else None
        )
        self.image_calibration_output_dir = (
            Path(image_calibration_output_dir) if image_calibration_output_dir else None
        )
        self.image_calibration_batch_size = int(image_calibration_batch_size)
        self.image_calibration_num_workers = int(image_calibration_num_workers)

        self.diagnostic_enable = bool(diagnostic_enable)
        self.diagnostic_good_dir = Path(diagnostic_good_dir) if diagnostic_good_dir else None
        self.diagnostic_acceptable_dir = Path(diagnostic_acceptable_dir) if diagnostic_acceptable_dir else None
        self.diagnostic_reject_dir = Path(diagnostic_reject_dir) if diagnostic_reject_dir else None
        self.diagnostic_output_dir = Path(diagnostic_output_dir) if diagnostic_output_dir else None
        self.diagnostic_batch_size = int(diagnostic_batch_size)
        self.diagnostic_num_workers = int(diagnostic_num_workers)
        self.diagnostic_target_precisions = tuple(float(v) for v in diagnostic_target_precisions)
        self.diagnostic_report_threshold = (
            None if diagnostic_report_threshold is None else float(diagnostic_report_threshold)
        )
        self.diagnostic_fail_on_overlap = bool(diagnostic_fail_on_overlap)
        self.diagnostic_reject_class_depth = int(diagnostic_reject_class_depth)

        self.model = TolerantAnomalyDINOModel(
            encoder_name=encoder_name,
            num_neighbours=num_neighbours,
            masking=masking,
            scoring_mode=scoring_mode,
            pca_components=pca_components,
            coreset_subsampling=coreset_subsampling,
            sampling_ratio=sampling_ratio,
            residual_creation_threshold=residual_creation_threshold,
            residual_topk_per_image=residual_topk_per_image,
            t_normal=t_normal,
            t_known=t_known,
            t_accept_known=t_accept_known,
            t_reject_known=t_reject_known,
            margin=margin,
            accept_damping=accept_damping,
            image_accept_enable=image_accept_enable,
            image_accept_threshold=image_accept_threshold,
            image_accept_damping=image_accept_damping,
            projected_image_accept_enable=projected_image_accept_enable,
            projected_image_accept_threshold=projected_image_accept_threshold,
            projected_image_accept_damping=projected_image_accept_damping,
            image_reject_veto_enable=image_reject_veto_enable,
            image_reject_veto_threshold=image_reject_veto_threshold,
            image_reject_boost_enable=image_reject_boost_enable,
            image_reject_boost_advantage_threshold=image_reject_boost_advantage_threshold,
            image_reject_boost_lambda=image_reject_boost_lambda,
            image_reject_boost_max=image_reject_boost_max,
            residual_projection_enable=residual_projection_enable,
            residual_projection_dim=residual_projection_dim,
            residual_projection_margin=residual_projection_margin,
            residual_projection_epochs=residual_projection_epochs,
            residual_projection_steps_per_epoch=residual_projection_steps_per_epoch,
            residual_projection_batch_size=residual_projection_batch_size,
            residual_projection_lr=residual_projection_lr,
            residual_projection_weight_decay=residual_projection_weight_decay,
            residual_projection_orthogonality_weight=residual_projection_orthogonality_weight,
            residual_projection_use_magnitude=residual_projection_use_magnitude,
            residual_projection_magnitude_transform=residual_projection_magnitude_transform,
            residual_projection_magnitude_scale=residual_projection_magnitude_scale,
            residual_projection_magnitude_clip=residual_projection_magnitude_clip,
            residual_projection_magnitude_eps=residual_projection_magnitude_eps,
            residual_projection_reject_type_weights=residual_projection_reject_type_weights,
            residual_projection_seed=residual_projection_seed,
            residual_projection_fit_device=residual_projection_fit_device,
            residual_projection_mode=residual_projection_mode,
            dual_projection_enable=dual_projection_enable,
            residual_projection_use_relative_xy=residual_projection_use_relative_xy,
            residual_projection_xy_scale=residual_projection_xy_scale,
            residual_projection_intermediate_layers=residual_projection_intermediate_layers,
            residual_projection_intermediate_scale=residual_projection_intermediate_scale,
            projected_reject_boost_enable=projected_reject_boost_enable,
            projected_reject_boost_advantage_threshold=projected_reject_boost_advantage_threshold,
            projected_reject_boost_lambda=projected_reject_boost_lambda,
            projected_reject_boost_max=projected_reject_boost_max,
            detector_only=detector_only,
            strict_stock_when_damping_one=strict_stock_when_damping_one,
            emit_patch_class=emit_patch_class,
            residual_bank_chunk_size=residual_bank_chunk_size,
            normal_bank_chunk_size=normal_bank_chunk_size,
            calibration_query_chunk_size=calibration_query_chunk_size,
            calibration_seed=calibration_seed,
        )

        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())
        if precision == PrecisionType.FLOAT16:
            self.model = self.model.half()
        elif precision == PrecisionType.FLOAT32:
            self.model = self.model.float()
        else:
            raise ValueError(
                f"Unsupported precision: {precision}. "
                f"Supported: {PrecisionType.FLOAT16}, {PrecisionType.FLOAT32}."
            )

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | int | None = None,
    ) -> PreProcessor:
        image_size = image_size or (252, 252)
        transform = Compose([
            Resize(image_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return PreProcessor(transform=transform)

    def set_residual_projection_mode(self, mode: str) -> None:
        """Persist a deployment projection mode on the wrapped torch model.

        This is an artifact-configuration operation, not an inference argument.
        After saving a checkpoint/state_dict, subsequent loads restore the mode
        automatically and ONNX export traces only the selected Python branch.
        """
        self.model.set_residual_projection_mode(mode)

    @property
    def residual_projection_mode(self) -> str:
        """Return the currently persisted deployment projection mode."""
        return self.model.residual_projection_mode

    @staticmethod
    def configure_optimizers() -> None:
        return

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        return PostProcessor()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs
        _ = self.model(batch.image)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Finalize the normal bank, build tolerance banks, calibrate, diagnose."""
        logger.info("TolerantAnomalyDINO: finalizing stock AnomalyDINO normal bank...")
        self.model.fit()
        logger.info(
            "Normal bank: shape=%s dtype=%s device=%s",
            tuple(self.model.memory_bank.shape),
            self.model.memory_bank.dtype,
            self.model.memory_bank.device,
        )
        if self.model.residual_projection_intermediate_enabled:
            logger.info(
                "Intermediate normal bank: layers=%s shape=%s dtype=%s",
                self.model.residual_projection_intermediate_layers,
                tuple(self.model.intermediate_memory_bank.shape),
                self.model.intermediate_memory_bank.dtype,
            )

        if self.model.detector_only:
            logger.info("detector_only=True: skipping residual-bank construction/calibration.")
            if self.diagnostic_enable:
                logger.warning("Diagnostics requested in detector_only mode; tolerance fields will be empty/base-only.")
                self._run_heldout_diagnostics()
            return

        transform = self._get_defect_transform()
        for label, dir_path in (("accept", self.acceptable_dir), ("reject", self.reject_dir)):
            if dir_path is None or not dir_path.exists():
                logger.warning("No %s_dir provided or path missing; skipping.", label)
                continue
            logger.info("Building %s residual bank from: %s", label, dir_path)
            self._extract_defect_residuals(dir_path, label, transform)  # type: ignore[arg-type]

        self.model.finalize_residual_banks()

        projection_fit = self.model.fit_residual_projection()
        logger.info("Residual projection training summary: %s", projection_fit)

        if self.calibrate:
            logger.info(
                "Calibrating tolerance thresholds with matched kNN semantics (calibration_seed=%d)...",
                self.calibration_seed,
            )
            with torch.no_grad():
                d_n_sample = self.model.sample_normal_self_distances(
                    max_samples=self.calibration_normal_samples,
                )
                self.model.calibrate_thresholds(
                    val_d_n=d_n_sample,
                    t_normal_percentile=self.calibrate_t_normal_percentile,
                    t_known_percentile=self.calibrate_t_known_percentile,
                )

        if self.model.image_accept_enable and self.image_accept_calibrate:
            self._run_image_accept_calibration()

        if (
            bool(getattr(self.model, "projected_image_accept_enable", False))
            and self.projected_image_accept_calibrate
        ):
            self._run_projected_image_accept_calibration()

        if self.model.image_reject_boost_enable and self.image_reject_boost_calibrate:
            self._run_image_reject_boost_calibration()

        if self.model.projected_reject_boost_enable and self.projected_reject_boost_calibrate:
            self._run_projected_reject_boost_calibration()

        if self.diagnostic_enable:
            self._run_heldout_diagnostics()

    def _get_defect_transform(self):
        if self.pre_processor is not None and hasattr(self.pre_processor, "transform"):
            return self.pre_processor.transform
        image_size = (252, 252)
        return Compose([
            Resize(image_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract_defect_residuals(
        self,
        dir_path: Path,
        label: Literal["accept", "reject"],
        transform,
    ) -> None:
        dataset = _FlatImageDataset(
            dir_path,
            transform=transform,
            return_path=(label == "reject" and self.model.residual_projection_enable),
        )
        if len(dataset) == 0:
            logger.warning("%s: no images found in %s", label, dir_path)
            return

        logger.info("%s residual source images: %d", label, len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=self.defect_batch_size,
            num_workers=self.defect_num_workers,
            shuffle=False,
            pin_memory=False,
        )

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    batch_imgs, batch_paths = batch
                else:
                    batch_imgs, batch_paths = batch, None
                batch_imgs = batch_imgs.to(self.device, dtype=self.model.memory_bank.dtype)
                feats, intermediate_feats = self.model.extract_features_with_context(batch_imgs)
                grid_size = self.model._grid_size_from_input(batch_imgs)
                reject_types = None
                if label == "reject" and batch_paths is not None:
                    reject_types = [
                        _reject_type_from_path(
                            Path(path), dir_path, self.residual_projection_reject_class_depth
                        )
                        for path in batch_paths
                    ]
                self.model.add_defect_features(
                    feats,
                    label,
                    reject_types=reject_types,
                    intermediate_features=intermediate_feats,
                    grid_size=grid_size,
                )

    # ------------------------------------------------------------------
    # Held-out diagnostics
    # ------------------------------------------------------------------

    def _diagnostic_dirs(self) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for source, path in (
            ("good", self.diagnostic_good_dir),
            ("acceptable", self.diagnostic_acceptable_dir),
            ("reject", self.diagnostic_reject_dir),
        ):
            if path is not None and path.exists():
                result[source] = path
        return result

    def _image_calibration_dirs(self) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for source, path in (
            ("good", self.image_calibration_good_dir),
            ("acceptable", self.image_calibration_acceptable_dir),
            ("reject", self.image_calibration_reject_dir),
        ):
            if path is not None and path.exists():
                result[source] = path
        return result

    def _reference_dirs(self) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for source, path in (
            ("good", self.normal_reference_dir),
            ("acceptable", self.acceptable_dir),
            ("reject", self.reject_dir),
        ):
            if path is not None and path.exists():
                result[source] = path
        return result

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, int]:
        st = path.stat()
        return (int(st.st_dev), int(st.st_ino))

    def _assert_no_cross_partition_overlap(self) -> None:
        """Ensure reference, calibration and final-test images are disjoint.

        Resolved path catches symlink aliases and inode identity catches hardlinks.
        Duplicate aliases *within* one partition are not treated as cross-split
        leakage; only an identity occurring in two different partitions fails.
        """
        partitions = {
            "reference": self._reference_dirs(),
            "calibration": self._image_calibration_dirs(),
            "test": self._diagnostic_dirs(),
        }
        seen_inode: dict[tuple[int, int], str] = {}
        seen_path: dict[str, str] = {}
        overlaps: list[str] = []

        for partition, dirs in partitions.items():
            for source, root in dirs.items():
                owner = f"{partition}:{source}"
                for path in _iter_image_paths(root):
                    try:
                        inode = self._file_identity(path)
                        resolved = str(path.resolve())
                    except OSError:
                        continue
                    prev_inode = seen_inode.get(inode)
                    prev_path = seen_path.get(resolved)
                    if prev_inode is not None and not prev_inode.startswith(partition + ":"):
                        overlaps.append(f"{owner}:{path} overlaps {prev_inode}")
                    elif prev_path is not None and not prev_path.startswith(partition + ":"):
                        overlaps.append(f"{owner}:{path} overlaps {prev_path}")
                    else:
                        seen_inode.setdefault(inode, owner)
                        seen_path.setdefault(resolved, owner)
                    if len(overlaps) >= 20:
                        break
                if len(overlaps) >= 20:
                    break
            if len(overlaps) >= 20:
                break

        if overlaps:
            msg = (
                "Reference/calibration/test leakage detected. The same image identity appears across partitions. "
                f"Examples: {overlaps[:5]}"
            )
            if self.diagnostic_fail_on_overlap:
                raise RuntimeError(msg)
            logger.warning(msg)

    def _collect_diagnostic_rows(
        self,
        dirs: dict[str, Path],
        *,
        batch_size: int,
        num_workers: int,
        apply_image_accept: bool,
        apply_reject_boost: bool,
        apply_projected_reject_boost: bool,
        purpose: str,
    ) -> list[dict[str, Any]]:
        transform = self._get_defect_transform()
        rows: list[dict[str, Any]] = []
        self.model.eval()

        with torch.no_grad():
            for source in ("good", "acceptable", "reject"):
                if source not in dirs:
                    continue
                dataset = _FlatImageDataset(dirs[source], transform=transform, return_path=True)
                logger.info("%s source %s: %d images", purpose, source, len(dataset))
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=False,
                )
                for batch_imgs, batch_paths in loader:
                    batch_imgs = batch_imgs.to(self.device, dtype=self.model.memory_bank.dtype)
                    _, diag = self.model.forward_with_diagnostics(
                        batch_imgs,
                        apply_image_accept=apply_image_accept,
                        apply_reject_boost=apply_reject_boost,
                        apply_projected_reject_boost=apply_projected_reject_boost,
                    )
                    diag_cpu = {k: v.detach().float().cpu().numpy() for k, v in diag.items()}
                    for i, image_path in enumerate(batch_paths):
                        row: dict[str, Any] = {
                            "source": source,
                            "label": 1 if source == "reject" else 0,
                            "path": str(image_path),
                            "reject_type": (
                                _reject_type_from_path(
                                    Path(image_path),
                                    dirs[source],
                                    self.diagnostic_reject_class_depth,
                                )
                                if source == "reject"
                                else ""
                            ),
                        }
                        for key, values in diag_cpu.items():
                            # Diagnostics written to the image-level CSV must be
                            # one scalar per image.  Squeeze singleton dimensions
                            # (e.g. [B, 1]) but never silently reduce a true vector
                            # diagnostic, since doing so would corrupt calibration.
                            sample = np.asarray(values[i]).squeeze()
                            if sample.ndim != 0:
                                raise RuntimeError(
                                    f"Diagnostic '{key}' must be scalar per image; "
                                    f"got batch array shape={values.shape}, "
                                    f"sample shape={np.asarray(values[i]).shape}."
                                )
                            value = float(sample)
                            row[key] = None if math.isnan(value) or math.isinf(value) else value
                        rows.append(row)
        return rows

    def _run_image_accept_calibration(self) -> None:
        dirs = self._image_calibration_dirs()
        missing = {"acceptable", "reject"} - set(dirs)
        if missing:
            logger.warning(
                "Image-level acceptance calibration skipped; missing calibration dirs: %s. "
                "Image-level acceptance remains disabled by an infinite threshold.",
                sorted(missing),
            )
            self.model.image_accept_threshold.fill_(float("inf"))
            return

        self._assert_no_cross_partition_overlap()
        logger.info("Running image-level ACCEPT calibration (image suppression disabled during calibration)...")
        rows = self._collect_diagnostic_rows(
            dirs,
            batch_size=self.image_calibration_batch_size,
            num_workers=self.image_calibration_num_workers,
            apply_image_accept=False,
            apply_reject_boost=False,
            apply_projected_reject_boost=False,
            purpose="image-accept-calibration",
        )
        if not rows:
            logger.warning("Image-level acceptance calibration produced no rows.")
            self.model.image_accept_threshold.fill_(float("inf"))
            return

        calibration = _calibrate_image_accept_rule(
            rows,
            max_reject_false_accept_rate=self.image_accept_max_reject_false_accept_rate,
            safety_margin=self.image_accept_safety_margin,
            min_threshold=self.image_accept_min_threshold,
        )
        calibration["calibration_seed"] = int(self.calibration_seed)
        threshold = calibration["threshold"]
        self.model.image_accept_threshold.fill_(float(threshold))

        out_dir = self.image_calibration_output_dir or self.diagnostic_output_dir
        if out_dir is None:
            trainer_root = getattr(getattr(self, "trainer", None), "default_root_dir", None)
            out_dir = Path(trainer_root or "/app/out") / "tolerance_diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "image_accept_calibration.csv"
        json_path = out_dir / "image_accept_calibration.json"
        self._write_diagnostic_csv(csv_path, rows)
        json_path.write_text(json.dumps(calibration, indent=2, sort_keys=True))

        logger.info(
            "IMAGE-CALIBRATION threshold=%.6f acceptable_accept_rate=%.4f (%d/%d) "
            "reject_false_accept_rate=%.4f (%d/%d) safety_margin=%.6f",
            calibration["threshold"],
            calibration["acceptable_accept_rate"],
            calibration["acceptable_accepted"],
            calibration["acceptable_count"],
            calibration["reject_false_accept_rate"],
            calibration["reject_false_accepted"],
            calibration["reject_count"],
            calibration["safety_margin"],
        )
        logger.info("Image calibration CSV: %s", csv_path)
        logger.info("Image calibration JSON: %s", json_path)

    def _run_projected_image_accept_calibration(self) -> None:
        """Calibrate v8.3 projected whole-image ACCEPT on calibration images only.

        The projected gate is calibrated *after* the raw ACCEPT threshold is frozen
        and before either REJECT boost is calibrated.  The safety constraint is
        applied to the union of raw OR projected ACCEPT on reject calibration
        images, so the new fallback cannot silently consume a second independent
        false-accept budget.
        """
        dirs = self._image_calibration_dirs()
        missing = {"acceptable", "reject"} - set(dirs)
        if missing:
            logger.warning(
                "Projected image ACCEPT calibration skipped; missing calibration dirs: %s. "
                "Projected image acceptance remains disabled by an infinite threshold.",
                sorted(missing),
            )
            self.model.projected_image_accept_threshold.fill_(float("inf"))
            return

        mode = self.model.residual_projection_mode
        direction_available = self.model.residual_projection_direction_weight.numel() > 0
        magnitude_available = self.model.residual_projection_magnitude_weight.numel() > 0
        mode_available = (
            (mode == "direction" and direction_available)
            or (mode == "magnitude" and magnitude_available)
            or (mode == "dual" and direction_available and magnitude_available)
        )
        if not mode_available:
            logger.warning(
                "Projected image ACCEPT calibration skipped; required projected head(s) "
                "are unavailable for mode=%s (direction=%s magnitude=%s).",
                mode, direction_available, magnitude_available,
            )
            self.model.projected_image_accept_threshold.fill_(float("inf"))
            return

        self._assert_no_cross_partition_overlap()
        logger.info(
            "Running projected image ACCEPT calibration: mode=%s with raw ACCEPT frozen; "
            "all ACCEPT/REJECT score modifications disabled during evidence collection...",
            mode,
        )
        rows = self._collect_diagnostic_rows(
            dirs,
            batch_size=self.image_calibration_batch_size,
            num_workers=self.image_calibration_num_workers,
            apply_image_accept=False,
            apply_reject_boost=False,
            apply_projected_reject_boost=False,
            purpose="projected-image-accept-calibration",
        )
        if not rows:
            logger.warning("Projected image ACCEPT calibration produced no rows.")
            self.model.projected_image_accept_threshold.fill_(float("inf"))
            return

        calibration = _calibrate_projected_image_accept_rule(
            rows,
            mode=mode,
            max_combined_reject_false_accept_rate=self.projected_image_accept_max_reject_false_accept_rate,
            safety_margin=self.projected_image_accept_safety_margin,
            min_threshold=self.projected_image_accept_min_threshold,
            raw_image_accept_enable=bool(self.model.image_accept_enable),
            raw_image_accept_threshold=float(self.model.image_accept_threshold.detach().float().cpu()),
            raw_image_reject_veto_enable=bool(self.model.image_reject_veto_enable),
            raw_image_reject_veto_threshold=float(
                self.model.image_reject_veto_threshold.detach().float().cpu()
            ),
        )
        calibration["calibration_seed"] = int(self.calibration_seed)
        self.model.projected_image_accept_threshold.fill_(float(calibration["threshold"]))

        out_dir = self.image_calibration_output_dir or self.diagnostic_output_dir
        if out_dir is None:
            trainer_root = getattr(getattr(self, "trainer", None), "default_root_dir", None)
            out_dir = Path(trainer_root or "/app/out") / "tolerance_diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "projected_image_accept_calibration.csv"
        json_path = out_dir / "projected_image_accept_calibration.json"
        self._write_diagnostic_csv(csv_path, rows)
        json_path.write_text(json.dumps(calibration, indent=2, sort_keys=True))

        logger.info(
            "PROJECTED-IMAGE-ACCEPT-CALIBRATION mode=%s threshold=%.6f "
            "acceptable_projected_rate=%.4f (%d/%d) projected_reject_rate=%.4f (%d/%d) "
            "combined_reject_false_accept_rate=%.4f (%d/%d) raw_baseline_rate=%.4f status=%s",
            mode,
            calibration["threshold"],
            calibration["acceptable_projected_accept_rate"],
            calibration["acceptable_projected_accepted"],
            calibration["acceptable_count"],
            calibration["reject_projected_false_accept_rate"],
            calibration["reject_projected_false_accepted"],
            calibration["reject_count"],
            calibration["combined_reject_false_accept_rate"],
            calibration["combined_reject_false_accepted"],
            calibration["reject_count"],
            calibration["raw_reject_false_accept_rate"],
            calibration["status"],
        )
        logger.info("Projected image ACCEPT calibration CSV: %s", csv_path)
        logger.info("Projected image ACCEPT calibration JSON: %s", json_path)

    def _run_image_reject_boost_calibration(self) -> None:
        """Calibrate v5 continuous REJECT boost using calibration images only."""
        dirs = self._image_calibration_dirs()
        missing = {"good", "acceptable", "reject"} - set(dirs)
        if missing:
            logger.warning(
                "Image-level REJECT boost calibration skipped; missing calibration dirs: %s. "
                "Keeping configured boost parameters.",
                sorted(missing),
            )
            return

        self._assert_no_cross_partition_overlap()
        logger.info(
            "Running image-level REJECT boost calibration with ACCEPT rule frozen "
            "(reject boost disabled during candidate scoring)..."
        )
        rows = self._collect_diagnostic_rows(
            dirs,
            batch_size=self.image_calibration_batch_size,
            num_workers=self.image_calibration_num_workers,
            apply_image_accept=True,
            apply_reject_boost=False,
            apply_projected_reject_boost=False,
            purpose="image-reject-boost-calibration",
        )
        if not rows:
            logger.warning("Image-level REJECT boost calibration produced no rows.")
            return

        calibration = _calibrate_image_reject_boost_rule(
            rows,
            target_precision=self.image_reject_boost_target_precision,
            advantage_candidates=self.image_reject_boost_advantage_candidates,
            lambda_candidates=self.image_reject_boost_lambda_candidates,
            max_boost=float(self.model.image_reject_boost_max.detach().float().cpu()),
        )
        calibration["calibration_seed"] = int(self.calibration_seed)
        self.model.image_reject_boost_advantage_threshold.fill_(
            float(calibration["advantage_threshold"])
        )
        self.model.image_reject_boost_lambda.fill_(float(calibration["lambda"]))

        out_dir = self.image_calibration_output_dir or self.diagnostic_output_dir
        if out_dir is None:
            trainer_root = getattr(getattr(self, "trainer", None), "default_root_dir", None)
            out_dir = Path(trainer_root or "/app/out") / "tolerance_diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "image_reject_boost_calibration.csv"
        json_path = out_dir / "image_reject_boost_calibration.json"
        self._write_diagnostic_csv(csv_path, rows)
        json_path.write_text(json.dumps(calibration, indent=2, sort_keys=True))

        op = calibration["selected_operating_point"]
        logger.info(
            "REJECT-BOOST-CALIBRATION target_P=%.4f thresholdA=%.6f lambda=%.6f "
            "met=%s P=%.4f R=%.4f FP=%d TP=%d baseline_R=%.4f",
            calibration["target_precision"],
            calibration["advantage_threshold"],
            calibration["lambda"],
            op["met"],
            op["precision"],
            op["recall"],
            op["confusion_matrix"]["FP"],
            op["confusion_matrix"]["TP"],
            calibration["baseline_operating_point"]["recall"],
        )
        logger.info("Reject boost calibration CSV: %s", csv_path)
        logger.info("Reject boost calibration JSON: %s", json_path)

    def _run_projected_reject_boost_calibration(self) -> None:
        """Calibrate both projected heads independently of runtime/export mode.

        ``residual_projection_mode`` only controls inference/export execution. During
        calibration both fitted heads are temporarily enabled so their thresholds and
        lambdas are derived from the exact same frozen-v5 rows.
        """
        dirs = self._image_calibration_dirs()
        missing = {"good", "acceptable", "reject"} - set(dirs)
        mode = self.model.residual_projection_mode
        if missing:
            logger.warning(
                "Projected REJECT boost calibration skipped; missing calibration dirs: %s.",
                sorted(missing),
            )
            return

        direction_available = self.model.residual_projection_direction_weight.numel() > 0
        magnitude_available = self.model.residual_projection_magnitude_weight.numel() > 0
        if not (direction_available and magnitude_available):
            logger.warning(
                "Mode-invariant projected calibration requires both fitted heads "
                "(direction=%s magnitude=%s).", direction_available, magnitude_available
            )
            return

        self._assert_no_cross_partition_overlap()
        logger.info(
            "Running mode-invariant projected REJECT calibration: runtime_mode=%s with v5 frozen...",
            mode,
        )
        # Temporarily execute both heads while collecting calibration diagnostics.
        # Restore the requested runtime/export mode immediately afterward.
        requested_mode = mode
        self.model.set_residual_projection_mode("dual")
        try:
            rows = self._collect_diagnostic_rows(
                dirs,
                batch_size=self.image_calibration_batch_size,
                num_workers=self.image_calibration_num_workers,
                apply_image_accept=True,
                apply_reject_boost=True,
                apply_projected_reject_boost=False,
                purpose="all-heads-projected-reject-boost-calibration",
            )
        finally:
            self.model.set_residual_projection_mode(requested_mode)
        if not rows:
            logger.warning("Projected REJECT boost calibration produced no rows for mode=%s.", mode)
            return

        common = dict(
            target_precision=self.projected_reject_boost_target_precision,
            advantage_candidates=self.projected_reject_boost_advantage_candidates,
            lambda_candidates=self.projected_reject_boost_lambda_candidates,
            max_boost=float(self.model.projected_reject_boost_max.detach().float().cpu()),
            reject_type_weights=self.projected_reject_boost_reject_type_weights,
            require_non_decreasing_overall_recall=self.projected_reject_boost_require_non_decreasing_overall_recall,
        )

        direction_calibration = _calibrate_projected_reject_boost_rule(
            rows,
            advantage_key="top_mean_direction_projected_accept_advantage",
            **common,
        )
        direction_calibration["calibration_seed"] = int(self.calibration_seed)
        direction_calibration["fit_policy"] = "both_heads_mode_invariant"
        self.model.direction_projected_reject_boost_advantage_threshold.fill_(
            float(direction_calibration["advantage_threshold"])
        )
        self.model.direction_projected_reject_boost_lambda.fill_(
            float(direction_calibration["lambda"])
        )

        magnitude_calibration = _calibrate_projected_reject_boost_rule(
            rows,
            advantage_key="top_mean_magnitude_projected_accept_advantage",
            **common,
        )
        magnitude_calibration["calibration_seed"] = int(self.calibration_seed)
        magnitude_calibration["fit_policy"] = "both_heads_mode_invariant"
        self.model.magnitude_projected_reject_boost_advantage_threshold.fill_(
            float(magnitude_calibration["advantage_threshold"])
        )
        self.model.magnitude_projected_reject_boost_lambda.fill_(
            float(magnitude_calibration["lambda"])
        )

        # Backward-compatible projected_* fields mirror the active single head.
        # In dual mode retain the historical magnitude alias; final inference still
        # uses max(direction_boost, magnitude_boost).
        if mode == "direction":
            self.model.projected_reject_boost_advantage_threshold.copy_(
                self.model.direction_projected_reject_boost_advantage_threshold
            )
            self.model.projected_reject_boost_lambda.copy_(
                self.model.direction_projected_reject_boost_lambda
            )
        else:
            self.model.projected_reject_boost_advantage_threshold.copy_(
                self.model.magnitude_projected_reject_boost_advantage_threshold
            )
            self.model.projected_reject_boost_lambda.copy_(
                self.model.magnitude_projected_reject_boost_lambda
            )

        labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
        v5_scores = np.asarray(
            [float(r.get("v5_pred_score", r["final_pred_score"])) for r in rows],
            dtype=np.float64,
        )
        dir_adv = np.asarray([
            float(r["top_mean_direction_projected_accept_advantage"])
            if r.get("top_mean_direction_projected_accept_advantage") is not None
            else float("nan")
            for r in rows
        ], dtype=np.float64)
        mag_adv = np.asarray([
            float(r["top_mean_magnitude_projected_accept_advantage"])
            if r.get("top_mean_magnitude_projected_accept_advantage") is not None
            else float("nan")
            for r in rows
        ], dtype=np.float64)

        _dir_scores, _dir_evidence, dir_boost = _apply_reject_boost_numpy(
            v5_scores,
            dir_adv,
            advantage_threshold=float(direction_calibration["advantage_threshold"]),
            boost_lambda=float(direction_calibration["lambda"]),
            max_boost=float(self.model.projected_reject_boost_max.detach().float().cpu()),
        )
        _mag_scores, _mag_evidence, mag_boost = _apply_reject_boost_numpy(
            v5_scores,
            mag_adv,
            advantage_threshold=float(magnitude_calibration["advantage_threshold"]),
            boost_lambda=float(magnitude_calibration["lambda"]),
            max_boost=float(self.model.projected_reject_boost_max.detach().float().cpu()),
        )

        # Calibrate dual fusion safety without mutating either head's standalone
        # threshold/lambda. Store only exported 0/1 fusion scales.
        self.model.dual_projection_direction_fusion_scale.fill_(1.0)
        self.model.dual_projection_magnitude_fusion_scale.fill_(1.0)
        dual_boost = np.maximum(dir_boost, mag_boost)
        dual_scores = v5_scores + dual_boost
        dual_op = _target_precision_operating_point(
            labels, dual_scores, float(self.projected_reject_boost_target_precision)
        )
        fusion_fallback = None
        if not bool(dual_op.get("met", False)):
            d_obj = float(direction_calibration.get(
                "selected_weighted_objective",
                direction_calibration["selected_operating_point"]["recall"],
            ))
            m_obj = float(magnitude_calibration.get(
                "selected_weighted_objective",
                magnitude_calibration["selected_operating_point"]["recall"],
            ))
            if d_obj >= m_obj:
                self.model.dual_projection_magnitude_fusion_scale.fill_(0.0)
                safe_dual_boost = dir_boost
                fusion_fallback = "direction_only"
            else:
                self.model.dual_projection_direction_fusion_scale.fill_(0.0)
                safe_dual_boost = mag_boost
                fusion_fallback = "magnitude_only"
            dual_scores = v5_scores + safe_dual_boost
            dual_op = _target_precision_operating_point(
                labels, dual_scores, float(self.projected_reject_boost_target_precision)
            )
            logger.warning(
                "Dual max fusion missed calibration precision target; fallback=%s P=%.4f R=%.4f",
                fusion_fallback, dual_op["precision"], dual_op["recall"],
            )

        # Keep legacy projected_* aliases descriptive without mutating either
        # standalone head calibration.
        if mode == "dual" and fusion_fallback == "direction_only":
            self.model.projected_reject_boost_advantage_threshold.copy_(
                self.model.direction_projected_reject_boost_advantage_threshold
            )
            self.model.projected_reject_boost_lambda.copy_(
                self.model.direction_projected_reject_boost_lambda
            )
        elif mode == "dual" and fusion_fallback == "magnitude_only":
            self.model.projected_reject_boost_advantage_threshold.copy_(
                self.model.magnitude_projected_reject_boost_advantage_threshold
            )
            self.model.projected_reject_boost_lambda.copy_(
                self.model.magnitude_projected_reject_boost_lambda
            )

        if mode == "direction":
            fused_boost = dir_boost
            fusion_name = "direction_only"
        elif mode == "magnitude":
            fused_boost = mag_boost
            fusion_name = "magnitude_only"
        else:
            fused_boost = np.maximum(
                dir_boost * float(self.model.dual_projection_direction_fusion_scale.detach().cpu()),
                mag_boost * float(self.model.dual_projection_magnitude_fusion_scale.detach().cpu()),
            )
            fusion_name = "max_boost" if fusion_fallback is None else fusion_fallback

        fused_scores = v5_scores + fused_boost
        fused_op = _target_precision_operating_point(
            labels, fused_scores, float(self.projected_reject_boost_target_precision)
        )

        out_dir = self.image_calibration_output_dir or self.diagnostic_output_dir
        if out_dir is None:
            trainer_root = getattr(getattr(self, "trainer", None), "default_root_dir", None)
            out_dir = Path(trainer_root or "/app/out") / "tolerance_diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)

        mode_csv = out_dir / f"{mode}_projected_reject_boost_calibration.csv"
        aggregate_json = out_dir / "projected_reject_boost_calibration.json"
        self._write_diagnostic_csv(mode_csv, rows)
        (out_dir / "direction_projected_reject_boost_calibration.json").write_text(
            json.dumps(direction_calibration, indent=2, sort_keys=True)
        )
        (out_dir / "magnitude_projected_reject_boost_calibration.json").write_text(
            json.dumps(magnitude_calibration, indent=2, sort_keys=True)
        )

        aggregate_payload = {
            "mode": mode,
            "fusion": fusion_name,
            "target_precision": float(self.projected_reject_boost_target_precision),
            "calibration_seed": int(self.calibration_seed),
            "direction": direction_calibration,
            "magnitude": magnitude_calibration,
            "selected_operating_point": fused_op,
            "fusion_fallback": fusion_fallback,
            "fit_policy": "both_heads_mode_invariant",
            "fit_device": str(self.model.residual_projection_fit_device),
            "dual_operating_point": dual_op,
            "dual_direction_fusion_scale": float(self.model.dual_projection_direction_fusion_scale.detach().cpu()),
            "dual_magnitude_fusion_scale": float(self.model.dual_projection_magnitude_fusion_scale.detach().cpu()),
        }
        aggregate_json.write_text(json.dumps(aggregate_payload, indent=2, sort_keys=True))
        if mode == "dual":
            (out_dir / "dual_projected_reject_boost_calibration.json").write_text(
                json.dumps(aggregate_payload, indent=2, sort_keys=True)
            )

        if direction_available:
            d_op = direction_calibration["selected_operating_point"]
            logger.info(
                "DIRECTION-PROJECTION-CALIBRATION A0=%.6f lambda=%.6f P=%.4f R=%.4f FP=%d TP=%d",
                direction_calibration["advantage_threshold"],
                direction_calibration["lambda"],
                d_op["precision"], d_op["recall"],
                d_op["confusion_matrix"]["FP"], d_op["confusion_matrix"]["TP"],
            )
        if magnitude_available:
            m_op = magnitude_calibration["selected_operating_point"]
            logger.info(
                "MAGNITUDE-PROJECTION-CALIBRATION A0=%.6f lambda=%.6f P=%.4f R=%.4f FP=%d TP=%d",
                magnitude_calibration["advantage_threshold"],
                magnitude_calibration["lambda"],
                m_op["precision"], m_op["recall"],
                m_op["confusion_matrix"]["FP"], m_op["confusion_matrix"]["TP"],
            )
        logger.info(
            "PROJECTED-CALIBRATION mode=%s fusion=%s P=%.4f R=%.4f FP=%d TP=%d met=%s",
            mode, fusion_name,
            fused_op["precision"], fused_op["recall"],
            fused_op["confusion_matrix"]["FP"], fused_op["confusion_matrix"]["TP"],
            fused_op["met"],
        )
        logger.info("Projected boost calibration JSON: %s", aggregate_json)

    def _run_heldout_diagnostics(self) -> None:
        dirs = self._diagnostic_dirs()
        missing = {"good", "acceptable", "reject"} - set(dirs)
        if missing:
            logger.warning("Held-out diagnostics skipped; missing source dirs: %s", sorted(missing))
            return

        self._assert_no_cross_partition_overlap()
        logger.info(
            "Running FINAL held-out tolerance diagnostics with frozen image_accept_threshold=%.6f...",
            float(self.model.image_accept_threshold.detach().float().cpu()),
        )
        rows = self._collect_diagnostic_rows(
            dirs,
            batch_size=self.diagnostic_batch_size,
            num_workers=self.diagnostic_num_workers,
            apply_image_accept=True,
            apply_reject_boost=True,
            apply_projected_reject_boost=True,
            purpose="final-test",
        )

        if not rows:
            logger.warning("Held-out diagnostics produced no rows.")
            return

        out_dir = self.diagnostic_output_dir
        if out_dir is None:
            trainer_root = getattr(getattr(self, "trainer", None), "default_root_dir", None)
            out_dir = Path(trainer_root or "/app/out") / "tolerance_diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "tolerance_diagnostics.csv"
        json_path = out_dir / "tolerance_diagnostics.json"
        self._write_diagnostic_csv(csv_path, rows)
        summary = self._summarize_diagnostics(rows)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        logger.info("Tolerance diagnostic CSV: %s", csv_path)
        logger.info("Tolerance diagnostic JSON: %s", json_path)
        self._log_diagnostic_headline(summary)

    @staticmethod
    def _write_diagnostic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _summarize_diagnostics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
        base = np.asarray([float(r["base_pred_score"]) for r in rows], dtype=np.float64)
        patch = np.asarray([float(r.get("patch_pred_score", r["final_pred_score"])) for r in rows], dtype=np.float64)
        veto = np.asarray([
            float(r.get("veto_pred_score", r.get("patch_pred_score", r["final_pred_score"]))) for r in rows
        ], dtype=np.float64)
        accept = np.asarray([
            float(r.get("accept_pred_score", r.get("veto_pred_score", r["final_pred_score"]))) for r in rows
        ], dtype=np.float64)
        v5 = np.asarray([float(r.get("v5_pred_score", r["final_pred_score"])) for r in rows], dtype=np.float64)
        direction_projected = np.asarray([float(r.get("direction_projected_pred_score", r.get("v5_pred_score", r["final_pred_score"]))) for r in rows], dtype=np.float64)
        magnitude_projected = np.asarray([float(r.get("magnitude_projected_pred_score", r.get("v5_pred_score", r["final_pred_score"]))) for r in rows], dtype=np.float64)
        final = np.asarray([float(r["final_pred_score"]) for r in rows], dtype=np.float64)
        sources = np.asarray([str(r["source"]) for r in rows], dtype=object)
        reject_types = np.asarray([str(r.get("reject_type") or "") for r in rows], dtype=object)

        stage_scores = {
            "base": base,
            "patch": patch,
            "veto": veto,
            "accept": accept,
            "v5": v5,
            "direction_projected": direction_projected,
            "magnitude_projected": magnitude_projected,
            "final": final,
        }

        summary: dict[str, Any] = {
            "counts": {source: int(np.sum(sources == source)) for source in ("good", "acceptable", "reject")},
            "random_seeds": {
                "calibration_seed": int(self.calibration_seed),
                "residual_projection_seed": int(self.model.residual_projection_seed),
            },
            "projection_fitting": {
                "policy": "both_heads_mode_invariant",
                "fit_device": str(self.model.residual_projection_fit_device),
            },
            "model_thresholds": {
                "t_normal": float(self.model.t_normal.detach().float().cpu()),
                "t_accept_known": float(self.model.t_accept_known.detach().float().cpu()),
                "t_reject_known": float(self.model.t_reject_known.detach().float().cpu()),
                "margin": float(self.model.margin.detach().float().cpu()),
                "accept_damping": float(self.model.accept_damping.detach().float().cpu()),
                "image_accept_enable": bool(self.model.image_accept_enable),
                "image_accept_threshold": float(self.model.image_accept_threshold.detach().float().cpu()),
                "image_accept_damping": float(self.model.image_accept_damping.detach().float().cpu()),
                "projected_image_accept_enable": bool(
                    getattr(self.model, "projected_image_accept_enable", False)
                ),
                "projected_image_accept_threshold": float(
                    getattr(self.model, "projected_image_accept_threshold", self.model.image_accept_threshold)
                    .detach().float().cpu()
                ),
                "projected_image_accept_damping": float(
                    getattr(self.model, "projected_image_accept_damping", self.model.image_accept_damping)
                    .detach().float().cpu()
                ),
                "image_reject_veto_enable": bool(self.model.image_reject_veto_enable),
                "image_reject_veto_threshold": float(self.model.image_reject_veto_threshold.detach().float().cpu()),
                "image_reject_boost_enable": bool(self.model.image_reject_boost_enable),
                "image_reject_boost_advantage_threshold": float(
                    self.model.image_reject_boost_advantage_threshold.detach().float().cpu()
                ),
                "image_reject_boost_lambda": float(self.model.image_reject_boost_lambda.detach().float().cpu()),
                "image_reject_boost_max": float(self.model.image_reject_boost_max.detach().float().cpu()),
                "residual_projection_enable": bool(self.model.residual_projection_enable),
                "residual_projection_mode": str(self.model.residual_projection_mode),
                "direction_projection_enabled": bool(self.model.direction_projection_enabled),
                "magnitude_projection_enabled": bool(self.model.magnitude_projection_enabled),
                "dual_projection_enable": bool(self.model.dual_projection_enable),
                "dual_projection_direction_fusion_scale": float(
                    self.model.dual_projection_direction_fusion_scale.detach().float().cpu()
                ),
                "dual_projection_magnitude_fusion_scale": float(
                    self.model.dual_projection_magnitude_fusion_scale.detach().float().cpu()
                ),
                "direction_projection_dim": int(self.model.residual_projection_direction_weight.shape[0]) if self.model.residual_projection_direction_weight.ndim == 2 else 0,
                "direction_projection_input_dim": int(self.model.residual_projection_direction_weight.shape[1]) if self.model.residual_projection_direction_weight.ndim == 2 else 0,
                "magnitude_projection_dim": int(self.model.residual_projection_magnitude_weight.shape[0]) if self.model.residual_projection_magnitude_weight.ndim == 2 else 0,
                "magnitude_projection_input_dim": int(self.model.residual_projection_magnitude_weight.shape[1]) if self.model.residual_projection_magnitude_weight.ndim == 2 else 0,
                "residual_projection_dim": int(self.model.residual_projection_weight.shape[0]) if self.model.residual_projection_weight.ndim == 2 else 0,
                "residual_projection_input_dim": int(self.model.residual_projection_weight.shape[1]) if self.model.residual_projection_weight.ndim == 2 else 0,
                "residual_projection_use_magnitude": bool(self.model.residual_projection_use_magnitude),
                "residual_projection_use_relative_xy": bool(self.model.residual_projection_use_relative_xy),
                "residual_projection_xy_scale": float(self.model.residual_projection_xy_scale),
                "residual_projection_intermediate_layers": list(self.model.residual_projection_intermediate_layers),
                "residual_projection_intermediate_scale": float(self.model.residual_projection_intermediate_scale),
                "residual_projection_magnitude_transform": str(self.model.residual_projection_magnitude_transform),
                "residual_projection_magnitude_scale": float(self.model.residual_projection_magnitude_scale),
                "residual_projection_magnitude_clip": float(self.model.residual_projection_magnitude_clip),
                "residual_projection_magnitude_mean": float(self.model.residual_projection_magnitude_mean.detach().float().cpu()),
                "residual_projection_magnitude_std": float(self.model.residual_projection_magnitude_std.detach().float().cpu()),
                "projected_reject_boost_enable": bool(self.model.projected_reject_boost_enable),
                "direction_projected_reject_boost_advantage_threshold": float(self.model.direction_projected_reject_boost_advantage_threshold.detach().float().cpu()),
                "direction_projected_reject_boost_lambda": float(self.model.direction_projected_reject_boost_lambda.detach().float().cpu()),
                "magnitude_projected_reject_boost_advantage_threshold": float(self.model.magnitude_projected_reject_boost_advantage_threshold.detach().float().cpu()),
                "magnitude_projected_reject_boost_lambda": float(self.model.magnitude_projected_reject_boost_lambda.detach().float().cpu()),
                "projected_reject_boost_advantage_threshold": float(self.model.projected_reject_boost_advantage_threshold.detach().float().cpu()),
                "projected_reject_boost_lambda": float(self.model.projected_reject_boost_lambda.detach().float().cpu()),
                "projected_reject_boost_max": float(self.model.projected_reject_boost_max.detach().float().cpu()),
            },
            "bank_sizes": {
                "normal": int(self.model.memory_bank.shape[0]),
                "normal_intermediate": int(self.model.intermediate_memory_bank.shape[0]) if self.model.intermediate_memory_bank.ndim == 3 else 0,
                "normal_intermediate_layers": int(self.model.intermediate_memory_bank.shape[1]) if self.model.intermediate_memory_bank.ndim == 3 else 0,
                "acceptable_residual": int(self.model.accept_dir_bank.shape[0]) if self.model.accept_dir_bank.ndim > 1 else 0,
                "reject_residual": int(self.model.reject_dir_bank.shape[0]) if self.model.reject_dir_bank.ndim > 1 else 0,
            },
            "score_stats_by_source": {},
            "suppression_by_source": {},
            "operating_points": {name: {} for name in stage_scores},
            "same_threshold_effects": {},
            "reject_type_diagnostics": {},
        }

        for source in ("good", "acceptable", "reject"):
            mask = sources == source
            summary["score_stats_by_source"][source] = {
                name: _score_stats(scores[mask]) for name, scores in stage_scores.items()
            }
            total_reduction = base[mask] - final[mask]
            patch_reductions = base[mask] - patch[mask]
            veto_restorations = veto[mask] - patch[mask]
            image_reductions = veto[mask] - accept[mask]
            boost_increases = v5[mask] - accept[mask]
            projected_boost_increases = final[mask] - v5[mask]
            source_rows = [row for row in rows if row["source"] == source]
            summary["suppression_by_source"][source] = {
                "images_net_reduced": int(np.sum(total_reduction > 1e-7)),
                "images_net_increased": int(np.sum(total_reduction < -1e-7)),
                "mean_net_score_change_final_minus_base": float(np.mean(final[mask] - base[mask])) if np.any(mask) else 0.0,
                "mean_patch_score_reduction": float(np.mean(patch_reductions)) if patch_reductions.size else 0.0,
                "mean_veto_score_restoration": float(np.mean(veto_restorations)) if veto_restorations.size else 0.0,
                "mean_image_accept_score_reduction": float(np.mean(image_reductions)) if image_reductions.size else 0.0,
                "mean_reject_boost_score_increase": float(np.mean(boost_increases)) if boost_increases.size else 0.0,
                "mean_projected_boost_score_increase": float(np.mean(projected_boost_increases)) if projected_boost_increases.size else 0.0,
                "images_reject_vetoed": int(sum((r.get("image_reject_veto_applied") or 0) > 0 for r in source_rows)),
                "reject_veto_rate": float(np.mean([(r.get("image_reject_veto_applied") or 0) > 0 for r in source_rows])) if source_rows else 0.0,
                "images_image_accepted": int(sum((r.get("image_accept_applied") or 0) > 0 for r in source_rows)),
                "image_accept_rate": float(np.mean([(r.get("image_accept_applied") or 0) > 0 for r in source_rows])) if source_rows else 0.0,
                "raw_image_accept_rate": float(np.mean([(r.get("raw_image_accept_applied") or 0) > 0 for r in source_rows])) if source_rows else 0.0,
                "projected_image_accept_rate": float(np.mean([(r.get("projected_image_accept_applied") or 0) > 0 for r in source_rows])) if source_rows else 0.0,
                "images_projected_image_accepted": int(sum((r.get("projected_image_accept_applied") or 0) > 0 for r in source_rows)),
                "images_reject_boosted": int(sum((r.get("image_reject_boost_applied") or 0) > 0 for r in source_rows)),
                "reject_boost_rate": float(np.mean([(r.get("image_reject_boost_applied") or 0) > 0 for r in source_rows])) if source_rows else 0.0,
                "images_projected_boosted": int(sum((r.get("projected_reject_boost_applied") or 0) > 0 for r in source_rows)),
                "projected_boost_rate": float(np.mean([(r.get("projected_reject_boost_applied") or 0) > 0 for r in source_rows])) if source_rows else 0.0,
                "direction_projected_boost_rate": float(np.mean([(r.get("direction_projected_reject_boost_applied") or 0) > 0 for r in source_rows])) if source_rows else 0.0,
                "magnitude_projected_boost_rate": float(np.mean([(r.get("magnitude_projected_reject_boost_applied") or 0) > 0 for r in source_rows])) if source_rows else 0.0,
                "images_with_top_accept_patch": int(sum((r.get("top_accept_patches") or 0) > 0 for r in source_rows)),
                "images_with_top_reject_patch": int(sum((r.get("top_reject_patches") or 0) > 0 for r in source_rows)),
                "images_with_top_unknown_patch": int(sum((r.get("top_unknown_patches") or 0) > 0 for r in source_rows)),
            }

        for name, scores in stage_scores.items():
            summary["operating_points"][name]["best_f1"] = _best_operating_point(labels, scores, objective="f1")
            summary["operating_points"][name]["best_balanced_acc"] = _best_operating_point(
                labels, scores, objective="balanced_acc"
            )
            summary["operating_points"][name]["average_precision"] = _average_precision_binary(labels, scores)
            summary["operating_points"][name]["auroc"] = _auroc_binary(labels, scores)
            for target in self.diagnostic_target_precisions:
                key = f"precision_{target:.4f}".rstrip("0").rstrip(".")
                summary["operating_points"][name][key] = _target_precision_operating_point(labels, scores, target)

        base_best_f1_threshold = float(summary["operating_points"]["base"]["best_f1"]["threshold"])
        summary["same_threshold_effects"]["base_best_f1_patch"] = _same_threshold_effect(
            labels, sources, base, patch, base_best_f1_threshold
        )
        summary["same_threshold_effects"]["base_best_f1_accept"] = _same_threshold_effect(
            labels, sources, base, accept, base_best_f1_threshold
        )
        summary["same_threshold_effects"]["base_best_f1_final"] = _same_threshold_effect(
            labels, sources, base, final, base_best_f1_threshold
        )

        base_best_bal_threshold = float(summary["operating_points"]["base"]["best_balanced_acc"]["threshold"])
        summary["same_threshold_effects"]["base_best_balanced_acc_final"] = _same_threshold_effect(
            labels, sources, base, final, base_best_bal_threshold
        )

        if self.diagnostic_report_threshold is not None:
            summary["same_threshold_effects"]["configured_final"] = _same_threshold_effect(
                labels,
                sources,
                base,
                final,
                float(self.diagnostic_report_threshold),
            )

        # If reject/ contains class/type subfolders, report how each class behaves
        # at the *global* operating thresholds. Subtype labels may influence only
        # reference projection sampling/loss weighting and calibration objective
        # weights; they are never inference inputs.
        reject_mask = sources == "reject"
        type_names = sorted(set(str(v) for v in reject_types[reject_mask] if str(v)))
        for reject_type in type_names:
            mask = reject_mask & (reject_types == reject_type)
            type_rows = [
                row for row in rows
                if row["source"] == "reject" and str(row.get("reject_type") or "") == reject_type
            ]
            type_summary: dict[str, Any] = {
                "count": int(np.sum(mask)),
                "score_stats": {name: _score_stats(scores[mask]) for name, scores in stage_scores.items()},
                "top_mean_accept_advantage": _score_stats(np.asarray([
                    float(r["top_mean_accept_advantage"])
                    for r in type_rows
                    if r.get("top_mean_accept_advantage") is not None
                ], dtype=np.float64)),
                "top_mean_direction_projected_accept_advantage": _score_stats(np.asarray([
                    float(r["top_mean_direction_projected_accept_advantage"])
                    for r in type_rows
                    if r.get("top_mean_direction_projected_accept_advantage") is not None
                ], dtype=np.float64)),
                "top_mean_magnitude_projected_accept_advantage": _score_stats(np.asarray([
                    float(r["top_mean_magnitude_projected_accept_advantage"])
                    for r in type_rows
                    if r.get("top_mean_magnitude_projected_accept_advantage") is not None
                ], dtype=np.float64)),
                "top_mean_projected_accept_advantage": _score_stats(np.asarray([
                    float(r["top_mean_projected_accept_advantage"])
                    for r in type_rows
                    if r.get("top_mean_projected_accept_advantage") is not None
                ], dtype=np.float64)),
                "top_mean_projection_magnitude_feature": _score_stats(np.asarray([
                    float(r["top_mean_projection_magnitude_feature"])
                    for r in type_rows
                    if r.get("top_mean_projection_magnitude_feature") is not None
                ], dtype=np.float64)),
                "image_accept_rate": float(np.mean([(r.get("image_accept_applied") or 0) > 0 for r in type_rows])) if type_rows else 0.0,
                "raw_image_accept_rate": float(np.mean([(r.get("raw_image_accept_applied") or 0) > 0 for r in type_rows])) if type_rows else 0.0,
                "projected_image_accept_rate": float(np.mean([(r.get("projected_image_accept_applied") or 0) > 0 for r in type_rows])) if type_rows else 0.0,
                "reject_veto_rate": float(np.mean([(r.get("image_reject_veto_applied") or 0) > 0 for r in type_rows])) if type_rows else 0.0,
                "reject_boost_rate": float(np.mean([(r.get("image_reject_boost_applied") or 0) > 0 for r in type_rows])) if type_rows else 0.0,
                "mean_reject_boost": float(np.mean([(r.get("image_reject_boost_amount") or 0.0) for r in type_rows])) if type_rows else 0.0,
                "projected_boost_rate": float(np.mean([(r.get("projected_reject_boost_applied") or 0) > 0 for r in type_rows])) if type_rows else 0.0,
                "mean_projected_boost": float(np.mean([(r.get("projected_reject_boost_amount") or 0.0) for r in type_rows])) if type_rows else 0.0,
                "direction_projected_boost_rate": float(np.mean([(r.get("direction_projected_reject_boost_applied") or 0) > 0 for r in type_rows])) if type_rows else 0.0,
                "magnitude_projected_boost_rate": float(np.mean([(r.get("magnitude_projected_reject_boost_applied") or 0) > 0 for r in type_rows])) if type_rows else 0.0,
                "recall_at_global_operating_points": {},
            }
            for stage_name, scores in stage_scores.items():
                stage_ops: dict[str, Any] = {}
                for target in self.diagnostic_target_precisions:
                    key = f"precision_{target:.4f}".rstrip("0").rstrip(".")
                    op = summary["operating_points"][stage_name][key]
                    threshold = op.get("threshold")
                    if threshold is None:
                        stage_ops[key] = {"threshold": None, "TP": 0, "FN": int(np.sum(mask)), "recall": 0.0}
                    else:
                        tp = int(np.sum(scores[mask] >= float(threshold)))
                        total = int(np.sum(mask))
                        stage_ops[key] = {
                            "threshold": float(threshold),
                            "TP": tp,
                            "FN": total - tp,
                            "recall": float(tp / total) if total else 0.0,
                        }
                type_summary["recall_at_global_operating_points"][stage_name] = stage_ops
            summary["reject_type_diagnostics"][reject_type] = type_summary

        return summary

    @staticmethod
    def _log_diagnostic_headline(summary: dict[str, Any]) -> None:
        counts = summary["counts"]
        logger.info(
            "DIAGNOSTICS counts good=%d acceptable=%d reject=%d",
            counts["good"], counts["acceptable"], counts["reject"],
        )
        thresholds = summary["model_thresholds"]
        logger.info(
            "DIAGNOSTICS image ACCEPT: threshold=%.6f damping=%.3f; veto=%s veto_threshold=%.6f",
            thresholds["image_accept_threshold"],
            thresholds["image_accept_damping"],
            thresholds["image_reject_veto_enable"],
            thresholds["image_reject_veto_threshold"],
        )
        logger.info(
            "DIAGNOSTICS projected image ACCEPT: enabled=%s mode=%s threshold=%.6f damping=%.3f",
            thresholds["projected_image_accept_enable"],
            thresholds["residual_projection_mode"],
            thresholds["projected_image_accept_threshold"],
            thresholds["projected_image_accept_damping"],
        )
        logger.info(
            "DIAGNOSTICS image REJECT boost: enabled=%s advantage_threshold=%.6f lambda=%.6f max=%.6f",
            thresholds["image_reject_boost_enable"],
            thresholds["image_reject_boost_advantage_threshold"],
            thresholds["image_reject_boost_lambda"],
            thresholds["image_reject_boost_max"],
        )
        logger.info(
            "DIAGNOSTICS v6c projected boost: enabled=%s dim=%d advantage_threshold=%.6f lambda=%.6f max=%.6f",
            thresholds["projected_reject_boost_enable"], thresholds["residual_projection_dim"],
            thresholds["projected_reject_boost_advantage_threshold"], thresholds["projected_reject_boost_lambda"],
            thresholds["projected_reject_boost_max"],
        )
        for source in ("good", "acceptable", "reject"):
            sup = summary["suppression_by_source"][source]
            logger.info(
                "DIAGNOSTICS %s: patch-reduction=%.6f veto-rate=%.4f image-accept-rate=%.4f "
                "raw-accept-rate=%.4f projected-accept-rate=%.4f accept-reduction=%.6f "
                "reject-boost-rate=%.4f mean-boost=%.6f projected-boost-rate=%.4f projected-mean-boost=%.6f",
                source,
                sup["mean_patch_score_reduction"],
                sup["reject_veto_rate"],
                sup["image_accept_rate"],
                sup["raw_image_accept_rate"],
                sup["projected_image_accept_rate"],
                sup["mean_image_accept_score_reduction"],
                sup["reject_boost_rate"],
                sup["mean_reject_boost_score_increase"],
                sup["projected_boost_rate"],
                sup["mean_projected_boost_score_increase"],
            )
        for model_name in ("base", "patch", "veto", "accept", "v5", "final"):
            ops = summary["operating_points"][model_name]
            best = ops["best_f1"]
            logger.info(
                "DIAGNOSTICS %s: AP=%.4f AUROC=%.4f best-F1 P=%.4f R=%.4f F1=%.4f threshold=%.6f FP=%d TP=%d",
                model_name,
                ops["average_precision"],
                ops["auroc"],
                best["precision"], best["recall"], best["f1"], best["threshold"],
                best["confusion_matrix"]["FP"], best["confusion_matrix"]["TP"],
            )
            for key, op in ops.items():
                if key.startswith("precision_"):
                    logger.info(
                        "DIAGNOSTICS %s %s: met=%s P=%.4f R=%.4f threshold=%s FP=%d TP=%d",
                        model_name,
                        key,
                        op["met"],
                        op["precision"],
                        op["recall"],
                        f'{op["threshold"]:.6f}' if op["threshold"] is not None else "n/a",
                        op["confusion_matrix"]["FP"],
                        op["confusion_matrix"]["TP"],
                    )

        effect = summary["same_threshold_effects"]["base_best_f1_final"]
        logger.info(
            "DIAGNOSTICS same base-best-F1 threshold %.6f: good positive %d->%d; acceptable positive %d->%d; reject TP %d->%d",
            effect["threshold"],
            effect["base_by_source"]["good"]["positive"], effect["final_by_source"]["good"]["positive"],
            effect["base_by_source"]["acceptable"]["positive"], effect["final_by_source"]["acceptable"]["positive"],
            effect["base_by_source"]["reject"]["positive"], effect["final_by_source"]["reject"]["positive"],
        )
        logger.info(
            "DIAGNOSTICS transitions: acceptable FP suppressed=%.4f added=%.4f; reject TP suppressed=%.4f recovered=%.4f; good FP removed=%.4f added=%.4f",
            effect["acceptable_fp_suppression_rate"],
            effect["acceptable_fp_addition_rate"],
            effect["reject_tp_false_suppression_rate"],
            effect["reject_tp_recovery_rate"],
            effect["good_fp_suppression_rate"],
            effect["good_fp_addition_rate"],
        )

        for reject_type, info in summary.get("reject_type_diagnostics", {}).items():
            final_ops = info["recall_at_global_operating_points"]["final"]
            pieces = []
            for key, op in final_ops.items():
                pieces.append(f"{key}:R={op['recall']:.3f}({op['TP']}/{info['count']})")
            logger.info(
                "DIAGNOSTICS reject-type %s count=%d accept-rate=%.4f projected-accept-rate=%.4f "
                "boost-rate=%.4f mean-boost=%.6f projected-rate=%.4f projected-mean=%.6f %s",
                reject_type,
                info["count"],
                info["image_accept_rate"],
                info["projected_image_accept_rate"],
                info["reject_boost_rate"],
                info["mean_reject_boost"],
                info["projected_boost_rate"],
                info["mean_projected_boost"],
                " ".join(pieces),
            )

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs
        predictions = self.model(batch.image)
        predictions = (
            predictions[0]
            if (isinstance(predictions, tuple) and not hasattr(predictions, "_asdict"))
            else predictions
        )
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0, "devices": 1}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS


class _FlatImageDataset(torch.utils.data.Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self, root: Path, transform=None, return_path: bool = False) -> None:
        self.paths = list(_iter_image_paths(root))
        self.transform = transform
        self.return_path = bool(return_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img_t = torch.from_numpy(np.array(img, dtype="float32") / 255.0).permute(2, 0, 1)
        if self.transform is not None:
            img_t = self.transform(img_t)
        if self.return_path:
            return img_t, str(path)
        return img_t


def _iter_image_paths(root: Path):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _FlatImageDataset.EXTENSIONS
    )


def _reject_type_from_path(path: Path, reject_root: Path, depth: int = 1) -> str:
    """Return a diagnostic reject class from nested folders under reject_root.

    Example: reject/rock/large/a.jpg -> ``rock`` when depth=1, ``rock/large``
    when depth=2, and the full parent path when depth<=0.  Files directly in
    reject_root are reported as ``__root__``.
    """
    try:
        rel = path.resolve().relative_to(reject_root.resolve())
    except (OSError, ValueError):
        try:
            rel = path.relative_to(reject_root)
        except ValueError:
            return "__unknown__"
    parts = rel.parent.parts
    if not parts or parts == (".",):
        return "__root__"
    if depth <= 0:
        return "/".join(parts)
    return "/".join(parts[:depth])


def _calibrate_image_accept_rule(
    rows: list[dict[str, Any]],
    *,
    max_reject_false_accept_rate: float,
    safety_margin: float,
    min_threshold: float,
) -> dict[str, Any]:
    """Calibrate the aggregate ACCEPT threshold without touching the test set.

    The decision is ``top_mean_accept_advantage >= threshold`` where positive
    values indicate that the highest-anomaly patches are closer to acceptable
    residuals than reject residuals.  We choose the *lowest* threshold that
    satisfies the configured reject false-accept constraint, because that
    maximizes acceptable suppression under the safety constraint.  A positive
    safety margin is then added to make the deployed threshold more conservative.
    """
    acc = np.asarray([
        float(r["top_mean_accept_advantage"])
        for r in rows
        if r["source"] == "acceptable" and r.get("top_mean_accept_advantage") is not None
    ], dtype=np.float64)
    rej = np.asarray([
        float(r["top_mean_accept_advantage"])
        for r in rows
        if r["source"] == "reject" and r.get("top_mean_accept_advantage") is not None
    ], dtype=np.float64)
    acc = acc[np.isfinite(acc)]
    rej = rej[np.isfinite(rej)]

    if acc.size == 0 or rej.size == 0:
        return {
            "threshold": float("inf"),
            "raw_threshold": float("inf"),
            "safety_margin": float(safety_margin),
            "min_threshold": float(min_threshold),
            "max_reject_false_accept_rate": float(max_reject_false_accept_rate),
            "acceptable_count": int(acc.size),
            "reject_count": int(rej.size),
            "acceptable_accepted": 0,
            "reject_false_accepted": 0,
            "acceptable_accept_rate": 0.0,
            "reject_false_accept_rate": 0.0,
            "status": "insufficient_calibration_samples",
        }

    # Threshold transitions happen at observed values.  Include a point just
    # above the maximum to guarantee a zero-accept fallback if required.
    values = np.unique(np.concatenate([acc, rej, np.asarray([min_threshold], dtype=np.float64)]))
    values = np.sort(values)
    fallback = np.nextafter(float(max(np.max(values), min_threshold)), float("inf"))
    candidates = np.concatenate([values, np.asarray([fallback], dtype=np.float64)])

    chosen: float | None = None
    for threshold in candidates:
        reject_rate = float(np.mean(rej >= threshold))
        if reject_rate <= max_reject_false_accept_rate + 1e-12:
            chosen = float(threshold)
            break
    if chosen is None:
        chosen = fallback

    raw_threshold = max(float(min_threshold), chosen)
    deployed = raw_threshold + max(0.0, float(safety_margin))
    acc_accept = acc >= deployed
    rej_accept = rej >= deployed

    return {
        "threshold": float(deployed),
        "raw_threshold": float(raw_threshold),
        "safety_margin": float(safety_margin),
        "min_threshold": float(min_threshold),
        "max_reject_false_accept_rate": float(max_reject_false_accept_rate),
        "acceptable_count": int(acc.size),
        "reject_count": int(rej.size),
        "acceptable_accepted": int(np.sum(acc_accept)),
        "reject_false_accepted": int(np.sum(rej_accept)),
        "acceptable_accept_rate": float(np.mean(acc_accept)),
        "reject_false_accept_rate": float(np.mean(rej_accept)),
        "acceptable_advantage_stats": _score_stats(acc),
        "reject_advantage_stats": _score_stats(rej),
        "status": "ok",
    }


def _calibrate_projected_image_accept_rule(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    max_combined_reject_false_accept_rate: float,
    safety_margin: float,
    min_threshold: float,
    raw_image_accept_enable: bool,
    raw_image_accept_threshold: float,
    raw_image_reject_veto_enable: bool,
    raw_image_reject_veto_threshold: float,
) -> dict[str, Any]:
    """Calibrate projected ACCEPT while respecting the frozen raw ACCEPT budget.

    Positive projected evidence means acceptable-like.  Direction/magnitude modes
    use their corresponding projected advantage; dual mode uses the minimum so
    both heads must agree.  The deployed threshold is the lowest conservative
    boundary whose *combined* raw-or-projected reject false-accept rate stays
    within the configured calibration budget.
    """
    if mode not in {"direction", "magnitude", "dual"}:
        raise ValueError(f"Unsupported residual_projection_mode for projected ACCEPT: {mode!r}")

    def projected_value(row: dict[str, Any]) -> float:
        direction = row.get("top_mean_direction_projected_accept_advantage")
        magnitude = row.get("top_mean_magnitude_projected_accept_advantage")
        d = float(direction) if direction is not None else float("nan")
        m = float(magnitude) if magnitude is not None else float("nan")
        if mode == "direction":
            return d
        if mode == "magnitude":
            return m
        if not (np.isfinite(d) and np.isfinite(m)):
            return float("nan")
        return float(min(d, m))

    def raw_accepts(row: dict[str, Any]) -> bool:
        if not raw_image_accept_enable or not np.isfinite(raw_image_accept_threshold):
            return False
        value = row.get("top_mean_accept_advantage")
        if value is None:
            return False
        advantage = float(value)
        if not np.isfinite(advantage) or not (advantage > raw_image_accept_threshold):
            return False
        if raw_image_reject_veto_enable and advantage <= raw_image_reject_veto_threshold:
            return False
        return True

    acc_rows = [r for r in rows if r.get("source") == "acceptable"]
    good_rows = [r for r in rows if r.get("source") == "good"]
    rej_rows = [r for r in rows if r.get("source") == "reject"]

    acc = np.asarray([projected_value(r) for r in acc_rows], dtype=np.float64)
    good = np.asarray([projected_value(r) for r in good_rows], dtype=np.float64)
    rej = np.asarray([projected_value(r) for r in rej_rows], dtype=np.float64)
    acc_finite = np.isfinite(acc)
    good_finite = np.isfinite(good)
    rej_finite = np.isfinite(rej)

    if not np.any(acc_finite) or not np.any(rej_finite):
        return {
            "threshold": float("inf"),
            "raw_threshold": float("inf"),
            "projection_mode": mode,
            "dual_fusion": "min(direction,magnitude)" if mode == "dual" else mode,
            "safety_margin": float(safety_margin),
            "min_threshold": float(min_threshold),
            "max_combined_reject_false_accept_rate": float(max_combined_reject_false_accept_rate),
            "acceptable_count": int(np.sum(acc_finite)),
            "good_count": int(np.sum(good_finite)),
            "reject_count": int(np.sum(rej_finite)),
            "acceptable_projected_accepted": 0,
            "acceptable_projected_accept_rate": 0.0,
            "good_projected_accepted": 0,
            "good_projected_accept_rate": 0.0,
            "reject_projected_false_accepted": 0,
            "reject_projected_false_accept_rate": 0.0,
            "raw_reject_false_accepted": 0,
            "raw_reject_false_accept_rate": 0.0,
            "combined_reject_false_accepted": 0,
            "combined_reject_false_accept_rate": 0.0,
            "status": "insufficient_calibration_samples",
        }

    # Keep all reject rows in the safety denominator. Non-finite projected
    # evidence simply means the projected gate cannot fire for that image; the
    # already-frozen raw gate can still fire and must remain part of the union.
    raw_rej_accept = np.asarray([raw_accepts(r) for r in rej_rows], dtype=bool)
    raw_rej_count = int(np.sum(raw_rej_accept))
    raw_rej_rate = float(np.mean(raw_rej_accept)) if raw_rej_accept.size else 0.0

    budget = float(max_combined_reject_false_accept_rate)
    if raw_rej_rate > budget + 1e-12:
        # The frozen raw gate already exceeds the requested combined budget.  Do
        # not allow projected ACCEPT to worsen it; effectively disable this gate.
        chosen = float("inf")
        status = "raw_baseline_exceeds_combined_budget"
    else:
        rej_values = rej[rej_finite]
        values = np.unique(np.concatenate([rej_values, np.asarray([min_threshold], dtype=np.float64)]))
        values = np.sort(values)
        fallback = np.nextafter(float(max(np.max(values), min_threshold)), float("inf"))
        candidates = np.concatenate([values, np.asarray([fallback], dtype=np.float64)])
        chosen = fallback
        status = "ok"
        for candidate in candidates:
            projected_rej_accept = rej_finite & (rej > float(candidate))
            combined = raw_rej_accept | projected_rej_accept
            combined_rate = float(np.mean(combined)) if combined.size else 0.0
            if combined_rate <= budget + 1e-12:
                chosen = float(candidate)
                break

    raw_threshold = max(float(min_threshold), float(chosen))
    deployed = (
        float("inf")
        if not np.isfinite(raw_threshold)
        else raw_threshold + max(0.0, float(safety_margin))
    )

    def projected_counts(values: np.ndarray, finite: np.ndarray) -> tuple[int, int, float]:
        valid = values[finite]
        accepted = valid > deployed
        count = int(valid.size)
        accepted_count = int(np.sum(accepted))
        return accepted_count, count, float(np.mean(accepted)) if count else 0.0

    acc_n_accept, acc_count, acc_rate = projected_counts(acc, acc_finite)
    good_n_accept, good_count, good_rate = projected_counts(good, good_finite)
    rej_finite_accept, rej_finite_count, _rej_finite_rate = projected_counts(rej, rej_finite)
    deployed_projected_rej = rej_finite & (rej > deployed)
    combined_rej = raw_rej_accept | deployed_projected_rej
    rej_count = int(len(rej_rows))
    rej_n_accept = int(np.sum(deployed_projected_rej))
    rej_rate = float(rej_n_accept / rej_count) if rej_count else 0.0

    return {
        "threshold": float(deployed),
        "raw_threshold": float(raw_threshold),
        "projection_mode": mode,
        "dual_fusion": "min(direction,magnitude)" if mode == "dual" else mode,
        "safety_margin": float(safety_margin),
        "min_threshold": float(min_threshold),
        "max_combined_reject_false_accept_rate": budget,
        "raw_image_accept_enable": bool(raw_image_accept_enable),
        "raw_image_accept_threshold": float(raw_image_accept_threshold),
        "raw_image_reject_veto_enable": bool(raw_image_reject_veto_enable),
        "raw_image_reject_veto_threshold": float(raw_image_reject_veto_threshold),
        "acceptable_count": acc_count,
        "acceptable_projected_accepted": acc_n_accept,
        "acceptable_projected_accept_rate": acc_rate,
        "good_count": good_count,
        "good_projected_accepted": good_n_accept,
        "good_projected_accept_rate": good_rate,
        "reject_count": rej_count,
        "reject_projected_evidence_count": int(rej_finite_count),
        "reject_projected_false_accepted": rej_n_accept,
        "reject_projected_false_accept_rate": rej_rate,
        "raw_reject_false_accepted": raw_rej_count,
        "raw_reject_false_accept_rate": raw_rej_rate,
        "combined_reject_false_accepted": int(np.sum(combined_rej)),
        "combined_reject_false_accept_rate": (
            float(np.mean(combined_rej)) if combined_rej.size else 0.0
        ),
        "acceptable_projected_evidence_stats": _score_stats(acc[acc_finite]),
        "good_projected_evidence_stats": _score_stats(good[good_finite]),
        "reject_projected_evidence_stats": _score_stats(rej[rej_finite]),
        "status": status,
    }


def _apply_reject_boost_numpy(
    scores: np.ndarray,
    advantages: np.ndarray,
    *,
    advantage_threshold: float,
    boost_lambda: float,
    max_boost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the v5 continuous REJECT boost in numpy for calibration."""
    finite = np.isfinite(advantages)
    evidence = np.zeros_like(scores, dtype=np.float64)
    evidence[finite] = np.maximum(float(advantage_threshold) - advantages[finite], 0.0)
    boost = evidence * float(boost_lambda)
    if max_boost > 0.0:
        boost = np.minimum(boost, float(max_boost))
    return scores + boost, evidence, boost


def _calibrate_image_reject_boost_rule(
    rows: list[dict[str, Any]],
    *,
    target_precision: float,
    advantage_candidates: tuple[float, ...] | list[float],
    lambda_candidates: tuple[float, ...] | list[float],
    max_boost: float,
) -> dict[str, Any]:
    """Choose REJECT-boost parameters on calibration data only.

    Candidate scores are:
        accept_pred_score + lambda * max(A0 - accept_advantage, 0)

    Parameters are selected to maximize recall at the configured precision
    constraint.  ``lambda=0`` is always evaluated, so calibration can choose
    the v4 score unchanged when boosting does not help.
    """
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    sources = np.asarray([str(r["source"]) for r in rows], dtype=object)
    scores = np.asarray([
        float(r.get("accept_pred_score", r.get("final_pred_score", 0.0))) for r in rows
    ], dtype=np.float64)
    advantages = np.asarray([
        float(r["top_mean_accept_advantage"])
        if r.get("top_mean_accept_advantage") is not None
        else float("nan")
        for r in rows
    ], dtype=np.float64)

    baseline = _target_precision_operating_point(labels, scores, float(target_precision))
    candidate_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_key: tuple[Any, ...] | None = None

    for a0 in sorted(set(float(v) for v in advantage_candidates)):
        for lam in sorted(set(float(v) for v in lambda_candidates)):
            boosted, evidence, boost = _apply_reject_boost_numpy(
                scores,
                advantages,
                advantage_threshold=a0,
                boost_lambda=lam,
                max_boost=max_boost,
            )
            op = _target_precision_operating_point(labels, boosted, float(target_precision))
            negative_mask = labels == 0
            reject_mask = labels == 1
            neg_boost_rate = float(np.mean(boost[negative_mask] > 1e-12)) if np.any(negative_mask) else 0.0
            rej_boost_rate = float(np.mean(boost[reject_mask] > 1e-12)) if np.any(reject_mask) else 0.0
            mean_neg_boost = float(np.mean(boost[negative_mask])) if np.any(negative_mask) else 0.0
            mean_rej_boost = float(np.mean(boost[reject_mask])) if np.any(reject_mask) else 0.0
            entry = {
                "advantage_threshold": float(a0),
                "lambda": float(lam),
                "operating_point": op,
                "negative_boost_rate": neg_boost_rate,
                "reject_boost_rate": rej_boost_rate,
                "mean_negative_boost": mean_neg_boost,
                "mean_reject_boost": mean_rej_boost,
                "max_observed_boost": float(np.max(boost)) if boost.size else 0.0,
            }
            candidate_results.append(entry)

            # Prefer meeting the precision target, then recall.  Ties are made
            # progressively more conservative: fewer FP, less negative-class
            # boosting, smaller lambda, and a more-negative activation boundary.
            if op["met"]:
                key = (
                    1,
                    float(op["recall"]),
                    float(op["precision"]),
                    -int(op["confusion_matrix"]["FP"]),
                    -neg_boost_rate,
                    -mean_neg_boost,
                    -float(lam),
                    -float(a0),
                )
            else:
                key = (
                    0,
                    float(op["precision"]),
                    float(op["recall"]),
                    -int(op["confusion_matrix"]["FP"]),
                    -neg_boost_rate,
                    -mean_neg_boost,
                    -float(lam),
                    -float(a0),
                )
            if selected_key is None or key > selected_key:
                selected_key = key
                selected = entry

    if selected is None:
        selected = {
            "advantage_threshold": 0.0,
            "lambda": 0.0,
            "operating_point": baseline,
            "negative_boost_rate": 0.0,
            "reject_boost_rate": 0.0,
            "mean_negative_boost": 0.0,
            "mean_reject_boost": 0.0,
            "max_observed_boost": 0.0,
        }

    selected_boosted, selected_evidence, selected_boost = _apply_reject_boost_numpy(
        scores,
        advantages,
        advantage_threshold=float(selected["advantage_threshold"]),
        boost_lambda=float(selected["lambda"]),
        max_boost=max_boost,
    )

    by_source: dict[str, Any] = {}
    for source in ("good", "acceptable", "reject"):
        mask = sources == source
        by_source[source] = {
            "count": int(np.sum(mask)),
            "boosted": int(np.sum(selected_boost[mask] > 1e-12)),
            "boost_rate": float(np.mean(selected_boost[mask] > 1e-12)) if np.any(mask) else 0.0,
            "mean_boost": float(np.mean(selected_boost[mask])) if np.any(mask) else 0.0,
            "mean_evidence": float(np.mean(selected_evidence[mask])) if np.any(mask) else 0.0,
            "score_stats_before": _score_stats(scores[mask]),
            "score_stats_after": _score_stats(selected_boosted[mask]),
        }

    # Keep JSON useful without making it enormous: rank the best 20 candidate
    # settings by the same target-precision objective.
    def _candidate_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        op = entry["operating_point"]
        return (
            bool(op["met"]),
            float(op["recall"]) if op["met"] else float(op["precision"]),
            float(op["precision"]) if op["met"] else float(op["recall"]),
            -entry["negative_boost_rate"],
            -entry["mean_negative_boost"],
        )

    top_candidates = sorted(candidate_results, key=_candidate_sort_key, reverse=True)[:20]
    return {
        "status": "ok",
        "target_precision": float(target_precision),
        "advantage_threshold": float(selected["advantage_threshold"]),
        "lambda": float(selected["lambda"]),
        "max_boost": float(max_boost),
        "baseline_operating_point": baseline,
        "selected_operating_point": selected["operating_point"],
        "selected_boost_stats_by_source": by_source,
        "advantage_candidates": [float(v) for v in advantage_candidates],
        "lambda_candidates": [float(v) for v in lambda_candidates],
        "top_candidates": top_candidates,
    }


def _calibrate_projected_reject_boost_rule(
    rows: list[dict[str, Any]], *, target_precision: float,
    advantage_candidates: tuple[float, ...] | list[float],
    lambda_candidates: tuple[float, ...] | list[float], max_boost: float,
    reject_type_weights: dict[str, float] | None,
    require_non_decreasing_overall_recall: bool,
    advantage_key: str = "top_mean_projected_accept_advantage",
) -> dict[str, Any]:
    """Calibrate projected-space REJECT boost with generic subtype weighting.

    ``reject_type_weights`` adds extra optimization importance to any reject
    subtype. For example ``{"type_a": 3.0}`` emphasizes that type.
    Unlisted types contribute only through global recall. The calibration never
    uses subtype labels at inference; they are only an objective on the held-out
    calibration partition.
    """
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    scores = np.asarray([float(r.get("v5_pred_score", r["final_pred_score"])) for r in rows], dtype=np.float64)
    advantages = np.asarray([
        float(r[advantage_key]) if r.get(advantage_key) is not None else float("nan")
        for r in rows
    ], dtype=np.float64)
    reject_types = np.asarray([str(r.get("reject_type") or "") for r in rows], dtype=object)
    weights = {str(k): max(0.0, float(v)) for k, v in (reject_type_weights or {}).items()}

    baseline = _target_precision_operating_point(labels, scores, float(target_precision))
    baseline_threshold = baseline.get("threshold")

    all_reject_types = sorted(set(str(v) for v in reject_types[labels == 1] if str(v)))
    missing_weight_types = sorted(set(weights) - set(all_reject_types))
    if missing_weight_types:
        logger.warning(
            "Projected boost calibration has configured reject-type weights with no calibration samples: %s",
            missing_weight_types,
        )

    def recalls_by_type(candidate_scores: np.ndarray, threshold: float | None) -> dict[str, float]:
        result: dict[str, float] = {}
        for reject_type in all_reject_types:
            mask = (labels == 1) & (reject_types == reject_type)
            total = int(np.sum(mask))
            if total == 0 or threshold is None:
                result[reject_type] = 0.0
            else:
                result[reject_type] = float(np.sum(candidate_scores[mask] >= float(threshold)) / total)
        return result

    baseline_type_recalls = recalls_by_type(scores, baseline_threshold)

    def weighted_objective(op: dict[str, Any], type_recalls: dict[str, float]) -> float:
        return float(op["recall"]) + sum(
            float(weight) * float(type_recalls.get(reject_type, 0.0))
            for reject_type, weight in weights.items()
        )

    baseline_weighted = weighted_objective(baseline, baseline_type_recalls)
    selected = None
    selected_key = None
    results: list[dict[str, Any]] = []

    for a0 in sorted(set(float(v) for v in advantage_candidates)):
        for lam in sorted(set(float(v) for v in lambda_candidates)):
            boosted, _evidence, boost = _apply_reject_boost_numpy(
                scores, advantages, advantage_threshold=a0, boost_lambda=lam, max_boost=max_boost
            )
            op = _target_precision_operating_point(labels, boosted, float(target_precision))
            threshold = op.get("threshold")
            type_recalls = recalls_by_type(boosted, threshold)
            overall_ok = (
                (not require_non_decreasing_overall_recall)
                or (float(op["recall"]) + 1e-12 >= float(baseline["recall"]))
            )
            weighted = weighted_objective(op, type_recalls)
            neg = labels == 0
            neg_rate = float(np.mean(boost[neg] > 1e-12)) if np.any(neg) else 0.0
            boost_rate_by_type = {}
            for reject_type in all_reject_types:
                mask = (labels == 1) & (reject_types == reject_type)
                boost_rate_by_type[reject_type] = (
                    float(np.mean(boost[mask] > 1e-12)) if np.any(mask) else 0.0
                )
            entry = {
                "advantage_threshold": a0,
                "lambda": lam,
                "operating_point": op,
                "recall_by_reject_type": type_recalls,
                "weighted_objective": weighted,
                "overall_recall_constraint_met": bool(overall_ok),
                "negative_boost_rate": neg_rate,
                "mean_negative_boost": float(np.mean(boost[neg])) if np.any(neg) else 0.0,
                "boost_rate_by_reject_type": boost_rate_by_type,
            }
            results.append(entry)

            weighted_type_vector = tuple(
                float(type_recalls.get(name, 0.0)) for name in sorted(weights)
            )
            eligible = bool(op["met"]) and bool(overall_ok)
            if eligible:
                key = (
                    1, weighted, weighted_type_vector, float(op["recall"]),
                    float(op["precision"]), -int(op["confusion_matrix"]["FP"]),
                    -neg_rate, -lam,
                )
            else:
                key = (
                    0, float(op["precision"]), weighted, weighted_type_vector,
                    float(op["recall"]), -int(op["confusion_matrix"]["FP"]),
                    -neg_rate, -lam,
                )
            if selected_key is None or key > selected_key:
                selected_key, selected = key, entry

    if selected is None:
        selected = {
            "advantage_threshold": 0.0,
            "lambda": 0.0,
            "operating_point": baseline,
            "recall_by_reject_type": baseline_type_recalls,
            "weighted_objective": baseline_weighted,
            "overall_recall_constraint_met": True,
            "negative_boost_rate": 0.0,
            "mean_negative_boost": 0.0,
            "boost_rate_by_reject_type": {name: 0.0 for name in all_reject_types},
        }

    def rank_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        op = entry["operating_point"]
        type_vector = tuple(
            float(entry["recall_by_reject_type"].get(name, 0.0)) for name in sorted(weights)
        )
        return (
            bool(op["met"] and entry["overall_recall_constraint_met"]),
            float(entry["weighted_objective"]),
            type_vector,
            float(op["recall"]), float(op["precision"]),
            -float(entry["negative_boost_rate"]),
        )

    return {
        "status": "ok",
        "advantage_key": str(advantage_key),
        "target_precision": float(target_precision),
        "reject_type_weights": weights,
        "require_non_decreasing_overall_recall": bool(require_non_decreasing_overall_recall),
        "advantage_threshold": float(selected["advantage_threshold"]),
        "lambda": float(selected["lambda"]),
        "max_boost": float(max_boost),
        "baseline_operating_point": baseline,
        "baseline_recall_by_reject_type": baseline_type_recalls,
        "baseline_weighted_objective": float(baseline_weighted),
        "selected_operating_point": selected["operating_point"],
        "selected_recall_by_reject_type": selected["recall_by_reject_type"],
        "selected_weighted_objective": float(selected["weighted_objective"]),
        "advantage_candidates": [float(v) for v in advantage_candidates],
        "lambda_candidates": [float(v) for v in lambda_candidates],
        "top_candidates": sorted(results, key=rank_key, reverse=True)[:30],
    }


def _score_stats(scores: np.ndarray) -> dict[str, float | int | None]:
    if scores.size == 0:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": int(scores.size),
        "min": float(np.min(scores)),
        "p50": float(np.quantile(scores, 0.50)),
        "p90": float(np.quantile(scores, 0.90)),
        "p95": float(np.quantile(scores, 0.95)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
    }


def _average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision for binary labels without a sklearn dependency."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos_total = int(np.sum(labels == 1))
    if pos_total == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    n = len(y)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        group = y[i:j]
        tp += int(np.sum(group == 1))
        fp += int(np.sum(group == 0))
        recall = tp / pos_total
        precision = tp / max(1, tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def _auroc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware Mann-Whitney AUROC for binary labels."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # one-based average rank
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = scores >= threshold
    pos = labels == 1
    neg = ~pos
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum((~pred) & neg))
    fn = int(np.sum((~pred) & pos))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    accuracy = (tp + tn) / max(1, labels.size)
    balanced = 0.5 * (recall + tnr)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_acc": float(balanced),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    # Every distinct score is a meaningful >= threshold transition. Add a value
    # just above max to represent predicting no positives for completeness.
    unique = np.unique(scores)
    unique = np.sort(unique)[::-1]
    if unique.size == 0:
        return np.asarray([], dtype=np.float64)
    return unique


def _best_operating_point(labels: np.ndarray, scores: np.ndarray, objective: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_value = -1.0
    for threshold in _candidate_thresholds(scores):
        metrics = _metrics_at_threshold(labels, scores, float(threshold))
        value = float(metrics[objective])
        if value > best_value + 1e-12 or (
            abs(value - best_value) <= 1e-12
            and best is not None
            and metrics["recall"] > best["recall"]
        ):
            best = metrics
            best_value = value
    if best is None:
        return _metrics_at_threshold(labels, scores, float("inf"))
    return best


def _target_precision_operating_point(
    labels: np.ndarray,
    scores: np.ndarray,
    target_precision: float,
) -> dict[str, Any]:
    qualifying: list[dict[str, Any]] = []
    all_nonempty: list[dict[str, Any]] = []
    for threshold in _candidate_thresholds(scores):
        metrics = _metrics_at_threshold(labels, scores, float(threshold))
        if metrics["confusion_matrix"]["TP"] > 0:
            all_nonempty.append(metrics)
            if metrics["precision"] + 1e-12 >= target_precision:
                qualifying.append(metrics)

    if qualifying:
        best = max(qualifying, key=lambda m: (m["recall"], m["precision"], -m["threshold"]))
        return {"met": True, "target_precision": float(target_precision), **best}

    if all_nonempty:
        best = max(all_nonempty, key=lambda m: (m["precision"], m["recall"]))
        return {"met": False, "target_precision": float(target_precision), **best}

    empty = _metrics_at_threshold(labels, scores, float("inf"))
    empty["threshold"] = None
    return {"met": False, "target_precision": float(target_precision), **empty}


def _same_threshold_effect(
    labels: np.ndarray,
    sources: np.ndarray,
    base: np.ndarray,
    final: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    base_pred = base >= threshold
    final_pred = final >= threshold

    def source_counts(pred: np.ndarray) -> dict[str, dict[str, int | float]]:
        out: dict[str, dict[str, int | float]] = {}
        for source in ("good", "acceptable", "reject"):
            mask = sources == source
            total = int(np.sum(mask))
            positive = int(np.sum(pred & mask))
            out[source] = {
                "total": total,
                "positive": positive,
                "negative": total - positive,
                "positive_rate": float(positive / total) if total else 0.0,
            }
        return out

    acceptable_mask = sources == "acceptable"
    reject_mask = sources == "reject"
    good_mask = sources == "good"

    accept_before = int(np.sum(base_pred & acceptable_mask))
    accept_removed = int(np.sum(base_pred & (~final_pred) & acceptable_mask))
    accept_added = int(np.sum((~base_pred) & final_pred & acceptable_mask))
    accept_base_negative = int(np.sum((~base_pred) & acceptable_mask))

    reject_before = int(np.sum(base_pred & reject_mask))
    reject_removed = int(np.sum(base_pred & (~final_pred) & reject_mask))
    reject_recovered = int(np.sum((~base_pred) & final_pred & reject_mask))
    reject_base_fn = int(np.sum((~base_pred) & reject_mask))

    good_before = int(np.sum(base_pred & good_mask))
    good_removed = int(np.sum(base_pred & (~final_pred) & good_mask))
    good_added = int(np.sum((~base_pred) & final_pred & good_mask))
    good_base_negative = int(np.sum((~base_pred) & good_mask))

    return {
        "threshold": float(threshold),
        "base_metrics": _metrics_at_threshold(labels, base, threshold),
        "final_metrics_same_threshold": _metrics_at_threshold(labels, final, threshold),
        "base_by_source": source_counts(base_pred),
        "final_by_source": source_counts(final_pred),
        "acceptable_fp_suppressed": accept_removed,
        "acceptable_fp_suppression_rate": float(accept_removed / accept_before) if accept_before else 0.0,
        "acceptable_fp_added": accept_added,
        "acceptable_fp_addition_rate": float(accept_added / accept_base_negative) if accept_base_negative else 0.0,
        "reject_tp_falsely_suppressed": reject_removed,
        "reject_tp_false_suppression_rate": float(reject_removed / reject_before) if reject_before else 0.0,
        "reject_tp_recovered": reject_recovered,
        "reject_tp_recovery_rate": float(reject_recovered / reject_base_fn) if reject_base_fn else 0.0,
        "good_fp_removed": good_removed,
        "good_fp_suppression_rate": float(good_removed / good_before) if good_before else 0.0,
        "good_fp_added": good_added,
        "good_fp_addition_rate": float(good_added / good_base_negative) if good_base_negative else 0.0,
    }
