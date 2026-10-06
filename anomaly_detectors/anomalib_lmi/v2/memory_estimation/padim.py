from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .base import BaseAnomalibMemoryEstimator
from .common import FeatureProfile, MemoryBudget, MemoryEstimate, bytes_to_mib


@dataclass
class PadimMemoryEstimate(MemoryEstimate):
    model_type: str = field(default="padim", init=False)


class PadimMemoryEstimator(BaseAnomalibMemoryEstimator):
    model_type = "padim"

    def __init__(
        self,
        model,
        tile_config,
        *,
        precision: str | torch.dtype = "float32",
        profiling_device: str = "cuda",
        profiling_dtype: torch.dtype | None = None,
        stats_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            model=model,
            tile_config=tile_config,
            precision=precision,
            profiling_device=profiling_device,
            profiling_dtype=profiling_dtype,
        )

        # PaDiM covariance / inverse-covariance stats run through torch.linalg.inv,
        # which uses float32 even when feature extraction runs fp16.
        self.stats_dtype = stats_dtype
        self.stats_dtype_bytes = self.dtype_nbytes(stats_dtype)

    def infer_n_features(self, default: int) -> int:
        return self._infer_hparam("n_features", default, cast=int)

    def resolve_train_embedding_dtype(self, profile: FeatureProfile) -> torch.dtype:
        """PaDiM training embeddings follow the extracted feature dtype."""
        if getattr(profile, "feature_dtypes", None):
            first_feature_dtype = next(iter(profile.feature_dtypes.values()))
            return self.dtype_from_string(
                first_feature_dtype,
                fallback=self.activation_dtype,
            )

        return self.activation_dtype

    def per_image_embedding_mib(
        self,
        profile: FeatureProfile,
        *,
        padim_workspace_factor: float = 0.25,
    ) -> dict[str, Any]:
        """Per-image cost of the accumulated training memory bank.

        PaDiM appends every batch embedding to ``memory_bank`` and ``torch.vstack``-es
        them in ``fit()``, so the bank scales with the number of training images.
        ``padim_workspace_factor`` covers the transient ``vstack`` copy and the
        gaussian-fit overlap on top of the raw per-image embedding.
        """
        raw_D = profile.embedding_dim
        d = self.infer_n_features(default=raw_D)

        train_embedding_dtype = self.resolve_train_embedding_dtype(profile)
        train_embedding_dtype_bytes = self.dtype_nbytes(train_embedding_dtype)

        train_embedding_bytes = profile.patches_per_image * d * train_embedding_dtype_bytes

        workspace_bytes = train_embedding_bytes * padim_workspace_factor
        peak_bytes = train_embedding_bytes + workspace_bytes

        return {
            "train_embedding_dtype": str(train_embedding_dtype),
            "train_embedding_dtype_bytes": train_embedding_dtype_bytes,
            "train_embedding_per_image_mib": bytes_to_mib(train_embedding_bytes),
            "workspace_per_image_mib": bytes_to_mib(workspace_bytes),
            "peak_per_image_mib": bytes_to_mib(peak_bytes),
            "padim_workspace_factor": padim_workspace_factor,
        }

    def stats_mib(self, profile: FeatureProfile) -> dict[str, Any]:
        """Fixed-size gaussian stats, independent of the training image count.

        During ``MultiVariateGaussian.fit`` PaDiM holds, simultaneously:
          - mean:           (d, P)        -> d * P
          - covariance:     (d, d, P)     -> d * d * P  (transient workspace)
          - inv_covariance: (P, d, d)     -> P * d * d  (persistent buffer)
          - identity:       (d, d)        -> d * d
        so the peak fixed cost is ~2 * P * d * d, not a single term.
        """
        raw_D = profile.embedding_dim
        d = self.infer_n_features(default=raw_D)
        P = profile.patches_per_image
        b = self.stats_dtype_bytes

        mean_bytes = P * d * b
        covariance_bytes = P * d * d * b  # transient (d, d, P) workspace
        inv_covariance_bytes = P * d * d * b  # persistent (P, d, d) buffer
        identity_bytes = d * d * b

        persistent_bytes = mean_bytes + inv_covariance_bytes
        peak_bytes = mean_bytes + covariance_bytes + inv_covariance_bytes + identity_bytes

        return {
            "stats_dtype": str(self.stats_dtype),
            "stats_dtype_bytes": b,
            "padim_mean_mib": bytes_to_mib(mean_bytes),
            "padim_covariance_mib": bytes_to_mib(covariance_bytes),
            "padim_inv_covariance_mib": bytes_to_mib(inv_covariance_bytes),
            "padim_identity_mib": bytes_to_mib(identity_bytes),
            "padim_stats_persistent_mib": bytes_to_mib(persistent_bytes),
            "padim_stats_peak_mib": bytes_to_mib(peak_bytes),
            # Retained for backward compatibility: total stats peak.
            "padim_stats_mib": bytes_to_mib(peak_bytes),
        }

    def calibrate_workspace_factor(
        self,
        *,
        observed_peak_mib: float,
        n_train_images: int,
        profile: FeatureProfile | None = None,
        fixed_overhead_mib: float = 0.0,
    ) -> float:
        """Recover ``padim_workspace_factor`` from an observed training peak."""
        if profile is None:
            profile = self.profile_features()

        per_raw = self.per_image_embedding_mib(profile, padim_workspace_factor=0.0)
        raw_per_image_mib = per_raw["train_embedding_per_image_mib"]

        activation_mib = self.activation_peak_mib(profile)
        stats_peak_mib = self.stats_mib(profile)["padim_stats_peak_mib"]

        bank_mib = observed_peak_mib - activation_mib - stats_peak_mib - fixed_overhead_mib

        if n_train_images <= 0 or raw_per_image_mib <= 0:
            return 0.0

        observed_total_factor = bank_mib / (n_train_images * raw_per_image_mib)

        # Peak model: raw * (1 + workspace_factor)
        return max(0.0, observed_total_factor - 1.0)

    def estimate_from_profile(
        self,
        *,
        profile: FeatureProfile,
        memory_budget: MemoryBudget,
        padim_workspace_factor: float = 0.25,
        n_train_images: int | None = None,
        **kwargs,
    ) -> PadimMemoryEstimate:
        raw_D = profile.embedding_dim
        d = self.infer_n_features(default=raw_D)

        per = self.per_image_embedding_mib(
            profile,
            padim_workspace_factor=padim_workspace_factor,
        )

        stats = self.stats_mib(profile)
        stats_peak_mib = stats["padim_stats_peak_mib"]

        activation_mib = self.activation_peak_mib(profile)

        gross_available_mib = self.gross_available_mib(
            memory_budget,
            activation_mib,
            extra_fixed_mib=stats_peak_mib,
        )
        effective_budget_mib = self.effective_budget_mib(
            memory_budget,
            activation_mib,
            extra_fixed_mib=stats_peak_mib,
        )

        per_image_peak_mib = per["peak_per_image_mib"]

        if effective_budget_mib <= 0 or per_image_peak_mib <= 0:
            max_images = 0
        else:
            max_images = int(effective_budget_mib // per_image_peak_mib)

        fits_requested = None
        requested: dict[str, Any] = {}

        if n_train_images is not None:
            fits_requested = n_train_images <= max_images

            requested = {
                "requested_train_images": n_train_images,
                "requested_train_embeddings_mib": (n_train_images * per["train_embedding_per_image_mib"]),
                "requested_workspace_mib": (n_train_images * per["workspace_per_image_mib"]),
                "requested_variable_peak_mib": (n_train_images * per["peak_per_image_mib"]),
                "requested_total_peak_mib": (
                    memory_budget.fixed_overhead_mib + activation_mib + stats_peak_mib + n_train_images * per["peak_per_image_mib"]
                ),
            }

        budget = self.common_budget_dict(
            memory_budget,
            activation_mib=activation_mib,
            effective_budget_mib=effective_budget_mib,
            extra={
                "padim_stats_peak_mib": stats_peak_mib,
                "padim_stats_persistent_mib": stats["padim_stats_persistent_mib"],
                "gross_available_mib_after_activation_and_stats": gross_available_mib,
            },
        )

        return PadimMemoryEstimate(
            max_train_images=max_images,
            requested_train_images=n_train_images,
            fits_requested_train_images=fits_requested,
            budget=budget,
            per_image=per,
            requested=requested,
            model={
                "raw_embedding_dim": raw_D,
                "padim_dim_used": d,
                "patches_per_image": profile.patches_per_image,
                "padim_workspace_factor": padim_workspace_factor,
                **self.common_model_dtype_dict(profile),
                **stats,
            },
            tiling=self.common_tiling_dict(profile),
            feature_profile=asdict(profile),
        )
