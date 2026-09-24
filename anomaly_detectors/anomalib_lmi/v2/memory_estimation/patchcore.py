from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .base import BaseAnomalibMemoryEstimator
from .common import FeatureProfile, MemoryBudget, MemoryEstimate, bytes_to_mib

# anomalib's PatchcoreModel.nearest_neighbors query chunk size (DEFAULT_CHUNK_SIZE).
# euclidean_dist is fully in-place since anomalib 2.3, so the transient factor is 1x.
ANOMALIB_QUERY_CHUNK_SIZE = 1024
EUCLIDEAN_DIST_TRANSIENT_FACTOR = 1.0


def euclidean_dist_transient_factor(inference_chunk_size: int | None) -> float:
    return EUCLIDEAN_DIST_TRANSIENT_FACTOR


def detect_inference_chunk_size() -> int | None:
    return ANOMALIB_QUERY_CHUNK_SIZE


@dataclass
class PatchCoreMemoryEstimate(MemoryEstimate):
    model_type: str = field(default="patchcore", init=False)


class PatchCoreMemoryEstimator(BaseAnomalibMemoryEstimator):
    model_type = "patchcore"

    def infer_coreset_sampling_ratio(self, default: float = 0.1) -> float:
        return self._infer_hparam("coreset_sampling_ratio", default, cast=float)

    def get_existing_memory_bank_dtype(self) -> torch.dtype | None:
        inner = self.unwrap_inner_model()

        bank = getattr(inner, "memory_bank", None)
        if isinstance(bank, torch.Tensor) and torch.is_floating_point(bank):
            return bank.dtype

        if isinstance(inner, torch.nn.Module):
            for _, module in inner.named_modules():
                bank = getattr(module, "memory_bank", None)
                if isinstance(bank, torch.Tensor) and torch.is_floating_point(bank):
                    return bank.dtype

        return None

    def resolve_bank_dtype(self, profile: FeatureProfile) -> torch.dtype:
        existing_dtype = self.get_existing_memory_bank_dtype()
        if existing_dtype is not None:
            return existing_dtype

        if getattr(profile, "feature_dtypes", None):
            first_feature_dtype = next(iter(profile.feature_dtypes.values()))
            return self.dtype_from_string(
                first_feature_dtype,
                fallback=self.activation_dtype,
            )

        return self.activation_dtype

    def per_image_bank_mib(
        self,
        profile: FeatureProfile,
        *,
        coreset_sampling_ratio: float | None = None,
        dataset_workspace_factor: float = 0.0,
        include_coreset_in_peak: bool = False,
    ) -> dict[str, Any]:
        if coreset_sampling_ratio is None:
            coreset_sampling_ratio = self.infer_coreset_sampling_ratio()

        bank_dtype = self.resolve_bank_dtype(profile)
        bank_dtype_bytes = self.dtype_nbytes(bank_dtype)

        raw_bytes = profile.patches_per_image * profile.embedding_dim * bank_dtype_bytes

        coreset_bytes = raw_bytes * coreset_sampling_ratio
        dataset_workspace_bytes = raw_bytes * dataset_workspace_factor

        raw_phase_bytes = raw_bytes + dataset_workspace_bytes
        coreset_phase_bytes = coreset_bytes

        if include_coreset_in_peak:
            peak_bytes = raw_bytes + dataset_workspace_bytes + coreset_bytes
        else:
            peak_bytes = max(raw_phase_bytes, coreset_phase_bytes)

        return {
            "bank_dtype": str(bank_dtype),
            "bank_dtype_bytes": bank_dtype_bytes,
            "raw_bank_per_image_mib": bytes_to_mib(raw_bytes),
            "coreset_bank_per_image_mib": bytes_to_mib(coreset_bytes),
            "dataset_workspace_per_image_mib": bytes_to_mib(dataset_workspace_bytes),
            "raw_phase_per_image_mib": bytes_to_mib(raw_phase_bytes),
            "coreset_phase_per_image_mib": bytes_to_mib(coreset_phase_bytes),
            "peak_bank_per_image_mib": bytes_to_mib(peak_bytes),
            "coreset_sampling_ratio": coreset_sampling_ratio,
            "dataset_workspace_factor": dataset_workspace_factor,
            "include_coreset_in_peak": include_coreset_in_peak,
        }

    def query_patches(self, profile: FeatureProfile, *, inference_chunk_size: int | None) -> int:
        """Number of query rows in the inference distance matrix.

        anomalib chunks the eval batch at ``inference_chunk_size`` patches; a None
        chunk size falls back to the full batch (conservative estimate).
        """
        n_query = self.tile_config.batch_size * profile.patches_per_image
        if inference_chunk_size:
            n_query = min(n_query, inference_chunk_size)
        return n_query

    def inference_distance_matrix_mib(
        self,
        profile: FeatureProfile,
        *,
        memory_bank_patches: float,
        inference_chunk_size: int | None,
    ) -> float:
        """Peak of the ``euclidean_dist`` matrix during validation/inference.

        ``euclidean_dist(x, y)`` materialises a ``(n_query, M)`` tensor where ``M``
        is the coreset/memory-bank patch count. euclidean_dist is in-place (1x
        transient), so cost is dominated by the coreset size and grows with
        #train images.
        """
        n_query = self.query_patches(profile, inference_chunk_size=inference_chunk_size)
        bank_dtype = self.resolve_bank_dtype(profile)
        n_bytes = n_query * memory_bank_patches * self.dtype_nbytes(bank_dtype)
        factor = euclidean_dist_transient_factor(inference_chunk_size)
        return bytes_to_mib(n_bytes) * factor

    def memory_bank_patches(
        self,
        profile: FeatureProfile,
        *,
        n_train_images: float,
        coreset_sampling_ratio: float,
    ) -> float:
        """Coreset/memory-bank patch count for a given number of training images."""
        return n_train_images * profile.patches_per_image * coreset_sampling_ratio

    def calibrate_workspace_factor(
        self,
        *,
        observed_peak_mib: float,
        n_train_images: int,
        profile: FeatureProfile | None = None,
        fixed_overhead_mib: float = 0.0,
    ) -> float:
        if profile is None:
            profile = self.profile_features()

        bank_dtype = self.resolve_bank_dtype(profile)
        bank_dtype_bytes = self.dtype_nbytes(bank_dtype)

        raw_bank_per_image_mib = bytes_to_mib(profile.patches_per_image * profile.embedding_dim * bank_dtype_bytes)

        activation_mib = self.activation_peak_mib(profile)

        dataset_mib = observed_peak_mib - activation_mib - fixed_overhead_mib

        if n_train_images <= 0 or raw_bank_per_image_mib <= 0:
            return 0.0

        observed_total_factor = dataset_mib / (n_train_images * raw_bank_per_image_mib)

        # Peak model: raw * (1 + workspace_factor)
        return max(0.0, observed_total_factor - 1.0)

    def estimate_from_profile(
        self,
        *,
        profile: FeatureProfile,
        memory_budget: MemoryBudget,
        n_train_images: int | None = None,
        coreset_sampling_ratio: float | None = None,
        dataset_workspace_factor: float = 0.0,
        include_coreset_in_peak: bool = False,
        inference_chunk_size: int | None = None,
    ) -> PatchCoreMemoryEstimate:
        if coreset_sampling_ratio is None:
            coreset_sampling_ratio = self.infer_coreset_sampling_ratio()

        chunk_size = detect_inference_chunk_size() if inference_chunk_size is None else inference_chunk_size

        per = self.per_image_bank_mib(
            profile,
            coreset_sampling_ratio=coreset_sampling_ratio,
            dataset_workspace_factor=dataset_workspace_factor,
            include_coreset_in_peak=include_coreset_in_peak,
        )

        activation_mib = self.activation_peak_mib(profile)
        effective_budget_mib = self.effective_budget_mib(
            memory_budget,
            activation_mib,
        )

        # Constraint 1: coreset/memory-bank storage (grows with #train images).
        per_image_peak_mib = per["peak_bank_per_image_mib"]
        max_images_bank = 0 if effective_budget_mib <= 0 else int(effective_budget_mib // per_image_peak_mib)

        # Constraint 2: validation/inference peak. anomalib scores each eval batch
        # against the whole coreset, so at that point the GPU holds BOTH the
        # resident coreset (M x d) AND the distance matrix (n_query x M). Both grow
        # with #train images (M = n_train * patches_per_image * coreset_ratio), so
        # the per-image cost is the sum of the two.
        n_query = self.query_patches(profile, inference_chunk_size=chunk_size)
        bank_dtype_bytes = per["bank_dtype_bytes"]
        patches_per_train_image = profile.patches_per_image * coreset_sampling_ratio
        distance_transient_factor = euclidean_dist_transient_factor(chunk_size)
        distance_mib_per_image = bytes_to_mib(n_query * patches_per_train_image * bank_dtype_bytes) * distance_transient_factor
        resident_coreset_mib_per_image = per["coreset_bank_per_image_mib"]
        inference_mib_per_image = distance_mib_per_image + resident_coreset_mib_per_image
        max_images_distance = (
            0 if effective_budget_mib <= 0 or inference_mib_per_image <= 0 else int(effective_budget_mib // inference_mib_per_image)
        )

        # The binding constraint is the smaller of the two.
        max_images = min(max_images_bank, max_images_distance)
        binding_constraint = "inference_distance_matrix" if max_images_distance <= max_images_bank else "memory_bank_storage"

        fits_requested = None
        requested = {}

        if n_train_images is not None:
            fits_requested = n_train_images <= max_images

            bank_patches = self.memory_bank_patches(
                profile,
                n_train_images=n_train_images,
                coreset_sampling_ratio=coreset_sampling_ratio,
            )
            distance_matrix_mib = self.inference_distance_matrix_mib(
                profile,
                memory_bank_patches=bank_patches,
                inference_chunk_size=chunk_size,
            )
            bank_peak_mib = n_train_images * per["peak_bank_per_image_mib"]
            resident_coreset_mib = n_train_images * per["coreset_bank_per_image_mib"]
            # Validation peak: the coreset stays resident while the distance matrix
            # is allocated, so they add up.
            inference_peak_mib = distance_matrix_mib + resident_coreset_mib

            requested = {
                "requested_train_images": n_train_images,
                "requested_raw_bank_mib": (n_train_images * per["raw_bank_per_image_mib"]),
                "requested_coreset_bank_mib": resident_coreset_mib,
                "requested_dataset_workspace_mib": (n_train_images * per["dataset_workspace_per_image_mib"]),
                "requested_variable_peak_bank_mib": bank_peak_mib,
                "requested_inference_distance_matrix_mib": distance_matrix_mib,
                "requested_inference_peak_mib": inference_peak_mib,
                "requested_total_peak_mib": (memory_budget.fixed_overhead_mib + activation_mib + max(bank_peak_mib, inference_peak_mib)),
            }

        return PatchCoreMemoryEstimate(
            max_train_images=max_images,
            requested_train_images=n_train_images,
            fits_requested_train_images=fits_requested,
            budget=self.common_budget_dict(
                memory_budget,
                activation_mib=activation_mib,
                effective_budget_mib=effective_budget_mib,
                extra={
                    "max_train_images_bank_storage": max_images_bank,
                    "max_train_images_inference_distance": max_images_distance,
                    "binding_constraint": binding_constraint,
                },
            ),
            per_image={
                **per,
                "inference_distance_matrix_per_image_mib": distance_mib_per_image,
            },
            requested=requested,
            model={
                "embedding_dim": profile.embedding_dim,
                "patches_per_image": profile.patches_per_image,
                "coreset_sampling_ratio": coreset_sampling_ratio,
                "dataset_workspace_factor": dataset_workspace_factor,
                "inference_chunk_size": chunk_size,
                "inference_query_patches": n_query,
                **self.common_model_dtype_dict(profile),
                "resolved_bank_dtype": per["bank_dtype"],
                "resolved_bank_dtype_bytes": per["bank_dtype_bytes"],
            },
            tiling=self.common_tiling_dict(profile),
            feature_profile=asdict(profile),
        )
