# tests/memory/test_patchcore_math.py

import pytest

from anomaly_detectors.anomalib_lmi.v2.memory_estimation.common import MemoryBudget
from anomaly_detectors.anomalib_lmi.v2.memory_estimation.patchcore import (
    ANOMALIB_QUERY_CHUNK_SIZE,
    EUCLIDEAN_DIST_TRANSIENT_FACTOR,
)
from anomaly_detectors.anomalib_lmi.v2.memory_estimation.patchcore import (
    PatchCoreMemoryEstimator as MemoryEstimator,
)

# --- Shared test inputs (derived from the tiled_config / patchcore_profile_fp16
# fixtures: patches_per_image=1936, embedding_dim=1536, fp16 bank). ---
CORESET_SAMPLING_RATIO = 0.07
DATASET_WORKSPACE_FACTOR = 1.1641

# Budget matching a 24 GB device (kept fixed so the math is deterministic
# regardless of the machine running the tests).
MEMORY_LIMIT_MIB = 24_000
RESERVE_MIB = 757

# Calibration inputs (an observed peak / image count from a real 24 GB run).
OBSERVED_PEAK_MIB = 23_243
OBSERVED_N_TRAIN = 1782

# --- Expected outputs for the fp16 fixture (coreset excluded from peak). ---
RAW_BANK_PER_IMAGE_MIB = 5.671875
CORESET_BANK_PER_IMAGE_MIB = 0.39703125
DATASET_WORKSPACE_PER_IMAGE_MIB = 6.60263
PEAK_BANK_PER_IMAGE_MIB = 12.274505
EXPECTED_MAX_TRAIN_IMAGES = 1839
EXPECTED_WORKSPACE_FACTOR = 1.234


def test_patchcore_fp16_per_image_math(tiled_config, patchcore_profile_fp16):
    est = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )

    per = est.per_image_bank_mib(
        patchcore_profile_fp16,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=DATASET_WORKSPACE_FACTOR,
    )

    assert per["bank_dtype"] == "torch.float16"
    assert per["bank_dtype_bytes"] == 2
    assert per["raw_bank_per_image_mib"] == pytest.approx(RAW_BANK_PER_IMAGE_MIB)
    assert per["coreset_bank_per_image_mib"] == pytest.approx(CORESET_BANK_PER_IMAGE_MIB)
    assert per["dataset_workspace_per_image_mib"] == pytest.approx(DATASET_WORKSPACE_PER_IMAGE_MIB, rel=1e-4)
    assert per["peak_bank_per_image_mib"] == pytest.approx(PEAK_BANK_PER_IMAGE_MIB, rel=1e-4)


def test_patchcore_fp32_per_image_is_about_2x_fp16(
    tiled_config,
    patchcore_profile_fp16,
    patchcore_profile_fp32,
):
    est16 = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )
    est32 = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float32",
        profiling_device="cpu",
    )

    per16 = est16.per_image_bank_mib(
        patchcore_profile_fp16,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=DATASET_WORKSPACE_FACTOR,
    )
    per32 = est32.per_image_bank_mib(
        patchcore_profile_fp32,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=DATASET_WORKSPACE_FACTOR,
    )

    assert per32["peak_bank_per_image_mib"] == pytest.approx(
        2 * per16["peak_bank_per_image_mib"],
        rel=1e-6,
    )


def test_patchcore_calibration_recovers_workspace_factor(
    tiled_config,
    patchcore_profile_fp16,
):
    est = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )

    factor = est.calibrate_workspace_factor(
        observed_peak_mib=OBSERVED_PEAK_MIB,
        n_train_images=OBSERVED_N_TRAIN,
        profile=patchcore_profile_fp16,
        fixed_overhead_mib=0.0,
    )

    assert factor == pytest.approx(EXPECTED_WORKSPACE_FACTOR, rel=0.01)


def test_patchcore_estimate_known_target_1782(
    tiled_config,
    patchcore_profile_fp16,
):
    est = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )

    estimate = est.estimate(
        memory_budget=MemoryBudget(
            memory_limit_mib=MEMORY_LIMIT_MIB,
            reserve_mib=RESERVE_MIB,
            fixed_overhead_mib=0,
            safety_fraction=1.0,
        ),
        profile=patchcore_profile_fp16,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=DATASET_WORKSPACE_FACTOR,
        # Chunked query keeps the inference distance matrix small so bank storage
        # is the binding constraint (matches the original 24 GB calibration).
        inference_chunk_size=ANOMALIB_QUERY_CHUNK_SIZE,
    )

    assert estimate.budget["binding_constraint"] == "memory_bank_storage"
    assert estimate.max_train_images == EXPECTED_MAX_TRAIN_IMAGES


def test_patchcore_inference_distance_binds_when_coreset_keeps_every_patch(
    tiled_config,
    patchcore_profile_fp16,
):
    """With the whole bank kept and no dataset workspace, the validation peak (coreset + distance matrix)
    outgrows bank storage."""
    est = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )

    estimate = est.estimate(
        memory_budget=MemoryBudget(
            memory_limit_mib=MEMORY_LIMIT_MIB,
            reserve_mib=RESERVE_MIB,
            fixed_overhead_mib=0,
            safety_fraction=1.0,
        ),
        profile=patchcore_profile_fp16,
        coreset_sampling_ratio=1.0,
        dataset_workspace_factor=0.0,
    )

    assert estimate.budget["binding_constraint"] == "inference_distance_matrix"
    assert estimate.max_train_images == estimate.budget["max_train_images_inference_distance"]
    assert estimate.max_train_images < estimate.budget["max_train_images_bank_storage"]


def test_patchcore_distance_matrix_is_one_query_chunk_by_bank(
    tiled_config,
    patchcore_profile_fp16,
):
    """euclidean_dist is in-place, so the peak is a single (chunk_size, M) tensor."""
    est = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )

    memory_bank_patches = 10_000
    bank_dtype_bytes = est.dtype_nbytes(est.resolve_bank_dtype(patchcore_profile_fp16))

    n_query = est.query_patches(patchcore_profile_fp16, inference_chunk_size=ANOMALIB_QUERY_CHUNK_SIZE)
    assert n_query == ANOMALIB_QUERY_CHUNK_SIZE

    distance = est.inference_distance_matrix_mib(
        patchcore_profile_fp16,
        memory_bank_patches=memory_bank_patches,
        inference_chunk_size=ANOMALIB_QUERY_CHUNK_SIZE,
    )
    single_mib = (n_query * memory_bank_patches * bank_dtype_bytes) / 1024**2
    assert distance == pytest.approx(EUCLIDEAN_DIST_TRANSIENT_FACTOR * single_mib)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_patchcore_rejects_non_positive_chunk_size(tiled_config, patchcore_profile_fp16, chunk_size):
    est = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )

    with pytest.raises(ValueError, match="inference_chunk_size"):
        est.estimate(
            memory_budget=MemoryBudget(memory_limit_mib=MEMORY_LIMIT_MIB, reserve_mib=RESERVE_MIB),
            profile=patchcore_profile_fp16,
            coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
            inference_chunk_size=chunk_size,
        )


def test_patchcore_fp32_max_roughly_half_fp16(
    tiled_config,
    patchcore_profile_fp16,
    patchcore_profile_fp32,
):
    budget = MemoryBudget(
        memory_limit_mib=MEMORY_LIMIT_MIB,
        reserve_mib=RESERVE_MIB,
        fixed_overhead_mib=0,
        safety_fraction=1.0,
    )

    est16 = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )
    est32 = MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float32",
        profiling_device="cpu",
    )

    e16 = est16.estimate(
        memory_budget=budget,
        profile=patchcore_profile_fp16,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=DATASET_WORKSPACE_FACTOR,
    )
    e32 = est32.estimate(
        memory_budget=budget,
        profile=patchcore_profile_fp32,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=DATASET_WORKSPACE_FACTOR,
    )

    assert e32.max_train_images < e16.max_train_images
    assert e32.max_train_images == pytest.approx(e16.max_train_images / 2, rel=0.10)
