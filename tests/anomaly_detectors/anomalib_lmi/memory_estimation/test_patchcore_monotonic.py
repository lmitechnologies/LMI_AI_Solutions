# tests/memory/test_patchcore_monotonic.py

from anomaly_detectors.anomalib_lmi.v2.memory_estimation.common import MemoryBudget
from anomaly_detectors.anomalib_lmi.v2.memory_estimation.patchcore import (
    ANOMALIB_QUERY_CHUNK_SIZE,
)
from anomaly_detectors.anomalib_lmi.v2.memory_estimation.patchcore import (
    PatchCoreMemoryEstimator as MemoryEstimator,
)

CORESET_SAMPLING_RATIO = 0.07
DATASET_WORKSPACE_FACTOR = 1.1641


def make_est(tiled_config):
    return MemoryEstimator(
        model=object(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )


def test_more_memory_increases_max_images(tiled_config, patchcore_profile_fp16):
    est = make_est(tiled_config)

    low = est.estimate(
        memory_budget=MemoryBudget(memory_limit_mib=12_000, reserve_mib=500),
        profile=patchcore_profile_fp16,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=DATASET_WORKSPACE_FACTOR,
        inference_chunk_size=ANOMALIB_QUERY_CHUNK_SIZE,
    )
    high = est.estimate(
        memory_budget=MemoryBudget(memory_limit_mib=24_000, reserve_mib=500),
        profile=patchcore_profile_fp16,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=DATASET_WORKSPACE_FACTOR,
        inference_chunk_size=ANOMALIB_QUERY_CHUNK_SIZE,
    )

    assert high.max_train_images > low.max_train_images


def test_higher_workspace_reduces_max_images(tiled_config, patchcore_profile_fp16):
    est = make_est(tiled_config)
    budget = MemoryBudget(memory_limit_mib=24_000, reserve_mib=500)

    low = est.estimate(
        memory_budget=budget,
        profile=patchcore_profile_fp16,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=0.0,
        inference_chunk_size=ANOMALIB_QUERY_CHUNK_SIZE,
    )
    high = est.estimate(
        memory_budget=budget,
        profile=patchcore_profile_fp16,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        dataset_workspace_factor=DATASET_WORKSPACE_FACTOR,
        inference_chunk_size=ANOMALIB_QUERY_CHUNK_SIZE,
    )

    assert high.max_train_images < low.max_train_images
