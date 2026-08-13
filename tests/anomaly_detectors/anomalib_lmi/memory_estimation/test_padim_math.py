# tests/memory/test_padim_math.py

import pytest
import torch

from anomaly_detectors.anomalib_lmi.v2.memory_estimation.common import MemoryBudget
from anomaly_detectors.anomalib_lmi.v2.memory_estimation.padim import (
    PadimMemoryEstimator as MemoryEstimator,
)

# Budget matching a 24 GB device (kept fixed so the math is deterministic
# regardless of the machine running the tests).
MEMORY_LIMIT_MIB = 24_000
RESERVE_MIB = 500

# PaDiM stats dimensions from the tiled_config / patchcore_profile_fp16 fixtures:
# patches_per_image=1936, n_features=100, float32 stats (4 bytes).
N_FEATURES = 100
PATCHES_PER_IMAGE = 1936
STATS_DTYPE_BYTES = 4
BYTES_PER_MIB = 1024**2


class FakePadim:
    n_features = N_FEATURES


def test_padim_stats_scale_with_n_features(tiled_config, patchcore_profile_fp16):
    est = MemoryEstimator(
        model=FakePadim(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
        stats_dtype=torch.float32,
    )

    estimate = est.estimate(
        memory_budget=MemoryBudget(memory_limit_mib=MEMORY_LIMIT_MIB, reserve_mib=RESERVE_MIB),
        profile=patchcore_profile_fp16,
        padim_workspace_factor=0.25,
        n_train_images=1000,
    )

    assert estimate.model["padim_dim_used"] == N_FEATURES
    assert estimate.model["padim_covariance_mib"] == pytest.approx(
        (PATCHES_PER_IMAGE * N_FEATURES * N_FEATURES * STATS_DTYPE_BYTES) / BYTES_PER_MIB
    )
    assert estimate.max_train_images > 0


def test_padim_stats_peak_counts_covariance_and_inv_covariance(
    tiled_config,
    patchcore_profile_fp16,
):
    est = MemoryEstimator(
        model=FakePadim(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
        stats_dtype=torch.float32,
    )

    stats = est.stats_mib(patchcore_profile_fp16)

    P, d, b = PATCHES_PER_IMAGE, N_FEATURES, STATS_DTYPE_BYTES
    cov = (P * d * d * b) / BYTES_PER_MIB
    inv = (P * d * d * b) / BYTES_PER_MIB
    mean = (P * d * b) / BYTES_PER_MIB
    identity = (d * d * b) / BYTES_PER_MIB

    assert stats["padim_covariance_mib"] == pytest.approx(cov)
    assert stats["padim_inv_covariance_mib"] == pytest.approx(inv)
    # Peak holds covariance workspace AND inv_covariance simultaneously (~2 P d d).
    assert stats["padim_stats_peak_mib"] == pytest.approx(cov + inv + mean + identity)
    assert stats["padim_stats_persistent_mib"] == pytest.approx(inv + mean)
    assert stats["padim_stats_mib"] == pytest.approx(stats["padim_stats_peak_mib"])


def test_padim_estimate_type_and_budget_uses_stats_peak(
    tiled_config,
    patchcore_profile_fp16,
):
    est = MemoryEstimator(
        model=FakePadim(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
        stats_dtype=torch.float32,
    )

    estimate = est.estimate(
        memory_budget=MemoryBudget(memory_limit_mib=MEMORY_LIMIT_MIB, reserve_mib=RESERVE_MIB),
        profile=patchcore_profile_fp16,
        n_train_images=1000,
    )

    assert estimate.model_type == "padim"
    assert estimate.budget["padim_stats_peak_mib"] == pytest.approx(estimate.model["padim_stats_peak_mib"])
    assert estimate.fits_requested_train_images is not None


def test_padim_calibration_recovers_workspace_factor(
    tiled_config,
    patchcore_profile_fp16,
):
    est = MemoryEstimator(
        model=FakePadim(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
        stats_dtype=torch.float32,
    )

    per = est.per_image_embedding_mib(patchcore_profile_fp16, padim_workspace_factor=0.0)
    activation = est.activation_peak_mib(patchcore_profile_fp16)
    stats_peak = est.stats_mib(patchcore_profile_fp16)["padim_stats_peak_mib"]

    n = 500
    factor = 0.3
    observed = activation + stats_peak + n * per["train_embedding_per_image_mib"] * (1 + factor)

    recovered = est.calibrate_workspace_factor(
        observed_peak_mib=observed,
        n_train_images=n,
        profile=patchcore_profile_fp16,
    )

    assert recovered == pytest.approx(factor, rel=1e-6)
