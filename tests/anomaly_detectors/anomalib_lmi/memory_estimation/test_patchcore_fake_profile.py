# tests/memory/test_patchcore_fake_profile.py

from anomaly_detectors.anomalib_lmi.v2.memory_estimation.patchcore import (
    PatchCoreMemoryEstimator as MemoryEstimator,
)


def test_profile_fp16_features_but_fp32_embedding_resolves_fp16_bank(
    fake_patchcore_model,
    tiled_config,
):
    est = MemoryEstimator(
        model=fake_patchcore_model,
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )

    profile = est.profile_features()

    per = est.per_image_bank_mib(
        profile,
        coreset_sampling_ratio=0.07,
        dataset_workspace_factor=1.1641,
    )

    assert profile.feature_dtypes["layer2"] == "torch.float16"
    assert profile.feature_dtypes["layer3"] == "torch.float16"

    # This intentionally reproduces your real behavior.
    assert profile.embedding_dtype == "torch.float32"

    # This is the critical behavior.
    assert per["bank_dtype"] == "torch.float16"


def test_profile_fp32_resolves_fp32_bank(fake_patchcore_model, tiled_config):
    est = MemoryEstimator(
        model=fake_patchcore_model,
        tile_config=tiled_config,
        precision="float32",
        profiling_device="cpu",
    )

    profile = est.profile_features()

    per = est.per_image_bank_mib(
        profile,
        coreset_sampling_ratio=0.07,
        dataset_workspace_factor=1.1641,
    )

    assert profile.feature_dtypes["layer2"] == "torch.float32"
    assert per["bank_dtype"] == "torch.float32"
