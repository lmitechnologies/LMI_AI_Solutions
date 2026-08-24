# tests/memory/test_factory.py

import pytest

from anomaly_detectors.anomalib_lmi.v2.memory_estimation import (
    PadimMemoryEstimator,
    PatchCoreMemoryEstimator,
    make_memory_estimator,
)


class Patchcore:
    pass


class Padim:
    n_features = 100


class EfficientAd:
    pass


class Unknown:
    pass


def test_factory_builds_patchcore(tiled_config):
    est = make_memory_estimator(
        Patchcore(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )
    assert isinstance(est, PatchCoreMemoryEstimator)


def test_factory_builds_padim(tiled_config):
    est = make_memory_estimator(
        Padim(),
        tile_config=tiled_config,
        precision="float16",
        profiling_device="cpu",
    )
    assert isinstance(est, PadimMemoryEstimator)


def test_factory_rejects_unknown(tiled_config):
    with pytest.raises(NotImplementedError):
        make_memory_estimator(Unknown(), tile_config=tiled_config)
