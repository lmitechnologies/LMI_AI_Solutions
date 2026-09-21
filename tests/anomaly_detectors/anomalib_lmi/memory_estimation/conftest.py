# tests/memory/conftest.py

import pytest
import torch

from anomaly_detectors.anomalib_lmi.v2.memory_estimation.common import FeatureProfile, TileConfig


@pytest.fixture
def tiled_config():
    return TileConfig(
        image_size=(348, 348),
        tile_size=(174, 174),
        stride=(87, 87),
        batch_size=32,
        channels=3,
    )


@pytest.fixture
def no_tiling_config():
    return TileConfig(
        image_size=(348, 348),
        tile_size=None,
        stride=None,
        batch_size=32,
        channels=3,
    )


@pytest.fixture
def patchcore_profile_fp16():
    return FeatureProfile(
        feature_shapes={
            "layer2": (1, 512, 22, 22),
            "layer3": (1, 1024, 11, 11),
        },
        feature_dtypes={
            "layer2": "torch.float16",
            "layer3": "torch.float16",
        },
        # Deliberately fp32 because your probe can return fp32 embedding
        # while actual PatchCore memory_bank follows feature/training dtype.
        embedding_shape_per_tile=(1, 1536, 22, 22),
        embedding_dtype="torch.float32",
        stitched_embedding_shape=(32, 1536, 44, 44),
    )


@pytest.fixture
def patchcore_profile_fp32():
    return FeatureProfile(
        feature_shapes={
            "layer2": (1, 512, 22, 22),
            "layer3": (1, 1024, 11, 11),
        },
        feature_dtypes={
            "layer2": "torch.float32",
            "layer3": "torch.float32",
        },
        embedding_shape_per_tile=(1, 1536, 22, 22),
        embedding_dtype="torch.float32",
        stitched_embedding_shape=(32, 1536, 44, 44),
    )


class FakeFeatureExtractor(torch.nn.Module):
    def forward(self, x):
        b = x.shape[0]
        dtype = x.dtype
        device = x.device

        return {
            "layer2": torch.zeros(b, 512, 22, 22, dtype=dtype, device=device),
            "layer3": torch.zeros(b, 1024, 11, 11, dtype=dtype, device=device),
        }


class FakePatchCoreInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FakeFeatureExtractor()

    def generate_embedding(self, features):
        # Reproduce the real mismatch you saw:
        # feature outputs fp16, generated embedding probe fp32.
        first = next(iter(features.values()))
        b = first.shape[0]
        return torch.zeros(b, 1536, 22, 22, dtype=torch.float32, device=first.device)


class FakePatchCoreOuter:
    def __init__(self):
        self.model = FakePatchCoreInner()


@pytest.fixture
def fake_patchcore_model():
    return FakePatchCoreOuter()
