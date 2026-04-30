"""Pin the device contract: outputs live on the same device as inputs."""

import pytest
import torch

from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

STEPS = [
    {"type": "resize", "configuration": {"width": 32, "height": 32, "preserve_aspect": True}},
    {"type": "tile", "configuration": {"tile_size": [16, 16], "stride": [16, 16]}},
]


def _device_type(t):
    """Compare device by type only ('cuda', 'cpu') — index is implementation detail."""
    return t.device.type if isinstance(t, torch.Tensor) else None


@pytest.fixture
def prep():
    return Preprocessor()


@pytest.fixture
def recon():
    return Reconstructor()


@pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda", marks=cuda)])
def test_preprocess_preserves_device(prep, device_str):
    device = torch.device(device_str)
    image = torch.zeros((40, 40, 3), dtype=torch.float32, device=device)

    processed, _ = prep.preprocess(image, STEPS)

    assert all(_device_type(t) == device.type for t in processed), [_device_type(t) for t in processed]


@pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda", marks=cuda)])
def test_reconstruct_images_preserves_device(prep, recon, device_str):
    device = torch.device(device_str)
    image = torch.zeros((40, 40, 3), dtype=torch.float32, device=device)

    processed, history = prep.preprocess(image, STEPS)
    restored = recon.reconstruct_images(processed, history)

    assert all(_device_type(t) == device.type for t in restored), [_device_type(t) for t in restored]


@pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda", marks=cuda)])
def test_reconstruct_coordinates_preserves_device(prep, recon, device_str):
    device = torch.device(device_str)
    image = torch.zeros((40, 40, 3), dtype=torch.float32, device=device)

    _, history = prep.preprocess(image, STEPS)

    # Build a per-tile result batch on `device`. After tile->resize revert,
    # coordinates should still be on `device`.
    n_tiles = 2 * 2  # 32 / 16 = 2 along each axis
    boxes_per_tile = torch.tensor([[1.0, 2.0, 5.0, 6.0]], device=device)
    masks_per_tile = torch.ones((1, 16, 16), device=device)
    segs_per_tile = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)]
    pts_per_tile = torch.tensor([[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]], device=device)
    scores_per_tile = torch.tensor([0.9], device=device)
    classes_per_tile = torch.tensor([0])

    results = {
        "boxes": [boxes_per_tile.clone() for _ in range(n_tiles)],
        "masks": [masks_per_tile.clone() for _ in range(n_tiles)],
        "segments": [list(segs_per_tile) for _ in range(n_tiles)],
        "points": [pts_per_tile.clone() for _ in range(n_tiles)],
        "scores": [scores_per_tile.clone() for _ in range(n_tiles)],
        "classes": [classes_per_tile.clone() for _ in range(n_tiles)],
    }

    out = recon.reconstruct_coordinates(results, history)

    for field in ("boxes", "masks", "points"):
        for t in out[field]:
            assert _device_type(t) == device.type, f"{field}: {_device_type(t)} != {device.type}"
    for segs in out["segments"]:
        for s in segs:
            assert _device_type(s) == device.type, f"segments: {_device_type(s)} != {device.type}"
