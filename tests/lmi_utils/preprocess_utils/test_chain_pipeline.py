"""Cross-op chain tests covering flip/pad mixed with crop/resize/tile, plus forward
apply_coords round-trips through full pipelines."""

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc_image(h, w, c=3):
    return torch.arange(h * w * c, dtype=torch.float32).reshape(h, w, c)


def _make_results(boxes):
    segs = [torch.stack([b[[0, 1]], b[[2, 1]], b[[2, 3]], b[[0, 3]]]) for b in boxes]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    pts = torch.cat([torch.stack([cx, cy], dim=-1).unsqueeze(1), torch.ones(len(boxes), 1, 1)], dim=-1)
    return {
        "boxes": [boxes],
        "scores": [torch.ones(len(boxes))],
        "classes": [np.zeros(len(boxes), dtype=np.int32)],
        "segments": [segs],
        "points": [pts],
    }


@pytest.fixture
def pipeline():
    return Preprocessor(), Reconstructor()


def test_pad_then_crop_image_round_trip(pipeline):
    """Pad then crop the padded region back out → reconstruct_images restores original."""
    pre, rec = pipeline
    img = torch.ones((10, 8, 3), dtype=torch.float32)
    steps = [
        {"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}},
        # crop back to just the original-content region of the padded image
        {"type": "crop", "configuration": {"boxes": [[2, 4, 10, 14]]}},
    ]
    out, history = pre.preprocess([img], steps)
    assert out[0].shape == (10, 8, 3)
    assert torch.equal(out[0], img)

    restored = rec.reconstruct_images(out, history)
    # Reconstruction lands in padded space → original size (10, 8).
    assert restored[0].shape == img.shape


def test_flip_then_resize_coord_round_trip(pipeline):
    """flip then resize: original-space box → forward → revert lands on the original."""
    pre, rec = pipeline
    img = _hwc_image(100, 80, 3)
    steps = [
        {"type": "flip", "configuration": {"lr": True}},
        {"type": "resize", "configuration": {"width": 40, "height": 50, "preserve_aspect": False}},
    ]
    _out, history = pre.preprocess([img], steps)

    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    original = _make_results(boxes)
    forward = rec.apply_coordinates(original, history)
    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], boxes, atol=1e-4)
    assert torch.allclose(round_trip["points"][0], original["points"][0], atol=1e-4)


def test_full_pipeline_coord_round_trip(pipeline):
    """pad → crop → resize → flip — full forward/reverse round-trip on coords."""
    pre, rec = pipeline
    img = _hwc_image(100, 80, 3)
    steps = [
        {"type": "pad", "configuration": {"pad": [4, 4, 6, 6]}},
        {"type": "crop", "configuration": {"boxes": [[8, 10, 80, 100]]}},
        {"type": "resize", "configuration": {"width": 64, "height": 64, "preserve_aspect": False}},
        {"type": "flip", "configuration": {"lr": True, "ud": True}},
    ]
    _out, history = pre.preprocess([img], steps)

    boxes = torch.tensor([[5.0, 5.0, 25.0, 35.0], [40.0, 10.0, 60.0, 60.0]])
    original = _make_results(boxes)

    forward = rec.apply_coordinates(original, history)
    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], boxes, atol=1e-3)
    assert torch.allclose(round_trip["points"][0], original["points"][0], atol=1e-3)
    for r, o in zip(round_trip["segments"][0], original["segments"][0]):
        assert torch.allclose(r, o, atol=1e-3)


def test_full_pipeline_image_reconstructs_to_padded_then_original_shape(pipeline):
    """Image reconstruction unwinds geometric ops back to the pre-pad input shape."""
    pre, rec = pipeline
    img = _hwc_image(100, 80, 3)
    steps = [
        {"type": "pad", "configuration": {"pad": [4, 4, 6, 6]}},
        {"type": "crop", "configuration": {"boxes": [[8, 10, 80, 100]]}},
        {"type": "resize", "configuration": {"width": 64, "height": 64, "preserve_aspect": False}},
        {"type": "flip", "configuration": {"lr": True, "ud": True}},
    ]
    out, history = pre.preprocess([img], steps)
    restored = rec.reconstruct_images(out, history)
    assert restored[0].shape == img.shape


def test_tile_with_flip_lossless(pipeline):
    """tile after flip: lossless round-trip on image data (both ops are exact)."""
    pre, rec = pipeline
    img = torch.randint(0, 256, (100, 100, 3), dtype=torch.uint8)
    steps = [
        {"type": "flip", "configuration": {"lr": True}},
        {"type": "tile", "configuration": {"tile_size": 50, "stride": 50}},
    ]
    out, history = pre.preprocess([img], steps)
    assert len(out) == 4
    restored = rec.reconstruct_images(out, history)
    assert torch.equal(restored[0], img)
