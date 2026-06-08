"""Cross-op chain tests covering flip/pad mixed with cropbox/resize/tile, plus forward
apply_coords round-trips through full pipelines."""

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
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


def test_pad_then_cropbox_image_round_trip(pipeline):
    pre, rec = pipeline
    img = torch.ones((10, 8, 3), dtype=torch.float32)
    configs = [
        steps.pad(pad=[2, 3, 4, 5]),
        steps.cropbox(boxes=[[2, 4, 10, 14]]),
    ]
    out, history = pre.preprocess([img], configs)
    assert out[0].shape == (10, 8, 3)
    assert torch.equal(out[0], img)

    restored = rec.reconstruct_images(out, history)
    assert restored[0].shape == img.shape


def test_flip_then_resize_coord_round_trip(pipeline):
    pre, rec = pipeline
    img = _hwc_image(100, 80, 3)
    configs = [
        steps.flip(lr=True),
        steps.resize(width=40, height=50, preserve_aspect=False),
    ]
    _out, history = pre.preprocess([img], configs)

    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    original = _make_results(boxes)
    forward = rec.apply_coordinates(original, history)
    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], boxes, atol=1e-4)
    assert torch.allclose(round_trip["points"][0], original["points"][0], atol=1e-4)


def test_full_pipeline_coord_round_trip(pipeline):
    pre, rec = pipeline
    img = _hwc_image(100, 80, 3)
    configs = [
        steps.pad(pad=[4, 4, 6, 6]),
        steps.cropbox(boxes=[[8, 10, 80, 100]]),
        steps.resize(width=64, height=64, preserve_aspect=False),
        steps.flip(lr=True, ud=True),
    ]
    _out, history = pre.preprocess([img], configs)

    boxes = torch.tensor([[5.0, 5.0, 25.0, 35.0], [40.0, 10.0, 60.0, 60.0]])
    original = _make_results(boxes)

    forward = rec.apply_coordinates(original, history)
    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], boxes, atol=1e-3)
    assert torch.allclose(round_trip["points"][0], original["points"][0], atol=1e-3)
    for r, o in zip(round_trip["segments"][0], original["segments"][0]):
        assert torch.allclose(r, o, atol=1e-3)


def test_full_pipeline_image_reconstructs_to_padded_then_original_shape(pipeline):
    pre, rec = pipeline
    img = _hwc_image(100, 80, 3)
    configs = [
        steps.pad(pad=[4, 4, 6, 6]),
        steps.cropbox(boxes=[[8, 10, 80, 100]]),
        steps.resize(width=64, height=64, preserve_aspect=False),
        steps.flip(lr=True, ud=True),
    ]
    out, history = pre.preprocess([img], configs)
    restored = rec.reconstruct_images(out, history)
    assert restored[0].shape == img.shape


def test_tile_with_flip_lossless(pipeline):
    pre, rec = pipeline
    img = torch.randint(0, 256, (100, 100, 3), dtype=torch.uint8)
    configs = [
        steps.flip(lr=True),
        steps.tile(tile_size=50, stride=50),
    ]
    out, history = pre.preprocess([img], configs)
    assert len(out) == 4
    restored = rec.reconstruct_images(out, history)
    assert torch.equal(restored[0], img)
