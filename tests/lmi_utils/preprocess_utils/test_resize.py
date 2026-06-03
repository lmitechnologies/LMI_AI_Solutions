"""Focused tests for ResizeOperation."""

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.ops import ResizeMeta
from lmi_utils.preprocess_utils.ops.resize import ResizeOperation
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc(h, w, c=3):
    return torch.arange(h * w * c, dtype=torch.float32).reshape(h, w, c)


def _empty_results(n=1, **overrides):
    results = {
        "boxes": [torch.zeros((0, 4)) for _ in range(n)],
        "scores": [torch.zeros((0,)) for _ in range(n)],
        "classes": [np.zeros((0,), dtype=np.int32) for _ in range(n)],
        "segments": [[] for _ in range(n)],
        "points": [torch.zeros((0, 1, 3)) for _ in range(n)],
    }
    results.update(overrides)
    return results


def test_resize_noop_when_target_matches_input():
    pre = Preprocessor()
    img = _hwc(40, 60, 3)
    out, history = pre.preprocess([img], [steps.resize(width=60, height=40)])

    assert out[0].shape == (40, 60, 3)
    assert torch.equal(out[0], img)
    meta = history[0]
    assert meta.src_sizes[0] == [60, 40]
    assert meta.dst_sizes[0] == [60, 40]
    assert meta.pads[0] == [0, 0, 0, 0]


def test_resize_defaults_to_current_dim_when_width_or_height_missing():
    pre = Preprocessor()
    img = _hwc(40, 60, 3)

    out_w, _ = pre.preprocess([img], [steps.resize(height=20)])
    assert out_w[0].shape == (20, 60, 3)

    out_h, _ = pre.preprocess([img], [steps.resize(width=30)])
    assert out_h[0].shape == (40, 30, 3)


def test_resize_apply_coordinates_forward_scales_and_pads():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(200, 100, 3)
    _out, history = pre.preprocess([img], [steps.resize(width=100, height=100, preserve_aspect=True)])

    results = _empty_results(boxes=[torch.tensor([[10.0, 20.0, 90.0, 180.0]])])
    forwarded = rec.apply_coordinates(results, history)
    assert torch.allclose(forwarded["boxes"][0], torch.tensor([[30.0, 10.0, 70.0, 90.0]]), atol=1e-4)


def test_resize_apply_coordinates_masks_preserve_aspect():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(200, 100, 3)
    _out, history = pre.preprocess([img], [steps.resize(width=100, height=100, preserve_aspect=True)])

    masks = torch.ones((1, 200, 100))
    forwarded = rec.apply_coordinates(_empty_results(masks=[masks]), history)

    out = forwarded["masks"][0]
    assert out.shape == (1, 100, 100)
    assert torch.all(out[:, :, :25] == 0)
    assert torch.all(out[:, :, 75:] == 0)
    assert torch.all(out[:, :, 25:75] == 1)


def test_resize_apply_then_revert_is_identity_on_boxes():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(120, 80, 3)
    _out, history = pre.preprocess([img], [steps.resize(width=40, height=60)])

    src_box = torch.tensor([[8.0, 12.0, 40.0, 90.0]])
    forwarded = rec.apply_coordinates(_empty_results(boxes=[src_box]), history)
    reverted = rec.reconstruct_coordinates(forwarded, history)
    assert torch.allclose(reverted["boxes"][0], src_box, atol=1e-4)


def test_resize_obb_boxes_round_trip():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(200, 100, 3)
    _out, history = pre.preprocess([img], [steps.resize(width=50, height=100)])

    obb_orig = torch.tensor([[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]])
    forward = rec.apply_coordinates(_empty_results(boxes=[obb_orig]), history)
    expected = torch.tensor([[[5.0, 10.0], [15.0, 10.0], [15.0, 20.0], [5.0, 20.0]]])
    assert forward["boxes"][0].shape == (1, 4, 2)
    assert torch.allclose(forward["boxes"][0], expected, atol=1e-4)

    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], obb_orig, atol=1e-4)


def test_resize_segments_variable_length():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(200, 100, 3)
    _out, history = pre.preprocess([img], [steps.resize(width=50, height=100)])

    seg_short = torch.tensor([[5.0, 10.0]])
    seg_long = torch.tensor([[2.5, 5.0], [12.5, 25.0], [40.0, 90.0]])
    results = _empty_results(segments=[[seg_short, seg_long]])
    reverted = rec.reconstruct_coordinates(results, history)
    out_segs = reverted["segments"][0]
    assert torch.allclose(out_segs[0], torch.tensor([[10.0, 20.0]]), atol=1e-4)
    assert torch.allclose(out_segs[1], torch.tensor([[5.0, 10.0], [25.0, 50.0], [80.0, 180.0]]), atol=1e-4)


def test_resize_preserve_aspect_pads_top_bottom_for_wide_image():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(60, 200, 3)
    _out, history = pre.preprocess([img], [steps.resize(width=100, height=100, preserve_aspect=True)])

    meta = history[0]
    assert meta.src_sizes[0] == [200, 60]
    assert meta.dst_sizes[0] == [100, 30]
    pL, pR, pT, pB = meta.pads[0]
    assert pL == 0 and pR == 0
    assert pT + pB == 70

    box_orig = torch.tensor([[40.0, 20.0, 80.0, 40.0]])
    forward = rec.apply_coordinates(_empty_results(boxes=[box_orig]), history)
    expected = torch.tensor([[20.0, 10.0 + pT, 40.0, 20.0 + pT]])
    assert torch.allclose(forward["boxes"][0], expected, atol=1e-4)

    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], box_orig, atol=1e-4)


def test_resize_preserve_aspect_no_pad_when_aspect_matches():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(100, 200, 3)
    _out, history = pre.preprocess([img], [steps.resize(width=100, height=50, preserve_aspect=True)])

    meta = history[0]
    assert meta.src_sizes[0] == [200, 100]
    assert meta.dst_sizes[0] == [100, 50]
    assert meta.pads[0] == [0, 0, 0, 0]

    box_orig = torch.tensor([[10.0, 20.0, 80.0, 40.0]])
    forward = rec.apply_coordinates(_empty_results(boxes=[box_orig]), history)
    assert torch.allclose(forward["boxes"][0], torch.tensor([[5.0, 10.0, 40.0, 20.0]]), atol=1e-4)

    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], box_orig, atol=1e-4)


def test_resize_revert_images_length_mismatch_raises():
    op = ResizeOperation()
    img = _hwc(10, 10, 3)
    meta = ResizeMeta(src_sizes=[[10, 10], [10, 10]], dst_sizes=[[10, 10], [10, 10]], pads=[[0, 0, 0, 0], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match=r"resize: input length \(1\) != metadata length \(2\)"):
        op.revert_images([img], meta)
