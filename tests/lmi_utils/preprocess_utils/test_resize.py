"""Focused tests for ResizeOperation behavior not covered by the pipeline/coord suites.

Existing coverage lives in:
- test_reconstruct_coordinates.py::TestResize (2x downscale revert, masks, preserve_aspect)
- test_preprocessor.py (parametrized forward)
- test_chain_pipeline.py (resize in chains)
- test_steps.py (builder kwargs)

This file fills the remaining gaps: no-op short-circuit, defaulted width/height,
forward-direction coordinate mapping (apply_coordinates), and length-mismatch errors.
"""

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.ops.resize import ResizeOperation
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc(h, w, c=3):
    return torch.arange(h * w * c, dtype=torch.float32).reshape(h, w, c)


def _empty_results(n=1, **overrides):
    """Per-image results dict with empty defaults for every coord field."""
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
    """No interpolation when tw == w0 and th == h0; metadata still emitted."""
    pre = Preprocessor()
    img = _hwc(40, 60, 3)
    steps = [{"type": "resize", "configuration": {"width": 60, "height": 40}}]
    out, history = pre.preprocess([img], steps)

    assert out[0].shape == (40, 60, 3)
    assert torch.equal(out[0], img)
    meta = history[0]["metadata"][0]
    assert meta["src_size"] == [60, 40]
    assert meta["dst_size"] == [60, 40]
    assert "pad" not in meta


def test_resize_defaults_to_current_dim_when_width_or_height_missing():
    """Omitting width keeps W; omitting height keeps H."""
    pre = Preprocessor()
    img = _hwc(40, 60, 3)

    out_w, _ = pre.preprocess([img], [{"type": "resize", "configuration": {"height": 20}}])
    assert out_w[0].shape == (20, 60, 3)

    out_h, _ = pre.preprocess([img], [{"type": "resize", "configuration": {"width": 30}}])
    assert out_h[0].shape == (40, 30, 3)


def test_resize_apply_coordinates_forward_scales_and_pads():
    """apply_coordinates maps original-space coords into preprocessed space.

    200(H)x100(W) → preserve_aspect resize to 100x100: scale=0.5 → 100x50, pad L/R=25.
    A box [10, 20, 90, 180] in original space should land at
    [10*0.5+25, 20*0.5, 90*0.5+25, 180*0.5] = [30, 10, 70, 90].
    """
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(200, 100, 3)
    steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": True}}]
    _out, history = pre.preprocess([img], steps)

    results = _empty_results(boxes=[torch.tensor([[10.0, 20.0, 90.0, 180.0]])])
    forwarded = rec.apply_coordinates(results, history)
    assert torch.allclose(forwarded["boxes"][0], torch.tensor([[30.0, 10.0, 70.0, 90.0]]), atol=1e-4)


def test_resize_apply_coordinates_masks_preserve_aspect():
    """Forward apply_coordinates on a mask: resize then pad into the full target canvas.

    200(H)x100(W) -> preserve_aspect 100x100: scale=0.5 -> 100x50, pad L/R=25 (T/B=0).
    A solid filled mask (1, 200, 100) of all ones should land as a (1, 100, 100) canvas
    that is ones inside columns [25, 75) and zeros in the L/R pad strips.
    """
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(200, 100, 3)
    steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": True}}]
    _out, history = pre.preprocess([img], steps)

    masks = torch.ones((1, 200, 100))
    forwarded = rec.apply_coordinates(_empty_results(masks=[masks]), history)

    out = forwarded["masks"][0]
    assert out.shape == (1, 100, 100)
    assert torch.all(out[:, :, :25] == 0), "left pad strip should be zero"
    assert torch.all(out[:, :, 75:] == 0), "right pad strip should be zero"
    assert torch.all(out[:, :, 25:75] == 1), "scaled mask region should be one"


def test_resize_apply_then_revert_is_identity_on_boxes():
    """Round trip original → preprocessed → original recovers the input box."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(120, 80, 3)
    steps = [{"type": "resize", "configuration": {"width": 40, "height": 60}}]
    _out, history = pre.preprocess([img], steps)

    src_box = torch.tensor([[8.0, 12.0, 40.0, 90.0]])
    forwarded = rec.apply_coordinates(_empty_results(boxes=[src_box]), history)
    reverted = rec.reconstruct_coordinates(forwarded, history)
    assert torch.allclose(reverted["boxes"][0], src_box, atol=1e-4)


def test_resize_obb_boxes_round_trip():
    """OBB boxes (N, 4, 2) take the corner-flatten path — every corner gets independently scaled."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(200, 100, 3)
    # Free stretch: sx = 50/100 = 0.5, sy = 100/200 = 0.5.
    steps = [{"type": "resize", "configuration": {"width": 50, "height": 100}}]
    _out, history = pre.preprocess([img], steps)

    obb_orig = torch.tensor([[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]])
    forward = rec.apply_coordinates(_empty_results(boxes=[obb_orig]), history)
    expected = torch.tensor([[[5.0, 10.0], [15.0, 10.0], [15.0, 20.0], [5.0, 20.0]]])
    assert forward["boxes"][0].shape == (1, 4, 2)
    assert torch.allclose(forward["boxes"][0], expected, atol=1e-4)

    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], obb_orig, atol=1e-4)


def test_resize_segments_variable_length():
    """Each segment is its own (Mi, 2) tensor; Mi can differ across segments in the same list."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc(200, 100, 3)
    # Free stretch: sx = sy = 0.5; revert scales by 2.
    steps = [{"type": "resize", "configuration": {"width": 50, "height": 100}}]
    _out, history = pre.preprocess([img], steps)

    # In preprocessed space.
    seg_short = torch.tensor([[5.0, 10.0]])
    seg_long = torch.tensor([[2.5, 5.0], [12.5, 25.0], [40.0, 90.0]])
    results = _empty_results(segments=[[seg_short, seg_long]])
    reverted = rec.reconstruct_coordinates(results, history)
    out_segs = reverted["segments"][0]
    assert len(out_segs) == 2
    assert torch.allclose(out_segs[0], torch.tensor([[10.0, 20.0]]), atol=1e-4)
    assert torch.allclose(out_segs[1], torch.tensor([[5.0, 10.0], [25.0, 50.0], [80.0, 180.0]]), atol=1e-4)


def test_resize_preserve_aspect_pads_top_bottom_for_wide_image():
    """Wide source (W > H) preserve_aspect-resized to a square pads top/bottom, not left/right."""
    pre, rec = Preprocessor(), Reconstructor()
    # 60(H)x200(W) → 100x100 preserve_aspect: scale=0.5, dst=100x30, pad 35 top + 35 bottom.
    img = _hwc(60, 200, 3)
    steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": True}}]
    _out, history = pre.preprocess([img], steps)

    meta = history[0]["metadata"][0]
    assert meta["src_size"] == [200, 60]
    assert meta["dst_size"] == [100, 30]
    pL, pR, pT, pB = meta["pad"]
    assert pL == 0 and pR == 0
    assert pT + pB == 70

    # Forward a box from original space: [40, 20, 80, 40] → scale 0.5 → [20, 10, 40, 20] → +pT on y.
    box_orig = torch.tensor([[40.0, 20.0, 80.0, 40.0]])
    forward = rec.apply_coordinates(_empty_results(boxes=[box_orig]), history)
    expected = torch.tensor([[20.0, 10.0 + pT, 40.0, 20.0 + pT]])
    assert torch.allclose(forward["boxes"][0], expected, atol=1e-4)

    # Round-trip recovers the original box.
    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], box_orig, atol=1e-4)


def test_resize_preserve_aspect_no_pad_when_aspect_matches():
    """When source aspect already matches target, the resize emits no 'pad' key."""
    pre, rec = Preprocessor(), Reconstructor()
    # 100(H)x200(W), aspect 2:1 — target 50x100 has the same aspect, so no padding.
    img = _hwc(100, 200, 3)
    steps = [{"type": "resize", "configuration": {"width": 100, "height": 50, "preserve_aspect": True}}]
    _out, history = pre.preprocess([img], steps)

    meta = history[0]["metadata"][0]
    assert meta["src_size"] == [200, 100]
    assert meta["dst_size"] == [100, 50]
    assert "pad" not in meta

    # Forward + revert still works for boxes without a pad term.
    box_orig = torch.tensor([[10.0, 20.0, 80.0, 40.0]])
    forward = rec.apply_coordinates(_empty_results(boxes=[box_orig]), history)
    assert torch.allclose(forward["boxes"][0], torch.tensor([[5.0, 10.0, 40.0, 20.0]]), atol=1e-4)

    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], box_orig, atol=1e-4)


def test_resize_revert_images_length_mismatch_raises():
    """_check_len rejects inputs that disagree with metadata length."""
    op = ResizeOperation()
    img = _hwc(10, 10, 3)
    meta = [{"src_size": [10, 10], "dst_size": [10, 10]}, {"src_size": [10, 10], "dst_size": [10, 10]}]
    with pytest.raises(ValueError, match=r"resize: input length \(1\) != metadata length \(2\)"):
        op.revert_images([img], meta)
