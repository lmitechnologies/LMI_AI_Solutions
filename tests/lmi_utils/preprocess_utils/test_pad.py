import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc_image(h, w, c=3):
    return torch.arange(h * w * c, dtype=torch.float32).reshape(h, w, c)


def _empty_results(n=1, **overrides):
    """Per-image results dict with empty defaults for every coord field.

    Pass `boxes=[...]`, `points=[...]`, etc. to override specific fields.
    """
    results = {
        "boxes": [torch.zeros((0, 4)) for _ in range(n)],
        "scores": [torch.zeros((0,)) for _ in range(n)],
        "classes": [np.zeros((0,), dtype=np.int32) for _ in range(n)],
        "segments": [[] for _ in range(n)],
        "points": [torch.zeros((0, 1, 3)) for _ in range(n)],
    }
    results.update(overrides)
    return results


def test_pad_forward_explicit_pad_op():
    pre = Preprocessor()
    img = torch.ones((10, 8, 3), dtype=torch.float32)
    steps = [{"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}}]
    out, history = pre.preprocess([img], steps)

    # H: 4 + 10 + 5 = 19; W: 2 + 8 + 3 = 13
    assert out[0].shape == (19, 13, 3)
    assert history[0]["metadata"][0]["pad"] == [2, 3, 4, 5]
    # Inside region keeps ones, padded border is zero.
    assert out[0][4:14, 2:10].eq(1).all()
    assert out[0][:4].eq(0).all()


def test_pad_forward_target_size_centers_padding():
    pre = Preprocessor()
    img = torch.ones((10, 8, 3), dtype=torch.float32)
    steps = [{"type": "pad", "configuration": {"width": 12, "height": 14}}]
    out, history = pre.preprocess([img], steps)

    assert out[0].shape == (14, 12, 3)
    # fit_im_to_size centers padding: extra 4 cols → 2 left, 2 right; extra 4 rows → 2 top, 2 bot.
    assert history[0]["metadata"][0]["pad"] == [2, 2, 2, 2]


def test_pad_forward_target_size_center_crops_when_smaller():
    pre = Preprocessor()
    img = _hwc_image(20, 16, 3)
    steps = [{"type": "pad", "configuration": {"width": 10, "height": 12}}]
    out, history = pre.preprocess([img], steps)

    assert out[0].shape == (12, 10, 3)
    # Negative entries indicate center-crop strips.
    pad = history[0]["metadata"][0]["pad"]
    assert pad[0] + pad[1] == 10 - 16
    assert pad[2] + pad[3] == 12 - 20


def test_pad_invalid_pad_length_raises():
    pre = Preprocessor()
    img = _hwc_image(10, 10, 3)
    steps = [{"type": "pad", "configuration": {"pad": [1, 2, 3]}}]
    with pytest.raises(ValueError, match="must be \\[L, R, T, B\\]"):
        pre.preprocess([img], steps)


def test_pad_revert_image_restores_original_size():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}}]
    out, history = pre.preprocess([img], steps)

    restored = rec.reconstruct_images(out, history)
    assert restored[0].shape == img.shape
    assert torch.equal(restored[0], img)


def test_pad_revert_coords_subtracts_offset():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}}]
    _out, history = pre.preprocess([img], steps)

    # Box in padded space at [5, 6, 9, 12] → original space [5-2, 6-4, 9-2, 12-4].
    results = _empty_results(
        boxes=[torch.tensor([[5.0, 6.0, 9.0, 12.0]])],
        segments=[[torch.tensor([[5.0, 6.0], [9.0, 6.0], [9.0, 12.0], [5.0, 12.0]])]],
        points=[torch.tensor([[[7.0, 9.0, 1.0]]])],
    )
    reverted = rec.reconstruct_coordinates(results, history)
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[3.0, 2.0, 7.0, 8.0]]))
    assert torch.allclose(reverted["segments"][0][0], torch.tensor([[3.0, 2.0], [7.0, 2.0], [7.0, 8.0], [3.0, 8.0]]))
    assert torch.allclose(reverted["points"][0], torch.tensor([[[5.0, 5.0, 1.0]]]))


def test_pad_apply_coords_adds_offset_round_trip():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}}]
    _out, history = pre.preprocess([img], steps)

    # Original-space box → padded-space.
    original = _empty_results(boxes=[torch.tensor([[3.0, 2.0, 7.0, 8.0]])])
    forward = rec.apply_coordinates(original, history)
    assert torch.allclose(forward["boxes"][0], torch.tensor([[5.0, 6.0, 9.0, 12.0]]))

    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], original["boxes"][0])


def test_pad_empty_results_passthrough():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [1, 1, 1, 1]}}]
    _out, history = pre.preprocess([img], steps)

    reverted = rec.reconstruct_coordinates(_empty_results(), history)
    assert reverted["boxes"][0].shape == (0, 4)


def test_pad_apply_coords_pads_masks():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}}]
    _out, history = pre.preprocess([img], steps)

    # One mask, full original size (1, 10, 8), all ones — should become (1, 19, 13) padded with zeros.
    masks = torch.ones((1, 10, 8))
    forward = rec.apply_coordinates(_empty_results(masks=[masks]), history)
    out_masks = forward["masks"][0]
    assert out_masks.shape == (1, 19, 13)
    # Original content lives at the inner region.
    assert out_masks[0, 4:14, 2:10].eq(1).all()
    # Borders are zero.
    assert out_masks[0, :4].eq(0).all()
    assert out_masks[0, 14:].eq(0).all()
    assert out_masks[0, :, :2].eq(0).all()
    assert out_masks[0, :, 10:].eq(0).all()


def test_pad_revert_coords_crops_masks_round_trip():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}}]
    _out, history = pre.preprocess([img], steps)

    # Mask in padded space (1, 19, 13). Reverting should crop back to (1, 10, 8).
    padded_mask = torch.zeros((1, 19, 13))
    padded_mask[0, 4:14, 2:10] = 1.0
    reverted = rec.reconstruct_coordinates(_empty_results(masks=[padded_mask]), history)
    out_masks = reverted["masks"][0]
    assert out_masks.shape == (1, 10, 8)
    assert out_masks.eq(1).all()


def test_pad_masks_round_trip():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}}]
    _out, history = pre.preprocess([img], steps)

    masks = torch.rand((2, 10, 8))
    forward = rec.apply_coordinates(_empty_results(masks=[masks]), history)
    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.equal(round_trip["masks"][0], masks)


def test_pad_obb_boxes_round_trip():
    """OBB boxes (N, 4, 2) take the corner-flatten path — every corner shifts by (L, T)."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}}]
    _out, history = pre.preprocess([img], steps)

    # OBB in original space; apply should shift each corner by (L=2, T=4).
    obb_orig = torch.tensor([[[3.0, 2.0], [7.0, 2.0], [7.0, 8.0], [3.0, 8.0]]])
    forward = rec.apply_coordinates(_empty_results(boxes=[obb_orig]), history)
    expected_padded = torch.tensor([[[5.0, 6.0], [9.0, 6.0], [9.0, 12.0], [5.0, 12.0]]])
    assert forward["boxes"][0].shape == (1, 4, 2)
    assert torch.allclose(forward["boxes"][0], expected_padded)

    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], obb_orig)


def test_pad_segments_variable_length():
    """Each segment is its own (Mi, 2) tensor; Mi can differ across segments in the same list."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [2, 3, 4, 5]}}]
    _out, history = pre.preprocess([img], steps)

    # In padded space: one 1-point segment and one 3-point segment.
    seg_short = torch.tensor([[5.0, 7.0]])
    seg_long = torch.tensor([[3.0, 6.0], [5.0, 8.0], [7.0, 10.0]])
    results = _empty_results(segments=[[seg_short, seg_long]])
    reverted = rec.reconstruct_coordinates(results, history)
    out_segs = reverted["segments"][0]
    # Revert subtracts (L=2, T=4).
    assert torch.allclose(out_segs[0], torch.tensor([[3.0, 3.0]]))
    assert torch.allclose(out_segs[1], torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_pad_masks_with_negative_pad_crops_and_zero_pads_on_revert():
    # Mixed: crop width strips (-1, -1), pad height (+2, +2).
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(10, 8, 3)
    steps = [{"type": "pad", "configuration": {"pad": [-1, -1, 2, 2]}}]
    _out, history = pre.preprocess([img], steps)
    # Output image: W = 8 - 2 = 6, H = 10 + 4 = 14.
    assert _out[0].shape == (14, 6, 3)

    masks = torch.ones((1, 10, 8))
    forward = rec.apply_coordinates(_empty_results(masks=[masks]), history)
    out_masks = forward["masks"][0]
    assert out_masks.shape == (1, 14, 6)
    # Height padded with zeros at top/bottom; the cropped width strip survived as ones.
    assert out_masks[0, 2:12].eq(1).all()
    assert out_masks[0, :2].eq(0).all()
    assert out_masks[0, 12:].eq(0).all()
