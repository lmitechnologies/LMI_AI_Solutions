import numpy as np
import torch

from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc_image(h, w, c=3):
    return torch.arange(h * w * c, dtype=torch.float32).reshape(h, w, c)


def _empty_results(n=1, **overrides):
    """Per-image results dict with empty defaults for every coord field.

    Pass `boxes=[...]`, `points=[...]`, etc. to override specific fields.
    Anything not overridden stays empty so handlers see no work for it.
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


def test_crop_forward_basic():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[10, 20, 60, 90]]}}]
    out, history = pre.preprocess([img], steps)

    assert len(out) == 1
    assert out[0].shape == (70, 50, 3)
    assert history[0]["type"] == "crop"
    assert history[0]["metadata"][0]["box"] == [10, 20, 60, 90]
    assert history[0]["metadata"][0]["orig_size"] == [80, 100]


def test_crop_clamps_out_of_bounds():
    pre = Preprocessor()
    img = _hwc_image(50, 50, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[-5, -5, 100, 100]]}}]
    out, history = pre.preprocess([img], steps)

    assert out[0].shape == (50, 50, 3)
    assert history[0]["metadata"][0]["box"] == [0, 0, 50, 50]


def test_crop_revert_image_pastes_into_canvas():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.ones((100, 80, 3), dtype=torch.float32)
    steps = [{"type": "crop", "configuration": {"boxes": [[10, 20, 60, 90]]}}]
    out, history = pre.preprocess([img], steps)

    restored = rec.reconstruct_images(out, history)
    assert restored[0].shape == (100, 80, 3)
    # Inside crop region: ones; outside: zeros.
    assert restored[0][20:90, 10:60].eq(1).all()
    assert restored[0][:20].eq(0).all()
    assert restored[0][90:].eq(0).all()
    assert restored[0][:, :10].eq(0).all()
    assert restored[0][:, 60:].eq(0).all()


def test_crop_revert_coords_adds_offset_to_boxes():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[10, 20, 60, 90]]}}]
    _out, history = pre.preprocess([img], steps)

    # Box detected in crop-space, e.g. [5, 5, 15, 15] inside the 50x70 crop.
    results = _empty_results(boxes=[torch.tensor([[5.0, 5.0, 15.0, 15.0]])])
    reverted = rec.reconstruct_coordinates(results, history)
    # Offset by (10, 20).
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[15.0, 25.0, 25.0, 35.0]]))


def test_crop_revert_coords_boxes_obb():
    """OBB boxes have shape (N, 4, 2) — every corner gets offset by the crop origin."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[5, 10, 55, 80]]}}]
    _out, history = pre.preprocess([img], steps)

    obb = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]])
    results = _empty_results(boxes=[obb])
    reverted = rec.reconstruct_coordinates(results, history)
    expected = torch.tensor([[[5.0, 10.0], [15.0, 10.0], [15.0, 15.0], [5.0, 15.0]]])
    assert reverted["boxes"][0].shape == (1, 4, 2)
    assert torch.allclose(reverted["boxes"][0], expected)


def test_crop_revert_coords_segments_variable_length():
    """Each segment is its own (Mi, 2) tensor and Mi can differ across segments."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[15, 20, 60, 90]]}}]
    _out, history = pre.preprocess([img], steps)

    seg_a = torch.tensor([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]])
    seg_b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0], [9.0, 0.0]])
    results = _empty_results(segments=[[seg_a, seg_b]])
    reverted = rec.reconstruct_coordinates(results, history)
    out_segs = reverted["segments"][0]
    assert len(out_segs) == 2
    assert torch.allclose(out_segs[0], seg_a + torch.tensor([15.0, 20.0]))
    assert torch.allclose(out_segs[1], seg_b + torch.tensor([15.0, 20.0]))


def test_crop_revert_coords_points_preserve_visibility():
    """Keypoints are (N, K, 3) with the visibility channel; only xy should move."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[8, 15, 48, 95]]}}]
    _out, history = pre.preprocess([img], steps)

    pts = torch.tensor([[[0.0, 0.0, 2.0], [20.0, 40.0, 1.0], [39.0, 79.0, 0.0]]])
    results = _empty_results(points=[pts])
    reverted = rec.reconstruct_coordinates(results, history)
    expected = torch.tensor([[[8.0, 15.0, 2.0], [28.0, 55.0, 1.0], [47.0, 94.0, 0.0]]])
    assert reverted["points"][0].shape == (1, 3, 3)
    assert torch.allclose(reverted["points"][0], expected)


def test_crop_revert_coords_masks_paste_into_full_canvas():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[12, 25, 72, 95]]}}]
    _out, history = pre.preprocess([img], steps)

    # One mask of shape (1, 70, 60) — full crop.
    mask = torch.ones((1, 70, 60), dtype=torch.float32)
    results = _empty_results(masks=[mask])
    reverted = rec.reconstruct_coordinates(results, history)
    out_mask = reverted["masks"][0]
    assert out_mask.shape == (1, 100, 80)
    assert out_mask[0, 25:95, 12:72].eq(1).all()
    assert out_mask[0, :25].eq(0).all()


def test_crop_apply_coords_forward_inverse_of_revert():
    """apply_coords maps original-space coords forward into crop space; reverting returns the input."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[20, 30, 70, 90]]}}]
    _out, history = pre.preprocess([img], steps)

    orig = _empty_results(
        boxes=[torch.tensor([[25.0, 35.0, 65.0, 75.0]])],
        segments=[[torch.tensor([[22.0, 32.0], [24.0, 36.0]])]],
        points=[torch.tensor([[[40.0, 50.0, 2.0]]])],
    )
    applied = rec.apply_coordinates(orig, history)
    # All xy shifted by (-20, -30); visibility untouched.
    assert torch.allclose(applied["boxes"][0], torch.tensor([[5.0, 5.0, 45.0, 45.0]]))
    assert torch.allclose(applied["segments"][0][0], torch.tensor([[2.0, 2.0], [4.0, 6.0]]))
    assert torch.allclose(applied["points"][0], torch.tensor([[[20.0, 20.0, 2.0]]]))

    round_trip = rec.reconstruct_coordinates(applied, history)
    assert torch.allclose(round_trip["boxes"][0], orig["boxes"][0])
    assert torch.allclose(round_trip["segments"][0][0], orig["segments"][0][0])
    assert torch.allclose(round_trip["points"][0], orig["points"][0])


def test_crop_apply_coords_masks_crops_full_image_masks():
    """apply_coords on masks slices the (N, H, W) full-image mask down to the crop region."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[18, 22, 78, 92]]}}]
    _out, history = pre.preprocess([img], steps)

    full = torch.zeros((1, 100, 80), dtype=torch.float32)
    full[0, 22:92, 18:78] = 1.0
    orig = _empty_results(masks=[full])
    applied = rec.apply_coordinates(orig, history)
    assert applied["masks"][0].shape == (1, 70, 60)
    assert applied["masks"][0].eq(1).all()
