import numpy as np
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc_image(h, w, c=3):
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


def test_crop_forward_basic():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    out, history = pre.preprocess([img], [steps.crop(boxes=[[10, 20, 60, 90]])])

    assert len(out) == 1
    assert out[0].shape == (70, 50, 3)
    assert history[0].boxes[0] == [10, 20, 60, 90]
    assert history[0].orig_sizes[0] == [80, 100]


def test_crop_clamps_out_of_bounds():
    pre = Preprocessor()
    img = _hwc_image(50, 50, 3)
    out, history = pre.preprocess([img], [steps.crop(boxes=[[-5, -5, 100, 100]])])

    assert out[0].shape == (50, 50, 3)
    assert history[0].boxes[0] == [0, 0, 50, 50]


def test_crop_revert_image_pastes_into_canvas():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.ones((100, 80, 3), dtype=torch.float32)
    out, history = pre.preprocess([img], [steps.crop(boxes=[[10, 20, 60, 90]])])

    restored = rec.reconstruct_images(out, history)
    assert restored[0].shape == (100, 80, 3)
    assert restored[0][20:90, 10:60].eq(1).all()
    assert restored[0][:20].eq(0).all()
    assert restored[0][90:].eq(0).all()
    assert restored[0][:, :10].eq(0).all()
    assert restored[0][:, 60:].eq(0).all()


def test_crop_revert_coords_adds_offset_to_boxes():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    _out, history = pre.preprocess([img], [steps.crop(boxes=[[10, 20, 60, 90]])])

    results = _empty_results(boxes=[torch.tensor([[5.0, 5.0, 15.0, 15.0]])])
    reverted = rec.reconstruct_coordinates(results, history)
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[15.0, 25.0, 25.0, 35.0]]))


def test_crop_revert_coords_boxes_obb():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    _out, history = pre.preprocess([img], [steps.crop(boxes=[[5, 10, 55, 80]])])

    obb = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]])
    results = _empty_results(boxes=[obb])
    reverted = rec.reconstruct_coordinates(results, history)
    expected = torch.tensor([[[5.0, 10.0], [15.0, 10.0], [15.0, 15.0], [5.0, 15.0]]])
    assert reverted["boxes"][0].shape == (1, 4, 2)
    assert torch.allclose(reverted["boxes"][0], expected)


def test_crop_revert_coords_segments_variable_length():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    _out, history = pre.preprocess([img], [steps.crop(boxes=[[15, 20, 60, 90]])])

    seg_a = torch.tensor([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]])
    seg_b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0], [9.0, 0.0]])
    results = _empty_results(segments=[[seg_a, seg_b]])
    reverted = rec.reconstruct_coordinates(results, history)
    out_segs = reverted["segments"][0]
    assert len(out_segs) == 2
    assert torch.allclose(out_segs[0], seg_a + torch.tensor([15.0, 20.0]))
    assert torch.allclose(out_segs[1], seg_b + torch.tensor([15.0, 20.0]))


def test_crop_revert_coords_points_preserve_visibility():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    _out, history = pre.preprocess([img], [steps.crop(boxes=[[8, 15, 48, 95]])])

    pts = torch.tensor([[[0.0, 0.0, 2.0], [20.0, 40.0, 1.0], [39.0, 79.0, 0.0]]])
    results = _empty_results(points=[pts])
    reverted = rec.reconstruct_coordinates(results, history)
    expected = torch.tensor([[[8.0, 15.0, 2.0], [28.0, 55.0, 1.0], [47.0, 94.0, 0.0]]])
    assert reverted["points"][0].shape == (1, 3, 3)
    assert torch.allclose(reverted["points"][0], expected)


def test_crop_revert_coords_masks_paste_into_full_canvas():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    _out, history = pre.preprocess([img], [steps.crop(boxes=[[12, 25, 72, 95]])])

    mask = torch.ones((1, 70, 60), dtype=torch.float32)
    results = _empty_results(masks=[mask])
    reverted = rec.reconstruct_coordinates(results, history)
    out_mask = reverted["masks"][0]
    assert out_mask.shape == (1, 100, 80)
    assert out_mask[0, 25:95, 12:72].eq(1).all()
    assert out_mask[0, :25].eq(0).all()


def test_crop_apply_coords_forward_inverse_of_revert():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    _out, history = pre.preprocess([img], [steps.crop(boxes=[[20, 30, 70, 90]])])

    orig = _empty_results(
        boxes=[torch.tensor([[25.0, 35.0, 65.0, 75.0]])],
        segments=[[torch.tensor([[22.0, 32.0], [24.0, 36.0]])]],
        points=[torch.tensor([[[40.0, 50.0, 2.0]]])],
    )
    applied = rec.apply_coordinates(orig, history)
    assert torch.allclose(applied["boxes"][0], torch.tensor([[5.0, 5.0, 45.0, 45.0]]))
    assert torch.allclose(applied["segments"][0][0], torch.tensor([[2.0, 2.0], [4.0, 6.0]]))
    assert torch.allclose(applied["points"][0], torch.tensor([[[20.0, 20.0, 2.0]]]))

    round_trip = rec.reconstruct_coordinates(applied, history)
    assert torch.allclose(round_trip["boxes"][0], orig["boxes"][0])
    assert torch.allclose(round_trip["segments"][0][0], orig["segments"][0][0])
    assert torch.allclose(round_trip["points"][0], orig["points"][0])


def test_crop_apply_coords_masks_crops_full_image_masks():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    _out, history = pre.preprocess([img], [steps.crop(boxes=[[18, 22, 78, 92]])])

    full = torch.zeros((1, 100, 80), dtype=torch.float32)
    full[0, 22:92, 18:78] = 1.0
    orig = _empty_results(masks=[full])
    applied = rec.apply_coordinates(orig, history)
    assert applied["masks"][0].shape == (1, 70, 60)
    assert applied["masks"][0].eq(1).all()


def test_crop_manual_revert_via_steps_namespace():
    """Manual inverse without history — build a CropMeta directly via steps.revert_crop."""
    rec = Reconstructor()
    results = _empty_results(boxes=[torch.tensor([[5.0, 5.0, 15.0, 15.0]])])
    history = [steps.revert_crop(boxes=[[10, 20, 60, 90]], orig_sizes=[[80, 100]])]
    reverted = rec.reconstruct_coordinates(results, history)
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[15.0, 25.0, 25.0, 35.0]]))
