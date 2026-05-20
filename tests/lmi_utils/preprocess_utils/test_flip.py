import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc_image(h, w, c=3):
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


def _make_results(boxes):
    """Results dict with boxes plus derived corner-polygon segments and centroid points."""
    segs = [torch.stack([b[[0, 1]], b[[2, 1]], b[[2, 3]], b[[0, 3]]]) for b in boxes]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    pts = torch.cat([torch.stack([cx, cy], dim=-1).unsqueeze(1), torch.ones(len(boxes), 1, 1)], dim=-1)
    return _empty_results(boxes=[boxes], segments=[segs], points=[pts])


@pytest.mark.parametrize(
    "lr, ud",
    [(True, False), (False, True), (True, True), (False, False)],
)
def test_flip_forward_pixels(lr, ud):
    pre = Preprocessor()
    img = _hwc_image(4, 5, 1)
    steps = [{"type": "flip", "configuration": {"lr": lr, "ud": ud}}]

    out, history = pre.preprocess([img], steps)
    expected = img
    if lr:
        expected = torch.flip(expected, dims=[1])
    if ud:
        expected = torch.flip(expected, dims=[0])
    assert torch.equal(out[0], expected)
    assert history[0]["metadata"][0] == {"lr": lr, "ud": ud, "size": [5, 4]}


def test_flip_image_revert_is_involutive():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(8, 6, 3)
    steps = [{"type": "flip", "configuration": {"lr": True, "ud": True}}]
    out, history = pre.preprocess([img], steps)

    restored = rec.reconstruct_images(out, history)
    assert torch.equal(restored[0], img)


def test_flip_revert_coords_xyxy_lr():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "flip", "configuration": {"lr": True}}]
    _out, history = pre.preprocess([img], steps)

    # In flipped space, box [10, 20, 30, 40] should map back to [50, 20, 70, 40]
    # (w=80; x' = 80 - x, and the corner swap keeps x1<x2).
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    results = _make_results(boxes)
    reverted = rec.reconstruct_coordinates(results, history)
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[50.0, 20.0, 70.0, 40.0]]))
    # Segments are corner polygons — they get mirrored point-wise, not corner-swapped.
    expected_corners = torch.tensor([[70.0, 20.0], [50.0, 20.0], [50.0, 40.0], [70.0, 40.0]])
    assert torch.allclose(reverted["segments"][0][0], expected_corners)
    # Point centroid (20, 30, 1) → (60, 30, 1).
    assert torch.allclose(reverted["points"][0][0, 0], torch.tensor([60.0, 30.0, 1.0]))


def test_flip_revert_coords_xyxy_ud():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "flip", "configuration": {"ud": True}}]
    _out, history = pre.preprocess([img], steps)

    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    reverted = rec.reconstruct_coordinates(_make_results(boxes), history)
    # h=100; y' = 100 - y, with corner swap.
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[10.0, 60.0, 30.0, 80.0]]))


def test_flip_revert_obb_boxes():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "flip", "configuration": {"lr": True}}]
    _out, history = pre.preprocess([img], steps)

    obb = torch.tensor([[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]])
    reverted = rec.reconstruct_coordinates(_empty_results(boxes=[obb]), history)
    expected = torch.tensor([[[70.0, 20.0], [50.0, 20.0], [50.0, 40.0], [70.0, 40.0]]])
    assert torch.allclose(reverted["boxes"][0], expected)


def test_flip_revert_masks():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(4, 5, 1)
    steps = [{"type": "flip", "configuration": {"lr": True, "ud": True}}]
    _out, history = pre.preprocess([img], steps)

    mask = torch.zeros((1, 4, 5), dtype=torch.float32)
    mask[0, 0, 0] = 1.0  # marker in top-left
    reverted = rec.reconstruct_coordinates(_empty_results(masks=[mask]), history)
    assert reverted["masks"][0][0, -1, -1] == 1.0
    assert reverted["masks"][0][0, 0, 0] == 0.0


@pytest.mark.parametrize(
    "lr, ud, expected",
    [
        (False, True, [[[10.0, 80.0], [30.0, 80.0], [30.0, 60.0], [10.0, 60.0]]]),
        (True, True, [[[70.0, 80.0], [50.0, 80.0], [50.0, 60.0], [70.0, 60.0]]]),
    ],
)
def test_flip_revert_obb_ud_and_both(lr, ud, expected):
    """OBB revert for ud-only and lr+ud — covers the box_fn 3D path beyond the lr-only case."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "flip", "configuration": {"lr": lr, "ud": ud}}]
    _out, history = pre.preprocess([img], steps)

    obb = torch.tensor([[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]])
    reverted = rec.reconstruct_coordinates(_empty_results(boxes=[obb]), history)
    assert torch.allclose(reverted["boxes"][0], torch.tensor(expected))


def test_flip_obb_apply_coords_round_trip():
    """apply_coords on OBB should be the inverse of revert (flip is involutive)."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "flip", "configuration": {"lr": True, "ud": True}}]
    _out, history = pre.preprocess([img], steps)

    obb = torch.tensor([[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]])
    forward = rec.apply_coordinates(_empty_results(boxes=[obb]), history)
    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], obb)


def test_flip_revert_segments_variable_length():
    """Each segment is its own (Mi, 2) tensor; Mi can differ across segments."""
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "flip", "configuration": {"lr": True}}]
    _out, history = pre.preprocess([img], steps)

    # In flipped space — revert with x' = 80 - x.
    seg_short = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    seg_long = torch.tensor([[5.0, 5.0], [70.0, 15.0]])
    results = _empty_results(segments=[[seg_short, seg_long]])
    reverted = rec.reconstruct_coordinates(results, history)
    out_segs = reverted["segments"][0]
    assert len(out_segs) == 2
    assert torch.allclose(out_segs[0], torch.tensor([[70.0, 20.0], [50.0, 40.0], [30.0, 60.0]]))
    assert torch.allclose(out_segs[1], torch.tensor([[75.0, 5.0], [10.0, 15.0]]))


def test_flip_apply_coords_is_inverse_of_revert():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "flip", "configuration": {"lr": True, "ud": True}}]
    _out, history = pre.preprocess([img], steps)

    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    original = _make_results(boxes)
    forward = rec.apply_coordinates(original, history)
    round_trip = rec.reconstruct_coordinates(forward, history)
    assert torch.allclose(round_trip["boxes"][0], boxes)
    assert torch.allclose(round_trip["points"][0], original["points"][0])
