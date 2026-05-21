import logging

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor

logger = logging.getLogger(__name__)


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


def test_rotate_forward_metadata_shape():
    pre = Preprocessor()
    img = _hwc_image(40, 60)
    out, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": 30.0}}])
    assert len(out) == 1
    assert history[0]["type"] == "rotate"
    m = history[0]["metadata"][0]
    assert m["angle"] == 30.0
    assert m["src_size"] == [60, 40]
    # 60*cos30 + 40*sin30 = 51.96 + 20 = 71; 40*cos30 + 60*sin30 = 34.64 + 30 = 64
    assert m["dst_size"] == [71, 64]
    assert out[0].shape[:2] == (64, 71)


def test_rotate_90_grows_canvas_correctly():
    pre = Preprocessor()
    img = _hwc_image(40, 60)
    out, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": 90.0}}])
    # 90° swap: (W, H) -> (H, W)
    assert history[0]["metadata"][0]["dst_size"] == [40, 60]
    assert out[0].shape[:2] == (60, 40)


def test_rotate_zero_is_noop_for_coords():
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(50, 70)
    boxes = torch.tensor([[10.0, 12.0, 30.0, 40.0]])
    results = _empty_results(boxes=[boxes])

    _, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": 0.0}}])
    out_fwd = rec.apply_coordinates(results, history)
    out_rev = rec.reconstruct_coordinates(out_fwd, history)
    assert torch.allclose(out_rev["boxes"][0], boxes, atol=1e-4)


def test_rotate_coord_round_trip_xyxy():
    """Forward then revert returns the same axis-aligned box (idempotent for 0/90 multiples)."""
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(100, 100)
    boxes = torch.tensor([[20.0, 30.0, 60.0, 70.0]])
    results = _empty_results(boxes=[boxes])

    _, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": 37.5}}])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    # xyxy boxes go through axis-aligned-bbox-of-rotated-corners, which is non-idempotent
    # for arbitrary angles — the round-trip bbox encloses the original but may be larger.
    # Assert it contains the original.
    b0, b1 = boxes[0], back["boxes"][0][0]
    assert b1[0] <= b0[0] + 1e-3 and b1[1] <= b0[1] + 1e-3
    assert b1[2] >= b0[2] - 1e-3 and b1[3] >= b0[3] - 1e-3


def test_rotate_coord_round_trip_points():
    """Points are lossless under forward-then-revert (pure 2x2 rotation)."""
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(120, 80)
    pts = torch.tensor([[[10.0, 20.0, 1.0], [50.0, 70.0, 1.0]]])  # (N=1, K=2, 3)
    results = _empty_results(points=[pts])

    _, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": 53.0}}])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    assert torch.allclose(back["points"][0], pts, atol=1e-3)


def test_rotate_apply_coord_matches_image():
    """A point in the original image, mapped via apply_coords, lands at the rotated pixel."""
    pre = Preprocessor()
    rec = Reconstructor()

    # Build a black image with a single white pixel at (x, y)
    H, W = 40, 60
    img = torch.zeros(H, W, 1)
    x, y = 15, 10
    img[y, x, 0] = 1.0

    pts = torch.tensor([[[float(x), float(y), 1.0]]])  # (N=1, K=1, 3)
    results = _empty_results(points=[pts])

    out, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": 22.5}}])
    fwd = rec.apply_coordinates(results, history)

    px, py = fwd["points"][0][0, 0, :2].tolist()
    # Find the brightest pixel in the rotated image; it should be near (px, py).
    rot = out[0][..., 0]
    flat_idx = int(rot.argmax().item())
    rH, rW = rot.shape
    iy, ix = divmod(flat_idx, rW)
    assert abs(ix - px) <= 1.5
    assert abs(iy - py) <= 1.5


@pytest.mark.parametrize("angle", [17.3, 37.5, 60.0, -22.5, 127.0])
def test_rotate_points_round_trip_various_angles(angle):
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(80, 100)
    pts = torch.tensor([[[5.0, 7.0, 1.0], [60.0, 40.0, 2.0], [95.0, 75.0, 0.0]]])
    results = _empty_results(points=[pts])

    _, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": angle}}])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    assert torch.allclose(back["points"][0], pts, atol=1e-3)


def test_rotate_obb_round_trip():
    """OBB (N, 4, 2) is a pure 4-corner rotation — lossless under forward then revert."""
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(80, 100)
    obb = torch.tensor([[[10.0, 20.0], [30.0, 25.0], [28.0, 50.0], [8.0, 45.0]]])  # (1, 4, 2)
    results = _empty_results(boxes=[obb])

    _, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": 47.0}}])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    assert back["boxes"][0].shape == obb.shape
    assert torch.allclose(back["boxes"][0], obb, atol=1e-3)


def test_rotate_segments_round_trip():
    """Variable-length segments rotate point-by-point and revert losslessly."""
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(60, 80)
    seg_a = torch.tensor([[10.0, 12.0], [40.0, 18.0], [42.0, 50.0], [11.0, 48.0]])
    seg_b = torch.tensor([[5.0, 5.0], [55.0, 55.0], [60.0, 30.0]])
    results = _empty_results(segments=[[seg_a, seg_b]])

    _, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": -33.7}}])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    assert torch.allclose(back["segments"][0][0], seg_a, atol=1e-3)
    assert torch.allclose(back["segments"][0][1], seg_b, atol=1e-3)


def test_rotate_image_matches_cv2_recipe():
    """The docstring claims metadata reproduces the rotation via cv2.warpAffine — verify it."""
    cv2 = pytest.importorskip("cv2")
    pre = Preprocessor()
    H, W = 50, 70
    img_np = (np.arange(H * W * 3).reshape(H, W, 3) % 256).astype(np.float32)
    img = torch.from_numpy(img_np)

    out, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": 37.5}}])
    m = history[0]["metadata"][0]
    nW, nH = m["dst_size"]

    M = cv2.getRotationMatrix2D((W // 2, H // 2), -m["angle"], 1.0)
    M[0, 2] += nW / 2 - W // 2
    M[1, 2] += nH / 2 - H // 2
    expected = cv2.warpAffine(img_np, M, (nW, nH))

    # align_corners=False matches cv2's pixel-edge boundary convention; residual diff is
    # cv2's 28-bit fixed-point INTER_LINEAR vs torch's fp32 bilinear, evenly distributed.
    ours = out[0].cpu().numpy()
    diff = np.abs(ours - expected)
    logger.info(f"max diff={diff.max():.2f}  mean={diff.mean():.3f}  (>1.0)={(diff > 1).sum()}")
    assert diff.mean() < 0.5
    assert diff.max() < 6.0


def test_rotate_revert_image_returns_src_size():
    """revert_images must produce shapes matching src_size — content fidelity is lossy by design."""
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(20, 30)
    out, history = pre.preprocess([img], [{"type": "rotate", "configuration": {"angle": 37.5}}])
    back = rec.reconstruct_images(out, history)
    assert back[0].shape == img.shape
