import logging
import math

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
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
    out, history = pre.preprocess([img], [steps.rotate(angle=30.0)])
    assert len(out) == 1
    m = history[0]
    assert m.angles[0] == 30.0
    assert m.src_sizes[0] == [60, 40]
    assert m.dst_sizes[0] == [72, 65]  # rounded up from 71.96 x 64.64
    assert out[0].shape[:2] == (65, 72)


def test_rotate_90_grows_canvas_correctly():
    pre = Preprocessor()
    img = _hwc_image(40, 60)
    out, history = pre.preprocess([img], [steps.rotate(angle=90.0)])
    assert history[0].dst_sizes[0] == [40, 60]
    assert out[0].shape[:2] == (60, 40)


def test_rotate_zero_is_noop_for_coords():
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(50, 70)
    boxes = torch.tensor([[10.0, 12.0, 30.0, 40.0]])
    results = _empty_results(boxes=[boxes])

    _, history = pre.preprocess([img], [steps.rotate(angle=0.0)])
    out_fwd = rec.apply_coordinates(results, history)
    out_rev = rec.reconstruct_coordinates(out_fwd, history)
    assert torch.allclose(out_rev["boxes"][0], boxes, atol=1e-4)


def test_rotate_coord_round_trip_xyxy():
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(100, 100)
    boxes = torch.tensor([[20.0, 30.0, 60.0, 70.0]])
    results = _empty_results(boxes=[boxes])

    _, history = pre.preprocess([img], [steps.rotate(angle=37.5)])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    b0, b1 = boxes[0], back["boxes"][0][0]
    assert b1[0] <= b0[0] + 1e-3 and b1[1] <= b0[1] + 1e-3
    assert b1[2] >= b0[2] - 1e-3 and b1[3] >= b0[3] - 1e-3


def test_rotate_coord_round_trip_points():
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(120, 80)
    pts = torch.tensor([[[10.0, 20.0, 1.0], [50.0, 70.0, 1.0]]])
    results = _empty_results(points=[pts])

    _, history = pre.preprocess([img], [steps.rotate(angle=53.0)])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    assert torch.allclose(back["points"][0], pts, atol=1e-3)


def test_rotate_apply_coord_matches_image():
    pre = Preprocessor()
    rec = Reconstructor()

    H, W = 40, 60
    img = torch.zeros(H, W, 1)
    x, y = 15, 10
    img[y, x, 0] = 1.0

    pts = torch.tensor([[[float(x), float(y), 1.0]]])
    results = _empty_results(points=[pts])

    out, history = pre.preprocess([img], [steps.rotate(angle=22.5)])
    fwd = rec.apply_coordinates(results, history)

    px, py = fwd["points"][0][0, 0, :2].tolist()
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

    _, history = pre.preprocess([img], [steps.rotate(angle=angle)])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    assert torch.allclose(back["points"][0], pts, atol=1e-3)


def test_rotate_obb_round_trip():
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(80, 100)
    obb = torch.tensor([[[10.0, 20.0], [30.0, 25.0], [28.0, 50.0], [8.0, 45.0]]])
    results = _empty_results(boxes=[obb])

    _, history = pre.preprocess([img], [steps.rotate(angle=47.0)])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    assert back["boxes"][0].shape == obb.shape
    assert torch.allclose(back["boxes"][0], obb, atol=1e-3)


def test_rotate_segments_round_trip():
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(60, 80)
    seg_a = torch.tensor([[10.0, 12.0], [40.0, 18.0], [42.0, 50.0], [11.0, 48.0]])
    seg_b = torch.tensor([[5.0, 5.0], [55.0, 55.0], [60.0, 30.0]])
    results = _empty_results(segments=[[seg_a, seg_b]])

    _, history = pre.preprocess([img], [steps.rotate(angle=-33.7)])
    fwd = rec.apply_coordinates(results, history)
    back = rec.reconstruct_coordinates(fwd, history)
    assert torch.allclose(back["segments"][0][0], seg_a, atol=1e-3)
    assert torch.allclose(back["segments"][0][1], seg_b, atol=1e-3)


@pytest.mark.parametrize("angle", [7.0, 37.5, -62.0, 123.4])
def test_rotate_image_matches_cv2_recipe(angle):
    """Cross-check against cv2, on the interior and at cv2 4.x's precision.

    cv2 4.x quantizes sample coords to 1/32 px, putting its interior error at local_gradient/64
    (0.047 here); 5.x resamples in float and lands near 3e-5. The tolerance covers either. Sub-pixel
    error also decides inside-vs-outside at the content edge, so both versions can differ by a whole
    pixel value in the outer fringe — hence the mask. Placement is pinned by the analytic tests.
    """
    cv2 = pytest.importorskip("cv2")
    pre = Preprocessor()
    H, W = 50, 70
    yy, xx = np.mgrid[0:H, 0:W]
    img_np = np.stack([xx + yy, 2.0 * xx, 255 - yy * 3.0], axis=-1).astype(np.float32)  # gradient 3/px, no cliffs

    out, history = pre.preprocess([torch.from_numpy(img_np)], [steps.rotate(angle=angle)])
    nW, nH = history[0].dst_sizes[0]

    # Centers are (S-1)/2, not the S//2 that cv2 examples usually pass; see _affine.
    M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), -angle, 1.0)
    M[0, 2] += (nW - 1) / 2 - (W - 1) / 2
    M[1, 2] += (nH - 1) / 2 - (H - 1) / 2
    expected = cv2.warpAffine(img_np, M, (nW, nH))
    coverage = cv2.warpAffine(np.ones((H, W), np.float32), M, (nW, nH))
    interior = cv2.erode(coverage, np.ones((3, 3), np.uint8)) > 0.999

    diff = np.abs(out[0].cpu().numpy() - expected).max(axis=-1)
    logger.info(f"a={angle}: interior max={diff[interior].max():.4f}  full-canvas max={diff.max():.2f}")
    assert interior.sum() > 0.5 * H * W
    assert diff[interior].max() < 0.1  # 3/64 = 0.047 from cv2's quantization, plus headroom


@pytest.mark.parametrize("angle", [7.0, 37.5, -62.0, 123.4])
def test_rotate_geometry_matches_analytic_ramp(angle):
    """Placement, against a closed-form reference and to float32 precision.

    Bilinear reproduces a linear function exactly, so a ramp's expected output is analytic — no
    reference resampler, no error floor. The inverse map here is written longhand, so unlike the
    other accuracy tests this shares no code with ``_affine`` and does check the geometry.
    """
    H, W = 61, 83
    alpha, beta = 0.37, -0.21
    gradient = math.hypot(alpha, beta)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    img = (alpha * xs + beta * ys).astype(np.float32)

    out, history = Preprocessor().preprocess([torch.from_numpy(img)], [steps.rotate(angle=angle)])
    nW, nH = history[0].dst_sizes[0]

    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out_y, out_x = np.mgrid[0:nH, 0:nW].astype(np.float64)
    u, v = out_x - (nW - 1) / 2, out_y - (nH - 1) / 2  # undo the canvas re-centring
    src_x = u * cos_t + v * sin_t + (W - 1) / 2  # rotate back by -angle, restore the src centre
    src_y = -u * sin_t + v * cos_t + (H - 1) / 2

    margin = 1e-3  # a tap landing within float-eps of the border reads the zero fill, which the ramp does not model
    interior = (src_x >= margin) & (src_x <= W - 1 - margin) & (src_y >= margin) & (src_y <= H - 1 - margin)
    err = np.abs(out[0].cpu().numpy() - (alpha * src_x + beta * src_y))[interior] / gradient
    logger.info(f"a={angle}: placement max={err.max():.6f} px  mean={err.mean():.6f} px")
    assert interior.sum() > 0.5 * H * W
    assert err.max() < 0.001  # float32 grid resolution: ~1e-5 px at this size (it grows to ~1e-3 px at 4000x3000)


@pytest.mark.parametrize("angle", [7.0, 37.5, -62.0])
def test_rotate_sampling_kernel_is_bilinear(angle):
    """The interpolation kernel itself, against a float64 bilinear reference.

    Needs non-linear content: every sane kernel reproduces a ramp exactly, so the analytic test
    above cannot tell bilinear from bicubic. On noise this separates them by ~70 intensity levels.
    Shares the production affine, so it pins the kernel, not the placement.
    """
    from lmi_utils.preprocess_utils.ops.rotate import _affine

    H, W = 50, 70
    img = (np.random.default_rng(7).random((H, W, 3)) * 255).astype(np.float32)

    out, history = Preprocessor().preprocess([torch.from_numpy(img)], [steps.rotate(angle=angle)])
    nW, nH = history[0].dst_sizes[0]

    a, b, tx, c, d, ty = _affine(nW, nH, W, H, -angle)  # the dst->src map
    ys, xs = np.mgrid[0:nH, 0:nW].astype(np.float64)
    sx, sy = a * xs + b * ys + tx, c * xs + d * ys + ty
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    padded = np.zeros((H + 2, W + 2, 3))
    padded[1 : H + 1, 1 : W + 1] = img  # 1px zero ring, so samples just outside read as the zero fill

    def tap(X, Y):
        inside = (X >= -1) & (X <= W) & (Y >= -1) & (Y <= H)
        return np.where(inside[..., None], padded[np.clip(Y + 1, 0, H + 1), np.clip(X + 1, 0, W + 1)], 0.0)

    top = tap(x0, y0) * (1 - fx) + tap(x0 + 1, y0) * fx
    bottom = tap(x0, y0 + 1) * (1 - fx) + tap(x0 + 1, y0 + 1) * fx
    expected = top * (1 - fy) + bottom * fy

    diff = np.abs(out[0].cpu().numpy() - expected)
    logger.info(f"a={angle}: max={diff.max():.5f} mean={diff.mean():.6f}")
    assert diff.max() < 0.01  # float32 rounding only


@pytest.mark.parametrize("angle, k", [(90.0, -1), (180.0, 2), (270.0, 1), (-90.0, 1), (360.0, 0)])
def test_rotate_right_angles_are_exact(angle, k):
    """Multiples of 90 must be a lossless transpose, not a resample."""
    pre = Preprocessor()
    img = _hwc_image(50, 70)
    out, history = pre.preprocess([img], [steps.rotate(angle=angle)])
    expected = torch.rot90(img, k, dims=(0, 1))
    assert out[0].shape == expected.shape
    assert torch.equal(out[0], expected)
    nW, nH = history[0].dst_sizes[0]
    assert (nH, nW) == expected.shape[:2]


def test_rotate_zero_is_exact_image_noop():
    pre = Preprocessor()
    img = _hwc_image(40, 60)
    out, _ = pre.preprocess([img], [steps.rotate(angle=0.0)])
    assert torch.equal(out[0], img)


@pytest.mark.parametrize("angle", [13.0, 30.0, 90.0, 180.0, -47.0])
def test_rotate_canvas_holds_all_corners(angle):
    """The expanded canvas must contain every source corner, or content is silently clipped."""
    pre = Preprocessor()
    H, W = 50, 70
    _, history = pre.preprocess([_hwc_image(H, W)], [steps.rotate(angle=angle)])
    nW, nH = history[0].dst_sizes[0]
    corners = torch.tensor([[[0.0, 0.0, 1.0], [W - 1.0, 0.0, 1.0], [0.0, H - 1.0, 1.0], [W - 1.0, H - 1.0, 1.0]]])
    moved = Reconstructor().apply_coordinates(_empty_results(points=[corners]), history)["points"][0][0, :, :2]
    assert moved[:, 0].min() >= -0.5 and moved[:, 0].max() <= nW - 0.5
    assert moved[:, 1].min() >= -0.5 and moved[:, 1].max() <= nH - 0.5


@pytest.mark.parametrize("angle", [30.0, 90.0, 180.0])
def test_rotate_preserves_content_area(angle):
    pre = Preprocessor()
    img = torch.full((100, 140, 1), 255.0)
    out, _ = pre.preprocess([img], [steps.rotate(angle=angle)])
    assert (out[0] > 127).sum().item() == 100 * 140


def test_rotate_coord_count_mismatch_raises():
    from lmi_utils.preprocess_utils.ops import RotateMeta

    rec = Reconstructor()
    meta = RotateMeta(angles=[20.0], src_sizes=[[50, 50]], dst_sizes=[[69, 69]])
    results = _empty_results(2, boxes=[torch.tensor([[1.0, 2.0, 3.0, 4.0]])] * 2)
    for fn in (rec.apply_coordinates, rec.reconstruct_coordinates):
        with pytest.raises(ValueError, match="result count"):
            fn(results, [meta])


def test_rotate_revert_image_returns_src_size():
    pre = Preprocessor()
    rec = Reconstructor()
    img = _hwc_image(20, 30)
    out, history = pre.preprocess([img], [steps.rotate(angle=37.5)])
    back = rec.reconstruct_images(out, history)
    assert back[0].shape == img.shape


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_rotate_channels_warp_independently(channels):
    """The affine warp puts channels on grid_sample's batch dim; each must still warp on its own."""
    pre = Preprocessor()
    H, W = 40, 60
    img = torch.zeros(H, W, channels)
    for c in range(channels):  # one distinct lit pixel per channel
        img[10 + c, 15 + 2 * c, c] = 1.0

    out, _ = pre.preprocess([img], [steps.rotate(angle=22.5)])
    rotated = out[0]
    assert rotated.shape[2] == channels
    for c in range(channels):
        peak = torch.nonzero(rotated[..., c] > 0.2)
        assert len(peak) >= 1
        # a channel's content must not bleed into its neighbours
        for other in range(channels):
            if other != c:
                assert rotated[peak[0][0], peak[0][1], other] == pytest.approx(0.0, abs=1e-4)


def test_rotate_hwc_matches_per_channel_hw_warp():
    """Warping HWC in one call must equal warping each channel separately as HW."""
    pre = Preprocessor()
    H, W = 37, 53
    torch.manual_seed(0)
    img = torch.rand(H, W, 3)

    together, _ = pre.preprocess([img], [steps.rotate(angle=31.0)])
    for c in range(3):
        alone, _ = pre.preprocess([img[..., c].contiguous()], [steps.rotate(angle=31.0)])
        assert torch.equal(together[0][..., c], alone[0])
