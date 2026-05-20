import logging

import cv2
import numpy as np
import pytest
import torch

import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils

logger = logging.getLogger(__name__)


class Test_resize_image:
    @pytest.mark.parametrize(
        "im, resize_args, expected_shape",
        [
            (
                torch.randint(0, 256, (150, 100, 3), dtype=torch.float32).numpy(),
                {"H": 300},
                (300, 200, 3),
            ),  # Test resize height
            (
                torch.randint(0, 256, (100, 150, 3), dtype=torch.uint8).numpy(),
                {"W": 30},
                (20, 30, 3),
            ),  # Test resize width
            (
                torch.randint(0, 256, (150, 100, 3), dtype=torch.int16).numpy(),
                {"W": 200, "H": 200},
                (200, 200, 3),
            ),  # Test warp
            (torch.rand(150, 100).numpy(), {"W": 50}, (75, 50)),  # Test gray image (2D)
            (
                torch.rand(150, 100, 1).numpy(),
                {"W": 50},
                (75, 50, 1),
            ),  # Test gray image (3D)
        ],
    )
    def test_cases(self, im, resize_args, expected_shape):
        im2 = pipeline_utils.resize_image(im, **resize_args)
        assert im2.shape == expected_shape
        assert im2.dtype == im.dtype

        im2 = pipeline_utils.resize_image(torch.from_numpy(im), **resize_args)
        assert im2.shape == expected_shape
        assert isinstance(im2, torch.Tensor)

        if torch.cuda.is_available():
            tmp = torch.from_numpy(im).cuda()
            im2 = pipeline_utils.resize_image(tmp, **resize_args)
            assert im2.shape == expected_shape
            assert isinstance(im2, torch.Tensor)
            assert im2.is_cuda


class Test_fit_im_to_size:
    def np_func(self, im, W=None, H=None):
        BLACK = (0, 0, 0)
        h_im, w_im = im.shape[:2]
        if W is None:
            W = w_im
        if H is None:
            H = h_im
        # pad or crop width
        if W >= w_im:
            pad_L = (W - w_im) // 2
            pad_R = W - w_im - pad_L
            im = cv2.copyMakeBorder(im, 0, 0, pad_L, pad_R, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad_L = (w_im - W) // 2
            pad_R = w_im - W - pad_L
            im = im[:, pad_L:-pad_R]
            pad_L *= -1
            pad_R *= -1
        # pad or crop height
        if H >= h_im:
            pad_T = (H - h_im) // 2
            pad_B = H - h_im - pad_T
            im = cv2.copyMakeBorder(im, pad_T, pad_B, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad_T = (h_im - H) // 2
            pad_B = h_im - H - pad_T
            im = im[pad_T:-pad_B, :]
            pad_T *= -1
            pad_B *= -1
        return im, pad_L, pad_R, pad_T, pad_B

    @pytest.mark.parametrize(
        "im, wh",
        [
            (torch.rand(100, 100, 3).numpy(), [121, None]),
            (torch.rand(100, 100, 3).numpy(), [None, 131]),
            (torch.rand(100, 100, 3).numpy(), [151, 131]),
            (torch.rand(100, 100, 3).numpy(), [71, 75]),
            (torch.rand(100, 100, 3).numpy(), [None, 89]),
            (torch.rand(100, 100, 3).numpy(), [89, None]),
            (torch.rand(100, 100).numpy(), [89, None]),
            (torch.rand(100, 100, 1).numpy(), [89, None]),
        ],
    )
    def test_cases(self, im, wh):
        W, H = wh
        im1, l1, r1, t1, b1 = self.np_func(im, W, H)

        im2, l2, r2, t2, b2 = pipeline_utils.fit_im_to_size(im, W=W, H=H)
        assert isinstance(im2, np.ndarray)
        assert np.array_equal(im1, np.squeeze(im2))
        assert l1 == l2 and r1 == r2 and t1 == t2 and b1 == b2

        im2, l2, r2, t2, b2 = pipeline_utils.fit_im_to_size(torch.from_numpy(im), W=W, H=H)
        assert isinstance(im2, torch.Tensor)
        assert np.array_equal(im1, np.squeeze(im2.numpy()))
        assert l1 == l2 and r1 == r2 and t1 == t2 and b1 == b2

        if torch.cuda.is_available():
            im = torch.from_numpy(im).cuda()
            im2, l2, r2, t2, b2 = pipeline_utils.fit_im_to_size(im, W=W, H=H)
            assert im2.is_cuda
            assert np.array_equal(im1, np.squeeze(im2.cpu().numpy()))
            assert l1 == l2 and r1 == r2 and t1 == t2 and b1 == b2


def _resize_entry(dst_w, dst_h, src_w, src_h, pad=None):
    """Build a unified `resize` history entry (B=1)."""
    md = {"src_size": [src_w, src_h], "dst_size": [dst_w, dst_h]}
    if pad is not None:
        md["pad"] = list(pad)
    return {"type": "resize", "metadata": [md]}


def _pad_entry(L, R, T, B):
    return {"type": "pad", "metadata": [{"pad": [L, R, T, B]}]}


def _flip_entry(lr, ud, w, h):
    return {"type": "flip", "metadata": [{"lr": lr, "ud": ud, "size": [w, h]}]}


class Test_revert_to_origin:
    """Sanity-check the new unified-schema revert_to_origin against a hand-rolled reference.

    The reference applies each atomic op's inverse (resize unscale, pad subtract, flip
    mirror) in REVERSE history order, matching how the wrapper dispatches through the
    Operation registry.
    """

    def np_func(self, pts, history, verbose=False):
        if isinstance(pts, list):
            pts = np.array(pts, dtype=np.float64)
        pts = pts.astype(np.float64).copy()
        r, c = pts.shape
        for entry in reversed(history):
            t = entry["type"]
            m = entry["metadata"][0]
            if t == "resize":
                src_w, src_h = m["src_size"]
                dst_w, dst_h = m["dst_size"]
                pL, _, pT, _ = m.get("pad", [0, 0, 0, 0])
                # revert: subtract pad offset, then unscale
                pts[:, 0] = (pts[:, 0] - pL) * (src_w / dst_w)
                pts[:, 1] = (pts[:, 1] - pT) * (src_h / dst_h)
                if c == 4:
                    pts[:, 2] = (pts[:, 2] - pL) * (src_w / dst_w)
                    pts[:, 3] = (pts[:, 3] - pT) * (src_h / dst_h)
            elif t == "pad":
                pL, _, pT, _ = m["pad"]
                pts[:, 0] -= pL
                pts[:, 1] -= pT
                if c == 4:
                    pts[:, 2] -= pL
                    pts[:, 3] -= pT
            elif t == "flip":
                lr, ud, w, h = m["lr"], m["ud"], m["size"][0], m["size"][1]
                if lr:
                    pts[:, 0] = w - pts[:, 0]
                    if c == 4:
                        pts[:, 2] = w - pts[:, 2]
                        pts[:, [0, 2]] = pts[:, [2, 0]]
                if ud:
                    pts[:, 1] = h - pts[:, 1]
                    if c == 4:
                        pts[:, 3] = h - pts[:, 3]
                        pts[:, [1, 3]] = pts[:, [3, 1]]
            if verbose:
                logger.info(f"after {t}, pts: {pts}")
        return np.maximum(np.round(pts), 0).astype(np.float32)

    @pytest.mark.parametrize(
        "pts, operations",
        [
            (
                [[10.1, 20.0], [30.2, 40.4], [50.5, 60.4], [70.2, 80.1]],
                [
                    _resize_entry(100, 100, 200, 300),
                    _pad_entry(8, 9, 10, 11),
                ],
            ),
            (
                np.array([[15.3, 25.8], [35.7, 45], [55.6, 65], [75, 85.3]]),
                [
                    _resize_entry(200, 300, 100, 100),
                    _pad_entry(-8, -9, 10, 11),
                ],
            ),
            (
                [[15, 25, 35, 45], [55, 65, 75, 85]],
                [
                    _resize_entry(200, 300, 100, 100),
                    _pad_entry(8, 9, 10, 11),
                ],
            ),
        ],
    )
    def test_cases(self, pts, operations):
        pts1 = self.np_func(pts, operations)

        pts2 = pipeline_utils.revert_to_origin(pts, operations)
        assert np.array_equal(pts1, np.asarray(pts2, dtype=np.float32))

        pts_np = pts if isinstance(pts, np.ndarray) else np.array(pts)
        pts2 = pipeline_utils.revert_to_origin(torch.from_numpy(pts_np.astype(np.float32)), operations)
        assert np.array_equal(pts1, pts2.numpy())

        if torch.cuda.is_available():
            cuda_pts = torch.tensor(pts_np, dtype=torch.float32).cuda()
            pts2 = pipeline_utils.revert_to_origin(cuda_pts, operations)
            assert pts2.is_cuda
            assert np.array_equal(pts1, pts2.cpu().numpy())


class Test_profile_to_3d:
    def np_func(self, profile, resolution, offset):
        if profile.dtype != np.int16:
            raise Exception(f"profile.dtype should be int16, got {profile.dtype}")
        TWO_TO_FIFTEEN = 2**15
        h, w = profile.shape[:2]
        x1, y1 = 0, 0
        x2, y2 = w, h
        mask = profile != -TWO_TO_FIFTEEN
        xx, yy = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
        X = offset[0] + xx * resolution[0]
        Y = offset[1] + yy * resolution[1]
        Z = offset[2] + profile * resolution[2]
        return X, Y, Z, mask

    @pytest.mark.parametrize(
        "profile, resolution, offset",
        [
            (
                torch.randint(-32768, 32767, (100, 100), dtype=torch.int16).numpy(),
                [0.7, 0.96, 0.9],
                [0.1, 0.1, 0.1],
            ),
            (
                torch.randint(-32768, 32767, (90, 100), dtype=torch.int16).numpy(),
                [1, 1, 1],
                [0, 0, 0],
            ),
        ],
    )
    def test_cases(self, profile, resolution, offset):
        x1, y1, z1, m1 = self.np_func(profile, resolution, offset)
        x2, y2, z2, m2 = pipeline_utils.profile_to_3d(profile, resolution, offset)
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
        assert np.array_equal(z1, z2)
        assert np.array_equal(m1, m2)

        x3, y3, z3, m3 = pipeline_utils.profile_to_3d(torch.from_numpy(profile), resolution, offset)
        assert np.array_equal(x1, x3.numpy())
        assert np.array_equal(y1, y3.numpy())
        assert np.array_equal(z1, z3.numpy())
        assert np.array_equal(m1, m3.numpy())

        profile = torch.from_numpy(profile)
        if torch.cuda.is_available():
            x4, y4, z4, m4 = pipeline_utils.profile_to_3d(profile.cuda(), resolution, offset)
            assert x4.is_cuda and y4.is_cuda and z4.is_cuda and m4.is_cuda
            assert np.array_equal(x1, x4.cpu().numpy())
            assert np.array_equal(y1, y4.cpu().numpy())
            assert np.array_equal(z1, z4.cpu().numpy())
            assert np.array_equal(m1, m4.cpu().numpy())

        with pytest.raises(Exception) as info:
            x, y, z, m = pipeline_utils.profile_to_3d(profile.to(torch.uint16), resolution, offset)
        logger.debug(info.value)


class Test_uint16_to_int16:
    def np_func(self, profile):
        if profile.dtype != np.uint16:
            raise Exception(f"dtype should be uint16, got {profile.dtype}")
        TWO_TO_FIFTEEN = 2**15
        return profile.view(np.int16) + np.int16(-TWO_TO_FIFTEEN)

    @pytest.mark.parametrize(
        "profile",
        [
            (torch.randint(0, 65535, (500, 500), dtype=torch.uint16).numpy()),
            (torch.randint(0, 65535, (520, 530), dtype=torch.uint16).numpy()),
        ],
    )
    def test_cases(self, profile):
        p1 = self.np_func(profile)
        p2 = pipeline_utils.uint16_to_int16(profile)
        assert np.array_equal(p1, p2)

        p2 = pipeline_utils.uint16_to_int16(torch.from_numpy(profile))
        assert np.array_equal(p1, p2.numpy())

        profile = torch.from_numpy(profile)
        if torch.cuda.is_available():
            p2 = pipeline_utils.uint16_to_int16(profile.cuda())
            assert p2.is_cuda
            assert np.array_equal(p1, p2.cpu().numpy())

        with pytest.raises(Exception) as info:
            pipeline_utils.uint16_to_int16(profile.to(torch.int16))
        logger.debug(info.value)


class Test_pts_to_3d:
    def np_func(self, pts, profile, resolution, offset):
        if profile.dtype != np.int16:
            raise Exception(f"profile.dtype should be int16, got {profile.dtype}")
        xyz = []
        for pt in pts:
            if len(pt) != 2:
                raise Exception(f"pts should be a list of (x,y) points, got {pt}")
            x, y = map(int, pt)
            nx = offset[0] + x * resolution[0]
            ny = offset[1] + y * resolution[1]
            nz = offset[2] + profile[y][x] * resolution[2]
            xyz += [[nx, ny, nz]]
        return np.array(xyz)

    @pytest.mark.parametrize(
        "pts, profile, resolution, offset",
        [
            (
                np.array([[10.0, 20.3], [30.4, 40.5], [50.1, 60], [70, 80]]),
                torch.randint(-32768, 32767, (100, 100), dtype=torch.int16).numpy(),
                [0.7, 0.96, 0.9],
                [0.1, 0.1, 0.1],
            ),
            (
                np.array([[15, 25], [35, 45], [55, 65], [75, 85]]),
                torch.randint(-32768, 32767, (90, 100), dtype=torch.int16).numpy(),
                [1, 1, 1],
                [0, 0, 0],
            ),
        ],
    )
    def test_cases(self, pts, profile, resolution, offset):
        xyz1 = self.np_func(pts, profile, resolution, offset)

        xyz2 = pipeline_utils.pts_to_3d(pts, profile, resolution, offset)
        assert np.array_equal(xyz1, xyz2)

        xyz2 = pipeline_utils.pts_to_3d(torch.from_numpy(pts), torch.from_numpy(profile), resolution, offset)
        assert np.array_equal(xyz1, xyz2.numpy())

        if torch.cuda.is_available():
            if not isinstance(pts, torch.Tensor):
                pts = torch.from_numpy(pts).cuda()
                profile = torch.from_numpy(profile).cuda()
            xyz3 = pipeline_utils.pts_to_3d(pts, profile, resolution, offset)
            assert xyz3.is_cuda
            assert np.array_equal(xyz1, xyz3.cpu().numpy())

    def test_error_handle(self):
        pts = np.array([[15, 25], [35, 45], [55, 65], [75, 85]])
        profile = torch.randint(-32768, 32767, (90, 100), dtype=torch.int16)
        resolution = [1, 1, 1]
        offset = [0, 0, 0]
        with pytest.raises(Exception) as info:
            pipeline_utils.pts_to_3d(pts, profile.to(torch.uint16), resolution, offset)
        logger.debug(info.value)

        pts2 = np.expand_dims(pts, axis=0)
        with pytest.raises(Exception) as info:
            pipeline_utils.pts_to_3d(pts2, profile.numpy(), resolution, offset)
        logger.debug(info.value)


class Test_apply_operations:
    """Forward complement of revert_to_origin. Applies each op's coord transform in order."""

    def np_func(self, pts, history):
        pts = np.array(pts, dtype=np.float64).copy()
        r, c = pts.shape
        if c not in (2, 4):
            raise Exception(f"pts should be Nx2 or Nx4, got shape: {pts.shape}")
        for entry in history:
            t = entry["type"]
            m = entry["metadata"][0]
            if t == "resize":
                src_w, src_h = m["src_size"]
                dst_w, dst_h = m["dst_size"]
                pL, _, pT, _ = m.get("pad", [0, 0, 0, 0])
                sx, sy = dst_w / src_w, dst_h / src_h
                pts[:, 0] = pts[:, 0] * sx + pL
                pts[:, 1] = pts[:, 1] * sy + pT
                if c == 4:
                    pts[:, 2] = pts[:, 2] * sx + pL
                    pts[:, 3] = pts[:, 3] * sy + pT
            elif t == "pad":
                pL, _, pT, _ = m["pad"]
                pts[:, 0] += pL
                pts[:, 1] += pT
                if c == 4:
                    pts[:, 2] += pL
                    pts[:, 3] += pT
            elif t == "flip":
                lr, ud, w, h = m["lr"], m["ud"], m["size"][0], m["size"][1]
                if lr:
                    pts[:, 0] = w - pts[:, 0]
                    if c == 4:
                        pts[:, 2] = w - pts[:, 2]
                        pts[:, [0, 2]] = pts[:, [2, 0]]
                if ud:
                    pts[:, 1] = h - pts[:, 1]
                    if c == 4:
                        pts[:, 3] = h - pts[:, 3]
                        pts[:, [1, 3]] = pts[:, [3, 1]]
        return np.maximum(np.round(pts), 0).astype(np.float32)

    @pytest.mark.parametrize(
        "pts, operations",
        [
            (
                [[10, 20], [30, 40], [50, 60], [70, 80]],
                [
                    _resize_entry(100, 100, 200, 300),
                    _pad_entry(8, 9, 10, 11),
                    _flip_entry(True, False, 100, 100),
                ],
            ),
            (
                np.array([[15.0, 25.2], [35.1, 45.0], [55.0, 65], [75.5, 85]]),
                [
                    _resize_entry(200, 300, 100, 100),
                    _pad_entry(6, 7, 9, 10),
                ],
            ),
            (
                [[15, 25, 35, 45], [55, 65, 75, 85]],
                [
                    _resize_entry(200, 300, 100, 100),
                    _pad_entry(11, 9, -2, -3),
                    _flip_entry(False, True, 200, 300),
                ],
            ),
        ],
    )
    def test_cases(self, pts, operations):
        pts1 = self.np_func(pts, operations)
        pts2 = pipeline_utils.apply_operations(pts, operations)
        assert np.array_equal(pts1, np.asarray(pts2, dtype=np.float32))

        pts_np = pts if isinstance(pts, np.ndarray) else np.array(pts)
        pts2 = pipeline_utils.apply_operations(torch.from_numpy(pts_np.astype(np.float32)), operations)
        assert np.array_equal(pts1, pts2.numpy())

        if torch.cuda.is_available():
            cuda_pts = torch.tensor(pts_np, dtype=torch.float32).cuda()
            pts2 = pipeline_utils.apply_operations(cuda_pts, operations)
            assert pts2.is_cuda
            assert np.array_equal(pts1, pts2.cpu().numpy())


class Test_revert_mask_to_origin:
    @pytest.mark.parametrize(
        "mask, operations, expected_shape",
        [
            (
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                [_pad_entry(8, 9, 10, 11), _flip_entry(True, False, 100, 100)],
                (79, 83, 3),
            ),
            (
                np.random.randint(0, 255, (90, 100), dtype=np.uint8),
                [_flip_entry(True, False, 100, 90)],
                (90, 100),
            ),
        ],
    )
    def test_cases(self, mask, operations, expected_shape):
        mask2 = pipeline_utils.revert_mask_to_origin(mask, operations)
        assert mask2.shape == expected_shape

        first = operations[0]
        if first["type"] == "flip":
            m = first["metadata"][0]
            mask3 = mask.copy()
            if m["lr"]:
                mask3 = np.flip(mask3, axis=1)
            if m["ud"]:
                mask3 = np.flip(mask3, axis=0)
            assert np.array_equal(mask2, mask3)
