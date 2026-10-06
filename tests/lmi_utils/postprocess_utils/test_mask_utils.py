import cv2
import numpy as np
import pytest
import torch

from lmi_utils.postprocess_utils.mask_utils import mask_to_obb, mask_to_polygon


def _two_pieces():
    mask = np.zeros((10, 30), dtype=np.uint8)
    mask[2:8, 0:5] = 1
    mask[2:8, 20:30] = 1
    return mask


def test_hull_wraps_every_piece():
    hull = mask_to_polygon(_two_pieces())
    assert hull.dtype == np.float64
    assert sorted(map(tuple, hull.tolist())) == [(0, 2), (0, 7), (29, 2), (29, 7)]


def test_torch_mask_matches_numpy_mask():
    mask = _two_pieces() * 3
    np.testing.assert_array_equal(mask_to_polygon(torch.from_numpy(mask), value=3), mask_to_polygon(mask, value=3))


@pytest.mark.parametrize("seed", range(20))
def test_box_is_the_smallest_rotated_box(seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.fillPoly(mask, [rng.integers(0, 200, (int(rng.integers(3, 8)), 2)).astype(np.int32)], 1)
    (_, _, w, h, _), _ = mask_to_obb(mask)
    (_, (w2, h2), _) = cv2.minAreaRect(mask_to_polygon(mask).astype(np.float32))
    assert w * h == pytest.approx(w2 * h2, rel=1e-4)
