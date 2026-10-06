import cv2
import numpy as np
import torch

from lmi_utils.postprocess_utils.mask_segments import masks_to_segments


def _two_pieces():
    mask = np.zeros((1, 10, 30), dtype=np.uint8)
    mask[0, 2:8, 0:5] = 1
    mask[0, 2:8, 20:30] = 1
    return mask


def _fill(segment, shape):
    out = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(out, [segment.round().astype(np.int32).reshape(-1, 1, 2)], 1)
    return out


def test_keeps_the_largest_piece():
    (seg,) = masks_to_segments(torch.from_numpy(_two_pieces()).float())
    assert seg.dtype == np.float32
    assert seg[:, 0].min() == 20 and seg[:, 0].max() == 29
    assert seg[:, 1].min() == 2 and seg[:, 1].max() == 7


def test_holes_are_dropped():
    mask = np.zeros((1, 20, 20), dtype=np.uint8)
    mask[0, 2:18, 2:18] = 1
    mask[0, 8:12, 8:12] = 0
    (seg,) = masks_to_segments(mask)
    assert _fill(seg, (20, 20))[8:12, 8:12].all()


def test_empty_mask_gives_an_empty_segment():
    (seg,) = masks_to_segments(np.zeros((1, 5, 5), dtype=bool))
    assert seg.shape == (0, 2)
