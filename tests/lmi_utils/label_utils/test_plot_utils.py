"""Tile-grid overlay: parity coloring and the inset that keeps shared seams visible."""

import numpy as np
import torch

from lmi_utils.image_utils.tiler import Tiler
from lmi_utils.label_utils.plot_utils import plot_boxes_by_group, plot_tile_grid


def _grid(im_hw=(1024, 1024), tile=512, stride=256):
    tiler = Tiler([tile, tile], [stride, stride])
    tiler.tile(torch.zeros(1, 3, *im_hw))
    return tiler.tile_boxes()


def _colors_drawn(boxes, im_hw):
    img = np.zeros((*im_hw, 3), np.uint8)
    plot_tile_grid(boxes, img, line_thickness=1, inset=2)
    return {tuple(c) for c in img.reshape(-1, 3).tolist()} - {(0, 0, 0)}


def test_overlapping_neighbours_get_different_colors():
    img = np.zeros((1024, 1024, 3), np.uint8)
    plot_tile_grid(_grid(), img, line_thickness=1, inset=2)
    colors = {tuple(c) for c in img.reshape(-1, 3).tolist()} - {(0, 0, 0)}
    assert len(colors) == 4


def test_a_shared_seam_is_drawn_as_two_lines():
    img = np.zeros((1024, 1024, 3), np.uint8)
    plot_tile_grid(_grid(), img, line_thickness=1, inset=2)
    lit = img.any(axis=2).sum(axis=1)
    # the tiles at y=256 are column groups 0 and 1, so their top edges land 0 and 2 px down, not on top of each other
    assert lit[256] > 100 and lit[258] > 100
    assert lit[257] < 20


def test_no_boxes_leaves_the_image_alone():
    img = np.zeros((8, 8, 3), np.uint8)
    plot_tile_grid(np.zeros((0, 4)), img)
    assert not img.any()


def test_a_third_of_a_tile_stride_needs_nine_colors():
    # tiles two steps apart still overlap here, so 2 x 2 parity would give two overlapping tiles one color
    boxes = _grid(im_hw=(640, 640), tile=384, stride=128)
    assert len(boxes) == 9
    assert len(_colors_drawn(boxes, (640, 640))) == 9


def test_quarter_overlap_still_needs_only_four():
    boxes = _grid(im_hw=(1280, 1280), tile=512, stride=384)
    assert len(_colors_drawn(boxes, (1280, 1280))) == 4


def test_tiles_that_do_not_overlap_share_one_color():
    boxes = _grid(im_hw=(1024, 1024), tile=512, stride=512)
    assert len(_colors_drawn(boxes, (1024, 1024))) == 1


def test_boxes_are_colored_by_their_group_code():
    img = np.zeros((100, 100, 3), np.uint8)
    boxes = [[10, 10, 40, 40], [50, 10, 80, 40], [10, 50, 40, 80]]
    red, green = (255, 0, 0), (0, 255, 0)
    plot_boxes_by_group(boxes, img, [0, 1, 0], colors=[red, green], line_thickness=1)
    # antialiased lines blend, so compare channels rather than exact values
    assert img[10, 25][0] > img[10, 25][1]  # top edge of box 0, group 0: redder
    assert img[10, 65][1] > img[10, 65][0]  # top edge of box 1, group 1: greener
    assert img[50, 25][0] > img[50, 25][1]  # top edge of box 2, group 0 again


def test_group_codes_must_match_the_boxes():
    import pytest

    with pytest.raises(ValueError, match="group codes"):
        plot_boxes_by_group([[0, 0, 5, 5]], np.zeros((10, 10, 3), np.uint8), [0, 1])
