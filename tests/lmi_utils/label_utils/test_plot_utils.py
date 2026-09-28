"""Tile-grid overlay: semi-transparent dotted gray lines."""

import numpy as np
import torch

from lmi_utils.image_utils.tiler import Tiler
from lmi_utils.label_utils.plot_utils import plot_boxes_by_group, plot_tile_grid


def _grid(im_hw=(1024, 1024), tile=512, stride=256):
    tiler = Tiler([tile, tile], [stride, stride])
    tiler.tile(torch.zeros(1, 3, *im_hw))
    return tiler.tile_boxes()


def test_grid_lines_are_see_through_gray():
    img = np.zeros((1024, 1024, 3), np.uint8)
    plot_tile_grid(_grid(), img, line_thickness=1)
    drawn = {tuple(c) for c in img.reshape(-1, 3).tolist()} - {(0, 0, 0)}
    assert len(drawn) == 1
    (c,) = drawn
    assert c[0] == c[1] == c[2] and 100 < c[0] < 220  # gray, blended with the image under it


def test_grid_lines_are_dotted():
    img = np.zeros((1024, 1024, 3), np.uint8)
    plot_tile_grid(_grid(), img, line_thickness=2)
    top = img[0, :512].any(axis=1)
    assert 0.4 < top.mean() < 0.6  # dots and gaps 4 px long
    assert top[:4].all() and not top[4:8].any()


def test_a_shared_edge_is_drawn_as_one_line():
    img = np.zeros((1024, 1024, 3), np.uint8)
    plot_tile_grid(_grid(), img, line_thickness=1)
    lit = img.any(axis=2).sum(axis=1)
    # tiles at x=0, 256 and 512 all have a top edge at y=256; their dots coincide
    assert 450 < lit[256] < 580
    assert lit[255] < 20 and lit[257] < 20


def test_no_boxes_leaves_the_image_alone():
    img = np.zeros((8, 8, 3), np.uint8)
    plot_tile_grid(np.zeros((0, 4)), img)
    assert not img.any()


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
