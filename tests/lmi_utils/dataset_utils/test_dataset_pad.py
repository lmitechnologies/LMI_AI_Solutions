import json

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from lmi_utils.dataset_utils.ops.dataset_pad import clip_shapes, pad_annotated_image
from lmi_utils.dataset_utils.representations import Box, BoxAnnotation, Mask, MaskAnnotation, Polygon, PolygonAnnotation
from lmi_utils.gadget_utils.pipeline_utils import fit_array_to_size


def area(annotation):
    return ShapelyPolygon(annotation.value.points).area


def test_a_clipped_polygon_keeps_json_serializable_points():
    [polygon], _ = clip_shapes([PolygonAnnotation("p", "dent", Polygon(points=[[10, 10], [150, 10], [150, 50]]))], 100, 100)
    assert all(isinstance(v, float) for point in json.loads(polygon.to_json())["value"]["points"] for v in point)


def test_a_polygon_outside_the_image_is_deleted():
    # its corners slid onto the edges would make a triangle inside the image
    kept, changed = clip_shapes([PolygonAnnotation("t", "dent", Polygon(points=[[-10, 5], [5, -10], [-10, -10]]))], 100, 100)
    assert kept == [] and changed


def test_a_polygon_crossing_the_edge_keeps_the_part_inside():
    [polygon], changed = clip_shapes([PolygonAnnotation("c", "dent", Polygon(points=[[80, 10], [130, 60], [80, 60]]))], 100, 100)
    assert changed
    assert area(polygon) == pytest.approx(800)


def test_a_polygon_inside_the_image_is_unchanged():
    value = Polygon(points=[[10.5, 10], [50, 10], [50, 50]])
    [polygon], changed = clip_shapes([PolygonAnnotation("i", "dent", value)], 100, 100)
    assert polygon.value is value and not changed


def test_a_flat_polygon_is_deleted():
    kept, changed = clip_shapes([PolygonAnnotation("f", "dent", Polygon(points=[[100, 10], [100, 50], [100, 30]]))], 100, 100)
    assert kept == [] and changed


def test_a_polygon_the_edge_cuts_apart_becomes_one_label_per_piece():
    # a U whose bottom lies outside the image: two arms are left
    u = [[10, 50], [30, 50], [30, 120], [60, 120], [60, 50], [80, 50], [80, 140], [10, 140]]
    kept, _ = clip_shapes([PolygonAnnotation("u", "dent", Polygon(points=u))], 100, 100)
    assert [p.id for p in kept] == ["u_p0", "u_p1"]
    assert sorted(area(p) for p in kept) == pytest.approx([1000, 1000])


def test_a_rotated_box_crossing_the_edge_is_refit_to_the_part_inside():
    original = Box(80, 40, 120, 60, 10)
    inside = ShapelyPolygon(original.to_polygon().points).intersection(ShapelyPolygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
    [clipped], changed = clip_shapes([BoxAnnotation("r", "string", Box(80, 40, 120, 60, 10))], 100, 100)
    refit = ShapelyPolygon(clipped.value.to_polygon().points)
    # the smallest rotated box around the part inside, not the original box; minAreaRect works in float32
    assert changed and clipped.value.angle != 0
    assert refit.buffer(1e-3).contains(inside) and refit.area < original.area()


@pytest.mark.parametrize("cells, kept", [((slice(10, 20), 0), True), ((slice(0, 0), slice(0, 0)), False)], ids=["on the edge", "empty"])
def test_a_mask_is_deleted_only_when_cropping_leaves_it_empty(cells, kept):
    mask = np.zeros((100, 100), np.uint8)
    mask[cells] = 1
    annotation = MaskAnnotation("m", "dent", Mask(mask=mask))
    _, annotations, _ = pad_annotated_image(np.zeros((100, 100, 3), np.uint8), [annotation], 100, 100)
    assert bool(annotations) == kept


def test_a_cropped_mask_keeps_its_pixels_inside_the_image():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:40, 10:30] = 1
    mask[60:90, 70:95] = 1
    _, [annotation], _ = pad_annotated_image(np.zeros((100, 100, 3), np.uint8), [MaskAnnotation("m", "dent", Mask(mask=mask))], 60, 60)
    expected, *_ = fit_array_to_size(mask, 60, 60)
    assert np.array_equal(annotation.value.to_numpy(h=60, w=60), expected)


def test_a_self_crossing_polygon_keeps_its_lobe_inside():
    # a figure eight half outside: its left lobe is inside
    eight = PolygonAnnotation("e", "dent", Polygon(points=[[50, 50], [150, 90], [150, 50], [50, 90]]))
    kept, changed = clip_shapes([eight], 100, 100)
    assert changed and [p.id for p in kept] == ["e"]
    assert area(kept[0]) == pytest.approx(1000)
    assert np.asarray(kept[0].value.points)[:, 0].max() <= 100
