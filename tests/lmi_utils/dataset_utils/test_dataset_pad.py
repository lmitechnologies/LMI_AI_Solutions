import json

import numpy as np
import pytest

from lmi_utils.dataset_utils.ops.dataset_pad import clip_shapes, pad_annotated_image
from lmi_utils.dataset_utils.representations import Mask, MaskAnnotation, Polygon, PolygonAnnotation
from lmi_utils.gadget_utils.pipeline_utils import fit_array_to_size


def test_a_clipped_polygon_keeps_json_serializable_points():
    polygon = PolygonAnnotation("p", "dent", Polygon(points=[[10, 10], [150, 10], [150, 50]]))
    clip_shapes([polygon], 100, 100)
    assert json.loads(polygon.to_json())["value"]["points"] == [[10.0, 10.0], [100.0, 10.0], [100.0, 50.0]]


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
