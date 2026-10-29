import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from lmi_utils.dataset_utils.ops.dataset_crop_by_label import crop_dataset_by_label
from lmi_utils.dataset_utils.representations import (
    Box,
    BoxAnnotation,
    Dataset,
    FileAnnotations,
    KeypointAnnotation,
    Label,
    Mask,
    MaskAnnotation,
    Point2d,
    Polygon,
    PolygonAnnotation,
)


def _crop(annotations, crop=None):
    image = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
    file = FileAnnotations("f", "a.png", 100, 100, [BoxAnnotation("roi", "roi", crop or Box(20, 20, 60, 60))] + annotations)
    images, dataset = crop_dataset_by_label(Dataset(labels=[Label("roi"), Label("dent")], files=[file]), {"a.png": image}, "roi")
    return images["a.png"], {a.id: a for a in dataset.files[0].annotations}, dataset.files[0]


def test_the_image_is_cropped_to_the_pixels_whose_centers_are_in_the_box():
    image, _, file = _crop([], crop=Box(20.4, 20.6, 59.6, 60.6))
    assert image.shape == (40, 40) and (file.height, file.width) == (40, 40)
    assert image[0, 0] == np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)[21, 20]


def test_a_polygon_outside_the_crop_is_deleted():
    _, kept, _ = _crop([PolygonAnnotation("p", "dent", Polygon(points=[[0, 0], [10, 0], [0, 10]]))])
    assert kept == {}


def test_a_polygon_crossing_the_crop_keeps_the_part_inside():
    _, kept, _ = _crop([PolygonAnnotation("p", "dent", Polygon(points=[[40, 30], [90, 80], [40, 80]]))])
    # the part inside the 40x40 crop lies between x=20 and x=40, below the diagonal
    assert ShapelyPolygon(kept["p"].value.points).area == pytest.approx(400)


def test_a_box_outside_the_crop_is_deleted_and_one_crossing_it_is_clipped():
    boxes = [
        BoxAnnotation("out", "dent", Box(70, 70, 90, 90)),
        BoxAnnotation("cross", "dent", Box(50, 10, 80, 30)),
        BoxAnnotation("on_top_edge", "dent", Box(10, 20, 30, 40)),
    ]
    _, kept, _ = _crop(boxes)
    assert list(kept) == ["cross", "on_top_edge"]
    assert kept["cross"].value.to_numpy().tolist() == [30, 0, 40, 10, 0]
    assert kept["on_top_edge"].value.to_numpy().tolist() == [0, 0, 10, 20, 0]


def test_keypoints_follow_the_crop():
    keypoints = [
        KeypointAnnotation("free", "nose", Point2d(30, 25)),
        KeypointAnnotation("on_roi", "nose", Point2d(40, 40), bounding_box_id="roi"),
        KeypointAnnotation("on_gone", "nose", Point2d(30, 30), bounding_box_id="gone"),
    ]
    _, kept, _ = _crop([BoxAnnotation("gone", "dent", Box(80, 80, 90, 90))] + keypoints)
    assert list(kept) == ["free"]
    assert (kept["free"].value.x, kept["free"].value.y) == (10, 5)


def test_a_mask_is_cropped_and_an_empty_one_deleted():
    inside = np.zeros((100, 100), dtype=np.uint8)
    inside[30:35, 25:28] = 1
    outside = np.zeros((100, 100), dtype=np.uint8)
    outside[80:90, 80:90] = 1
    _, kept, _ = _crop([MaskAnnotation("in", "dent", Mask(mask=inside)), MaskAnnotation("out", "dent", Mask(mask=outside))])
    assert list(kept) == ["in"]
    assert np.array_equal(kept["in"].value.to_numpy(h=40, w=40), inside[20:60, 20:60])


def test_a_rotated_crop_box_is_rejected():
    with pytest.raises(ValueError, match="rotated"):
        _crop([], crop=Box(20, 20, 60, 60, -10))
