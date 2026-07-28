import json
import logging
import os

import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.mask_encoder import mask2rle
from lmi_utils.dataset_utils.ops.dataset_pad import pad_annotated_image
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
from lmi_utils.label_utils.bbox_utils import rotate

logger = logging.getLogger(__name__)


# ============================
#  Geometry and Conversion Tests
# ============================


def test_point2d_from_dict_and_to_yolo():
    p = Point2d.from_dict({"x": 10, "y": 20, "visibility": 1})
    assert p.x == 10
    assert p.y == 20
    assert p.visibility == 1
    yolo = p.to_yolo(100, 200)  # height=100, width=200
    expected = [[10 / 200, 20 / 100]]
    assert yolo == expected


def test_point2d_resize_and_pad():
    p = Point2d(10, 20)
    p.resize(100, 100, 200, 200)
    # Coordinates should double.
    assert np.isclose(p.x, 20)
    assert np.isclose(p.y, 40)
    p.pad(pl=5, pt=5)
    assert np.isclose(p.x, 25)
    assert np.isclose(p.y, 45)


def test_padding_preserves_visibility_and_drops_points_with_box():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    kept_box = BoxAnnotation("kept", "person", Box(40, 40, 60, 60))
    kept_point = KeypointAnnotation("kept-point", "nose", Point2d(50, 50, visibility=1), bounding_box_id="kept")
    dropped_box = BoxAnnotation("dropped", "person", Box(0, 0, 20, 20))
    dropped_point = KeypointAnnotation("dropped-point", "nose", Point2d(50, 50), bounding_box_id="dropped")

    _, annotations, _ = pad_annotated_image(image, [kept_box, kept_point, dropped_box, dropped_point], 50, 50)

    assert [annotation.id for annotation in annotations] == ["kept", "kept-point"]
    assert annotations[1].value.visibility == 1


def test_box_from_dict_and_to_yolo_no_angle():
    b = Box.from_dict({"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 80, "angle": 0})
    yolo = b.to_yolo(100, 100)
    expected = [[0.3, 0.5, 0.4, 0.6]]
    assert yolo == expected


def test_box_to_yolo_obb_supports_negative_angles_and_clips():
    box = Box(0, 0, 20, 20, -45)
    yolo = np.array(box.to_yolo(100, 100, use_obb=True))
    assert yolo.shape == (4, 2)
    assert np.all((0 <= yolo) & (yolo <= 1))
    assert not np.allclose(yolo, [[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]])


def test_box_resize_and_pad():
    b = Box(10, 20, 50, 80, 0)
    b.resize(orig_h=100, orig_w=100, new_h=200, new_w=200)
    # Coordinates should double.
    assert np.isclose(b.x_min, 20)
    assert np.isclose(b.y_min, 40)
    assert np.isclose(b.x_max, 100)
    assert np.isclose(b.y_max, 160)
    b.pad(pl=5, pt=5)
    assert np.isclose(b.x_min, 25)
    assert np.isclose(b.y_min, 45)
    assert np.isclose(b.x_max, 105)
    assert np.isclose(b.y_max, 165)


def test_box_to_mask():
    b = Box(10, 20, 50, 80, 0)
    m = b.to_mask(h=100, w=100)
    assert isinstance(m, Mask)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    assert np.allclose(m.to_numpy(h=100, w=100), mask)

    # test rotated box
    b = Box(10, 20, 50, 80, 30)  # 30 degrees rotation
    m = b.to_mask(h=100, w=100)
    assert isinstance(m, Mask)
    mask = np.zeros((100, 100), dtype=np.uint8)
    pts = rotate(10, 20, 50 - 10, 80 - 20, 30)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    assert np.allclose(m.to_numpy(h=100, w=100), mask)
    poly = b.to_polygon()
    assert np.allclose(poly.to_numpy(), pts)


def test_box_invalid_coordinates():
    with pytest.raises(ValueError):
        # x_min > x_max should raise an exception.
        Box(50, 20, 10, 80, 0)
    with pytest.raises(ValueError):
        # y_min > y_max should raise an exception.
        Box(10, 80, 50, 20, 0)


def test_box_point_in_box():
    b = Box(10, 20, 50, 80, 0)
    assert b.point_in_box(30, 50)
    assert not b.point_in_box(5, 50)


def test_box_point_in_rotated_box():
    # A wide box pivoted 90 degrees about its top-left corner stands below that corner, not to its right.
    b = Box(10, 10, 50, 20, 90)
    assert b.point_in_box(5, 30)
    assert not b.point_in_box(30, 15)


def test_polygon_from_dict_and_to_yolo():
    poly = Polygon.from_dict({"points": [[10, 20], [30, 20], [30, 40], [10, 40]]})
    yolo = poly.to_yolo(100, 100)
    expected = [
        [10 / 100, 20 / 100],
        [30 / 100, 20 / 100],
        [30 / 100, 40 / 100],
        [10 / 100, 40 / 100],
    ]
    assert yolo == expected


def test_polygon_resize_and_pad():
    poly = Polygon([[10, 20], [30, 20], [30, 40], [10, 40]])
    poly.resize(100, 100, 200, 200)
    # Points should be doubled.
    pts = np.array(poly.points)
    np.testing.assert_allclose(pts, np.array([[20, 40], [60, 40], [60, 80], [20, 80]]))
    poly.pad(pl=5, pt=5)
    pts = np.array(poly.points)
    np.testing.assert_allclose(pts, np.array([[25, 45], [65, 45], [65, 85], [25, 85]]))


def test_polygon_to_mask():
    poly = Polygon([[10, 20], [30, 20], [30, 40], [10, 40]])
    m = poly.to_mask(h=100, w=100)
    assert isinstance(m, Mask)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly.points).astype(np.int32)], 1)
    assert np.allclose(m.to_numpy(h=100, w=100), mask)


@pytest.mark.parametrize("angle", [0, 7, 30, 45, 89, -15, -70])
def test_polygon_to_rbox_round_trips_exactly(angle):
    """
    A rotated rectangle survives polygon -> rbox -> polygon.

    Neither direction may round to the pixel grid: a box is stored in image coordinates and only rasterized
    at the point it is drawn, and a rounding of each corner compounds over an operation chain.
    """
    corners = cv2.boxPoints(((123.4, 210.7), (81.3, 46.9), angle))

    round_tripped = np.array(Polygon(points=corners.tolist()).to_rbox().to_polygon().points)

    def sorted_corners(pts):
        return pts[np.lexsort((pts[:, 0], pts[:, 1]))]

    assert sorted_corners(round_tripped) == pytest.approx(sorted_corners(corners), abs=1e-3)


def test_mask_from_dict_and_to_yolo():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    rle = mask2rle(mask)
    m = Mask(rle)
    yolo = m.to_yolo(100, 100)
    assert isinstance(yolo, list)
    assert len(yolo) > 0


def test_mask_resize_and_pad():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    rle = mask2rle(mask)
    m = Mask(rle)
    m = m.resize(100, 100, 200, 200)
    assert m.to_numpy(h=200, w=200).shape == (200, 200)
    m = m.pad(pad_h=10, pad_w=10, h=200, w=200)
    assert m.to_numpy(h=210, w=210).shape == (210, 210)


def test_mask_to_box():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    rle = mask2rle(mask)
    m = Mask(rle)
    b = m.to_box(merge_boxes=True, h=100, w=100)
    assert isinstance(b, Box)


# # ============================
# #     Annotation Tests
# # ============================


def test_box_annotation_to_yolo():
    b = Box(10, 20, 50, 80, 0)
    ba = BoxAnnotation("a1", "label1", b)
    yolo = ba.to_yolo(100, 100)
    expected = b.to_yolo(100, 100)
    assert yolo == expected


def test_mask_annotation_to_yolo():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    rle = mask2rle(mask)
    m = Mask(rle)
    ma = MaskAnnotation("a2", "label2", m)
    yolo = ma.to_yolo(h=100, w=100)
    expected = m.to_yolo(h=100, w=100)
    assert yolo == expected


def test_keypoint_annotation_to_yolo():
    p = Point2d(10, 20)
    ka = KeypointAnnotation("a3", "label3", p)
    yolo = ka.to_yolo(100, 100)
    expected = p.to_yolo(100, 100)
    assert yolo == expected


def test_polygon_annotation_to_yolo():
    poly = Polygon([[10, 20], [30, 20], [30, 40], [10, 40]])
    pa = PolygonAnnotation("a4", "label4", poly)
    yolo = pa.to_yolo(100, 100)
    expected = poly.to_yolo(100, 100)
    assert yolo == expected


# # ============================
# #  FileAnnotations Tests
# # ============================


@pytest.fixture
def dummy_file_annotations():
    # file = File("file1", "/dummy/path/image1.jpg", height=100, width=100)
    file_id = "file1"
    file_path = "/dummy/path/image1.jpg"
    height = 100
    width = 100

    b = Box(10, 20, 50, 80, 0)
    ba = BoxAnnotation(id="a1", label_id="label1", value=b)
    return FileAnnotations(id=file_id, path=file_path, height=height, width=width, annotations=[ba])


def test_file_annotations_relative_path(dummy_file_annotations):
    rel_path = dummy_file_annotations.relative_path("/dummy")
    expected = os.path.relpath(dummy_file_annotations.path, "/dummy")
    assert rel_path == expected


def test_file_annotations_update_file(dummy_file_annotations):
    # Create a new File and update the file annotation.
    file_id = "file1"
    file_path = "/dummy/path/image1.jpg"
    height = 100
    width = 100
    dummy_file_annotations.update_file(id=file_id, path=file_path, height=height, width=width)
    assert dummy_file_annotations.id == "file1"
    assert dummy_file_annotations.height == 100
    assert dummy_file_annotations.width == 100


def test_file_annotations_delete_annotation(dummy_file_annotations):
    # Try deleting an annotation that exists.
    result = dummy_file_annotations.delete_annotation("a1", list_type="annotations")
    assert result is True
    # Try deleting again (should not be found).
    result = dummy_file_annotations.delete_annotation("a1", list_type="annotations")
    assert result is False


def test_file_annotations_update_annotations(dummy_file_annotations):
    # Update annotations list.
    new_ann = BoxAnnotation("a_new", "label1", Box(5, 5, 15, 15, 0))
    dummy_file_annotations.update_annotations([new_ann], list_type="annotations")
    assert len(dummy_file_annotations.annotations) == 1
    assert dummy_file_annotations.annotations[0].id == "a_new"


def test_file_annotations_assign_keypoints_error():
    # Create a FileAnnotations with a keypoint that does not fall inside any box.
    file_id = "file1"
    file_path = "/dummy/path/image1.jpg"
    height = 100
    width = 100
    p = Point2d(5, 5)  # Outside any box we will add.
    ka = KeypointAnnotation("kp1", "label1", p)
    # No box annotation provided.
    fa = FileAnnotations(id=file_id, path=file_path, height=height, width=width, annotations=[ka])
    with pytest.raises(Exception, match="not assigned"):
        fa.assign_keypoints()


def test_file_annotations_assign_keypoints_accepts_prelinked():
    box = BoxAnnotation("box", "person", Box(10, 10, 90, 90))
    point = KeypointAnnotation("point", "nose", Point2d(20, 20), bounding_box_id="box")
    FileAnnotations("file", "image.jpg", 100, 100, [point, box]).assign_keypoints()
    assert point.bounding_box_id == "box"


def test_file_annotations_assign_keypoints_rejects_dangling_link():
    point = KeypointAnnotation("point", "nose", Point2d(20, 20), bounding_box_id="missing")
    annotations = FileAnnotations("file", "image.jpg", 100, 100, [point])
    with pytest.raises(ValueError, match="missing.*point"):
        annotations.assign_keypoints()


def test_file_annotations_assign_keypoints_requires_unique_containment():
    point = KeypointAnnotation("point", "nose", Point2d(20, 20))
    boxes = [
        BoxAnnotation("box-1", "person", Box(0, 0, 30, 30)),
        BoxAnnotation("box-2", "person", Box(10, 10, 40, 40)),
    ]
    annotations = FileAnnotations("file", "image.jpg", 100, 100, boxes + [point])
    with pytest.raises(ValueError, match="point.*multiple boxes"):
        annotations.assign_keypoints()

    annotations.annotations.pop(1)
    annotations.assign_keypoints()
    assert point.bounding_box_id == "box-1"


def test_file_annotations_assign_keypoints_may_keep_or_drop_the_ambiguous():
    # A bulk conversion reports the overlap and moves on; it is still never guessed at.
    def annotations():
        point = KeypointAnnotation("point", "nose", Point2d(20, 20))
        boxes = [
            BoxAnnotation("box-1", "person", Box(0, 0, 30, 30)),
            BoxAnnotation("box-2", "person", Box(10, 10, 40, 40)),
        ]
        return FileAnnotations("file", "image.jpg", 100, 100, boxes + [point]), point

    kept, point = annotations()
    kept.assign_keypoints(ambiguous="keep")
    assert point.bounding_box_id is None
    assert len(kept.annotations) == 3

    dropped, _ = annotations()
    dropped.assign_keypoints(ambiguous="drop")
    assert [annotation.id for annotation in dropped.annotations] == ["box-1", "box-2"]


def test_file_annotations_assign_keypoints_rejects_an_unknown_policy():
    annotations = FileAnnotations("file", "image.jpg", 100, 100, [])
    with pytest.raises(ValueError, match="ambiguous must be one of"):
        annotations.assign_keypoints(ambiguous="guess")


def test_file_annotations_to_yolo(dummy_file_annotations):
    yolo, label_ids = dummy_file_annotations.to_yolo(
        to_segmentation=False,
        to_object_detection=False,
        merge_boxes=False,
        target_classes=[],
        label_id_idx={"label1": 0},
    )
    assert isinstance(yolo, list)
    assert isinstance(label_ids, list)
    assert len(yolo) > 0


def test_file_annotations_to_yolo_uses_layout_slots_and_visibility():
    box = BoxAnnotation("box", "person", Box(10, 20, 50, 80))
    left = KeypointAnnotation("left", "left_eye", Point2d(20, 30, visibility=1), bounding_box_id="box")
    right = KeypointAnnotation("right", "right_eye", Point2d(40, 30), bounding_box_id="box")
    file_annotations = FileAnnotations("file", "image.jpg", 100, 100, [right, box, left])

    rows, _ = file_annotations.to_yolo(
        label_id_idx={"person": 0},
        keypoint_layouts={"person": ["left_eye", "nose", "right_eye"]},
        n_kpts=3,
    )

    assert rows == [[0, 0.3, 0.5, 0.4, 0.6, 0.2, 0.3, 1, 0, 0, 0, 0.4, 0.3, 2]]


def test_file_annotations_to_yolo_pads_mixed_layouts():
    annotations = [
        BoxAnnotation("person", "person", Box(0, 0, 20, 20)),
        KeypointAnnotation("nose", "nose", Point2d(10, 10), bounding_box_id="person"),
        BoxAnnotation("animal", "animal", Box(20, 20, 40, 40)),
        KeypointAnnotation("paw", "paw", Point2d(30, 30), bounding_box_id="animal"),
        BoxAnnotation("object", "object", Box(40, 40, 60, 60)),
    ]
    file_annotations = FileAnnotations("file", "image.jpg", 100, 100, annotations)
    rows, _ = file_annotations.to_yolo(
        label_id_idx={"person": 0, "animal": 1, "object": 2},
        keypoint_layouts={"person": ["nose", "eye"], "animal": ["paw"], "object": []},
        n_kpts=2,
    )

    assert all(len(row) == 11 for row in rows)
    assert rows[1][-3:] == [0, 0, 0]
    assert rows[2][-6:] == [0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (
            [
                KeypointAnnotation("a", "nose", Point2d(10, 10), bounding_box_id="box"),
                KeypointAnnotation("b", "nose", Point2d(11, 11), bounding_box_id="box"),
            ],
            "Duplicate keypoint",
        ),
        ([KeypointAnnotation("a", "ear", Point2d(10, 10), bounding_box_id="box")], "not in the layout"),
    ],
)
def test_file_annotations_to_yolo_rejects_invalid_slots(points, message):
    annotations = FileAnnotations("file", "image.jpg", 100, 100, [BoxAnnotation("box", "person", Box(0, 0, 20, 20))] + points)
    with pytest.raises(ValueError, match=message):
        annotations.to_yolo(label_id_idx={"person": 0}, keypoint_layouts={"person": ["nose"]}, n_kpts=1)


# ============================
#       Dataset Tests
# ============================


@pytest.fixture
def dummy_dataset():
    labels = [Label("label1", "Label One"), Label("label2", "Label Two")]
    file_id = "file1"
    file_path = "/dummy/path/image1.jpg"
    height = 100
    width = 100
    b = Box(10, 20, 50, 80, 0)
    ba = BoxAnnotation("a1", "label1", b)
    p = Point2d(30, 40)
    ka = KeypointAnnotation("a2", "label2", p)
    file_ann = FileAnnotations(id=file_id, path=file_path, height=height, width=width, annotations=[ba, ka])
    return Dataset(labels, [file_ann])


def test_dataset_from_dict(dummy_dataset):
    data = {
        "labels": [
            {"id": "label1", "name": "Label One"},
            {"id": "label2", "name": "Label Two"},
        ],
        "files": [
            {
                "id": "file1",
                "path": "/dummy/path/image1.jpg",
                "height": 100,
                "width": 100,
                "annotations": [
                    {
                        "id": "a1",
                        "label_id": "label1",
                        "type": "Box",
                        "value": {
                            "x_min": 10,
                            "y_min": 20,
                            "x_max": 50,
                            "y_max": 80,
                            "angle": 0,
                        },
                    },
                    {
                        "id": "a2",
                        "label_id": "label2",
                        "type": "Keypoint",
                        "value": {"x": 30, "y": 40},
                    },
                ],
                "predictions": [],
            }
        ],
    }
    ds = Dataset.from_dict(data)
    assert len(ds.labels) == 2
    assert len(ds.files) == 1


def test_dataset_label_to_index(dummy_dataset):
    idx = dummy_dataset.label_to_index("label1")
    assert isinstance(idx, int)
    with pytest.raises(ValueError):
        dummy_dataset.label_to_index("nonexistent")


def test_dataset_base_path(dummy_dataset):
    # Compute common prefix and ensure base_path is the directory.
    bp = dummy_dataset.base_path
    # In this dummy case, the base path should be the directory part of "/dummy/path/image1.jpg"
    expected = os.path.dirname("/dummy/path/image1.jpg")
    assert bp == expected


def test_dataset_to_yolo(dummy_dataset):
    yolo_data = dummy_dataset.to_yolo(to_segmentation=False, to_object_detection=False)
    assert "image_labels" in yolo_data
    assert "class_map" in yolo_data
    assert "n_kpts" in yolo_data
    for _key, annotations in yolo_data["image_labels"].items():
        assert len(annotations) > 0


def test_dataset_to_yolo_derives_keypoint_count_from_layouts():
    labels = [Label("person", keypoints=["nose", "eye"])]
    files = [
        FileAnnotations(
            "one",
            "one.jpg",
            100,
            100,
            [
                BoxAnnotation("box-1", "person", Box(0, 0, 50, 50)),
                KeypointAnnotation("nose-1", "nose", Point2d(10, 10), bounding_box_id="box-1"),
            ],
        ),
        FileAnnotations(
            "two",
            "two.jpg",
            100,
            100,
            [
                BoxAnnotation("box-2", "person", Box(0, 0, 50, 50)),
                BoxAnnotation("box-3", "person", Box(50, 50, 100, 100)),
                KeypointAnnotation("nose-2", "nose", Point2d(10, 10), bounding_box_id="box-2"),
                KeypointAnnotation("nose-3", "nose", Point2d(60, 60), bounding_box_id="box-3"),
            ],
        ),
    ]

    result = Dataset(labels, files).to_yolo()

    assert result["n_kpts"] == 2
    assert all(len(row) == 11 for rows in result["image_labels"].values() for row in rows)


def test_dataset_save_and_load(tmp_path, dummy_dataset):
    # Test the Base.save and Base.load functionality using a temporary file.
    file_path = tmp_path / "dataset.json"
    # Save dataset as JSON.
    dummy_dataset.save(str(file_path))
    # Now load the dataset back.
    loaded = Dataset.load(str(file_path))
    # Check that labels and files are preserved.
    assert len(loaded.labels) == len(dummy_dataset.labels)
    assert len(loaded.files) == len(dummy_dataset.files)
    # Check one field from a label.
    assert loaded.labels[0].id == dummy_dataset.labels[0].id


def test_base_to_dict_and_to_json(dummy_dataset):
    # Test that Base.to_dict and to_json work.
    d = dummy_dataset.to_dict()
    j = dummy_dataset.to_json()
    # Check that the JSON string is parseable.
    loaded = json.loads(j)
    assert isinstance(loaded, dict)
    assert isinstance(d, dict)


def test_point2d_flip():
    p = Point2d(10, 20)
    # Horizontal flip
    p.flip(flipx=True, w=100)
    assert p.x == 90
    assert p.y == 20
    # Vertical flip
    p.flip(flipy=True, h=50)
    assert p.x == 90
    assert p.y == 30
    # Both flips
    p = Point2d(10, 20)
    p.flip(flipx=True, flipy=True, w=100, h=50)
    assert p.x == 90
    assert p.y == 30
    # Invalid dimensions
    with pytest.raises(ValueError, match="Width must be positive"):
        Point2d(10, 20).flip(flipx=True, w=0)
    with pytest.raises(ValueError, match="Height must be positive"):
        Point2d(10, 20).flip(flipy=True, h=0)


def test_box_flip():
    b = Box(10, 20, 50, 80, 0)
    # Horizontal flip: w=100, flipx=True
    # new_x_min = 100 - 50 = 50
    # new_x_max = 100 - 10 = 90
    b.flip(flipx=True, w=100)
    assert b.x_min == 50
    assert b.x_max == 90
    assert b.y_min == 20
    assert b.y_max == 80

    # Vertical flip: h=100, flipy=True
    # new_y_min = 100 - 80 = 20
    # new_y_max = 100 - 20 = 80
    b = Box(10, 20, 50, 80, 0)
    b.flip(flipy=True, h=100)
    assert b.x_min == 10
    assert b.x_max == 50
    assert b.y_min == 20
    assert b.y_max == 80

    # Both flips
    b = Box(10, 20, 50, 80, 0)
    b.flip(flipx=True, flipy=True, w=100, h=100)
    assert b.x_min == 50
    assert b.x_max == 90
    assert b.y_min == 20
    assert b.y_max == 80

    # Invalid dimensions
    with pytest.raises(ValueError, match="Width must be positive"):
        Box(10, 20, 50, 80, 0).flip(flipx=True, w=0)


def get_flip_expected_corners(x, y, w, h, angle, flip_w, flip_h, flip_x=False, flip_y=False):
    """
    Helper to calculate ground truth for the test: the flipped box's four corners.

    The corners are the box itself. Which of them the flipped box reports as its pivot, and the angle that
    goes with it, is a choice of representation -- a box lying flat is equally the corner on its left at 0
    degrees or the corner on its right at 90 -- so the test compares the geometry rather than the choice.
    """
    # 1. Get exact corners of the input box
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Corners of a rectangle at (0,0) with w,h
    pts_local = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # Rotate
    pts_rot = np.zeros_like(pts_local, dtype=float)
    pts_rot[:, 0] = pts_local[:, 0] * cos_a - pts_local[:, 1] * sin_a
    pts_rot[:, 1] = pts_local[:, 0] * sin_a + pts_local[:, 1] * cos_a

    # Shift to pivot (x,y)
    pts = pts_rot + [x, y]

    # 2. Apply Flip
    if flip_x:
        pts[:, 0] = flip_w - pts[:, 0]
    if flip_y:
        pts[:, 1] = flip_h - pts[:, 1]

    return pts[np.lexsort((pts[:, 0], pts[:, 1]))]


@pytest.mark.parametrize(
    "box_in, flip_kwargs",
    [
        ((15, 25, 65, 55, 30), {"flipx": True, "w": 200}),
        ((80, 50, 200, 140, 45), {"flipx": True, "w": 200}),
        ((50, 50, 70, 130, 90), {"flipy": True, "h": 200}),
    ],
)
def test_flip_rotated_robust(box_in, flip_kwargs):
    x1, y1, x2, y2, a = box_in
    w_box = x2 - x1
    h_box = y2 - y1

    # Setup Object
    box = Box(x1, y1, x2, y2, a)

    # Perform Flip
    box.flip(**flip_kwargs)

    # Calculate Truth
    flip_w = flip_kwargs.get("w", 0)
    flip_h = flip_kwargs.get("h", 0)
    flip_x = flip_kwargs.get("flipx", False)
    flip_y = flip_kwargs.get("flipy", False)

    expected = get_flip_expected_corners(x1, y1, w_box, h_box, a, flip_w, flip_h, flip_x, flip_y)

    corners = np.array(box._rotated_corners(), dtype=float)
    corners = corners[np.lexsort((corners[:, 0], corners[:, 1]))]
    assert corners == pytest.approx(expected, abs=1.0)

    # verify the width and height
    old_dims = sorted([x2 - x1, y2 - y1])
    new_dims = sorted([box.x_max - box.x_min, box.y_max - box.y_min])
    assert new_dims == pytest.approx(old_dims, abs=1)


def test_polygon_flip():
    points = [[10, 20], [30, 20], [30, 40], [10, 40]]
    poly = Polygon(points)
    # Horizontal flip, w=100
    # expected: [[90, 20], [70, 20], [70, 40], [90, 40]]
    poly.flip(flipx=True, w=100)
    expected = [[90, 20], [70, 20], [70, 40], [90, 40]]
    np.testing.assert_allclose(poly.points, expected)

    # Vertical flip, h=100
    # current: [[90, 20], [70, 20], [70, 40], [90, 40]]
    # expected: [[90, 80], [70, 80], [70, 60], [90, 60]]
    poly.flip(flipy=True, h=100)
    expected = [[90, 80], [70, 80], [70, 60], [90, 60]]
    np.testing.assert_allclose(poly.points, expected)


def test_mask_flip():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 1:4] = 1  # y: [2,5), x: [1,4)
    rle = mask2rle(mask)
    m = Mask(rle)

    # Horizontal flip, w=10, h=10
    m.flip(flipx=True, w=10, h=10)
    flipped_mask = m.to_numpy(h=10, w=10)
    expected_mask = np.flip(mask, axis=1)
    assert np.allclose(flipped_mask, expected_mask)

    # Vertical flip, h=10, w=10
    m.flip(flipy=True, h=10, w=10)
    flipped_mask = m.to_numpy(h=10, w=10)
    expected_mask = np.flip(np.flip(mask, axis=1), axis=0)
    assert np.allclose(flipped_mask, expected_mask)


def test_dataset_json_format_preserved():
    """Verify that the JSON output format has all expected keys and correct value structures."""
    mask_arr = np.zeros((100, 100), dtype=np.uint8)
    mask_arr[20:80, 10:50] = 1
    rle = mask2rle(mask_arr)

    labels = [
        Label("label_box", "Box Label", keypoints=["label_kp"]),
        Label("label_kp", "Keypoint Label"),
        Label("label_poly", "Polygon Label"),
        Label("label_mask", "Mask Label"),
    ]
    annotations = [
        BoxAnnotation("ann_box", "label_box", Box(10, 20, 50, 80, 0)),
        KeypointAnnotation("ann_kp", "label_kp", Point2d(30, 40, visibility=1)),
        PolygonAnnotation("ann_poly", "label_poly", Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])),
        MaskAnnotation("ann_mask", "label_mask", Mask(rle)),
    ]
    file_ann = FileAnnotations(id="f1", path="/dummy/img.jpg", height=100, width=100, annotations=annotations)
    dataset = Dataset(labels=labels, files=[file_ann])

    data = json.loads(dataset.to_json())

    # Top-level keys
    assert set(data.keys()) >= {"labels", "files"}

    # Label structure
    for label in data["labels"]:
        assert "id" in label
        assert "color" in label
        assert "annotation_type" in label

    # File structure
    assert len(data["files"]) == 1
    f = data["files"][0]
    assert set(f.keys()) >= {"id", "path", "height", "width", "annotations", "predictions"}
    assert f["id"] == "f1"
    assert f["height"] == 100
    assert f["width"] == 100

    # Annotation structure — common fields
    anns_by_type = {a["type"]: a for a in f["annotations"]}
    for ann in f["annotations"]:
        assert set(ann.keys()) >= {"id", "label_id", "type", "value"}
        assert isinstance(ann["type"], str), "annotation type must be serialized as a string, not an enum"

    # BoxAnnotation value structure
    assert "Box" in anns_by_type
    box_value = anns_by_type["Box"]["value"]
    assert set(box_value.keys()) >= {"x_min", "y_min", "x_max", "y_max", "angle"}
    assert box_value["x_min"] == pytest.approx(10.0)
    assert box_value["y_min"] == pytest.approx(20.0)
    assert box_value["x_max"] == pytest.approx(50.0)
    assert box_value["y_max"] == pytest.approx(80.0)
    assert box_value["angle"] == pytest.approx(0.0)

    # KeypointAnnotation value structure
    assert "Keypoint" in anns_by_type
    kp_ann = anns_by_type["Keypoint"]
    kp_value = kp_ann["value"]
    assert set(kp_value.keys()) >= {"x", "y"}
    assert kp_value["x"] == pytest.approx(30.0)
    assert kp_value["y"] == pytest.approx(40.0)
    assert kp_value["visibility"] == 1
    assert "bounding_box_id" in kp_ann

    # PolygonAnnotation value structure
    assert "Polygon" in anns_by_type
    poly_value = anns_by_type["Polygon"]["value"]
    assert "points" in poly_value
    assert isinstance(poly_value["points"], list)
    assert len(poly_value["points"]) == 4

    # MaskAnnotation value structure (RLE dict)
    assert "Bitmask" in anns_by_type
    mask_value = anns_by_type["Bitmask"]["value"]
    assert isinstance(mask_value, dict)

    # Round-trip: load back and verify labels/files match
    loaded = Dataset.from_dict(data)
    assert len(loaded.labels) == len(dataset.labels)
    assert len(loaded.files) == len(dataset.files)
    assert [lb.id for lb in loaded.labels] == [lb.id for lb in dataset.labels]
    loaded_anns_by_id = {a.id: a for a in loaded.files[0].annotations}
    assert loaded_anns_by_id["ann_box"].value.x_min == pytest.approx(10.0)
    assert loaded_anns_by_id["ann_kp"].value.x == pytest.approx(30.0)
    assert loaded_anns_by_id["ann_kp"].value.visibility == 1
    assert loaded.labels[0].keypoints == ["label_kp"]
    assert len(loaded_anns_by_id["ann_poly"].value.points) == 4
