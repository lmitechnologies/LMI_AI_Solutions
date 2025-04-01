import os
import json
import numpy as np
import cv2
import pytest
import sys
import logging

PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# add a pytest fixture to add the root path to sys.path based on the argument passed
@pytest.fixture()
def add_root_path(request):
    if request.config.getoption("--test-package") is False:
        sys.path.append(os.path.join(ROOT, 'lmi_utils'))
        logger.info(f"Added {ROOT} to sys.path")
    else:
        logger.info("Skipping adding root path to sys.path")


from dataset_utils.representations import (
    Point2d,
    Box,
    Polygon,
    Mask,
    Label,
    BoxAnnotation,
    MaskAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
    FileAnnotations,
    Dataset,
    AnnotationType
)
from dataset_utils.mask_encoder import mask2rle


# ============================
#  Geometry and Conversion Tests
# ============================

def test_point2d_from_dict_and_to_yolo(add_root_path):
    p = Point2d.from_dict({"x": 10, "y": 20})
    assert p.x == 10
    assert p.y == 20
    yolo = p.to_yolo(100, 200)  # height=100, width=200
    expected = [[10/200, 20/100]]
    assert yolo == expected

def test_point2d_resize_and_pad(add_root_path):
    p = Point2d(10, 20)
    p.resize(100, 100, 200, 200)
    # Coordinates should double.
    assert np.isclose(p.x, 20)
    assert np.isclose(p.y, 40)
    p.pad(pl=5, pt=5)
    assert np.isclose(p.x, 25)
    assert np.isclose(p.y, 45)

def test_box_from_dict_and_to_yolo_no_angle(add_root_path):
    b = Box.from_dict({"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 80, "angle": 0})
    yolo = b.to_yolo(100, 100)
    expected = [[0.3, 0.5, 0.4, 0.6]]
    logger.warning(f'{yolo}')
    assert yolo == expected

def test_box_resize_and_pad(add_root_path):
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

def test_box_to_mask(add_root_path):
    b = Box(10, 20, 50, 80, 0)
    m = b.to_mask(h=100, w=100, mask_type=AnnotationType.MASK)
    assert isinstance(m, Mask)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    assert np.allclose(m.to_numpy(h=100, w=100), mask)

def test_box_invalid_coordinates(add_root_path):
    with pytest.raises(ValueError):
        # x_min > x_max should raise an exception.
        Box(50, 20, 10, 80, 0)
    with pytest.raises(ValueError):
        # y_min > y_max should raise an exception.
        Box(10, 80, 50, 20, 0)

def test_box_point_in_box(add_root_path):
    b = Box(10, 20, 50, 80, 0)
    assert b.point_in_box(30, 50)
    assert not b.point_in_box(5, 50)

def test_polygon_from_dict_and_to_yolo(add_root_path):
    poly = Polygon.from_dict({"points": [[10, 20], [30, 20], [30, 40], [10, 40]]})
    yolo = poly.to_yolo(100, 100)
    expected = [[10/100, 20/100], [30/100, 20/100], [30/100, 40/100], [10/100, 40/100]]
    assert yolo == expected

def test_polygon_resize_and_pad(add_root_path):
    poly = Polygon([[10, 20], [30, 20], [30, 40], [10, 40]])
    poly.resize(100, 100, 200, 200)
    # Points should be doubled.
    pts = np.array(poly.points)
    np.testing.assert_allclose(pts, np.array([[20, 40], [60, 40], [60, 80], [20, 80]]))
    poly.pad(pl=5, pt=5)
    pts = np.array(poly.points)
    np.testing.assert_allclose(pts, np.array([[25, 45], [65, 45], [65, 85], [25, 85]]))

def test_polygon_to_mask(add_root_path):
    poly = Polygon([[10, 20], [30, 20], [30, 40], [10, 40]])
    m = poly.to_mask(h=100, w=100)
    assert isinstance(m, Mask)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly.points).astype(np.int32)], 1)
    assert np.allclose(m.to_numpy(h=100, w=100), mask)

def test_mask_from_dict_and_to_yolo(add_root_path):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    rle = mask2rle(mask)
    m = Mask(rle)
    yolo = m.to_yolo(100, 100)
    assert isinstance(yolo, list)
    assert len(yolo) > 0

def test_mask_resize_and_pad(add_root_path):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    rle = mask2rle(mask)
    m = Mask(rle)
    m = m.resize(100, 100, 200, 200)
    assert m.to_numpy(h=200, w=200).shape == (200, 200)
    m = m.pad(pad_h=10, pad_w=10, h=200, w=200)
    assert m.to_numpy(h=210, w=210).shape == (210, 210)
    

def test_mask_to_box(add_root_path):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    rle = mask2rle(mask)
    m = Mask(rle)
    b = m.to_box(merge_boxes=True, h=100, w=100)
    assert isinstance(b, Box)

# # ============================
# #     Annotation Tests
# # ============================

def test_box_annotation_to_yolo(add_root_path):
    b = Box(10, 20, 50, 80, 0)
    ba = BoxAnnotation("a1", "label1", b)
    yolo = ba.to_yolo(100, 100)
    expected = b.to_yolo(100, 100)
    assert yolo == expected

def test_mask_annotation_to_yolo(add_root_path):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 1
    rle = mask2rle(mask)
    m = Mask(rle)
    ma = MaskAnnotation("a2", "label2", m)
    yolo = ma.to_yolo(h=100, w=100)
    expected = m.to_yolo(h=100, w=100)
    assert yolo == expected

def test_keypoint_annotation_to_yolo(add_root_path):
    p = Point2d(10, 20)
    ka = KeypointAnnotation("a3", "label3", p)
    yolo = ka.to_yolo(100, 100)
    expected = p.to_yolo(100, 100)
    assert yolo == expected

def test_polygon_annotation_to_yolo(add_root_path):
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
    return FileAnnotations(id=file_id,path=file_path, height=height,width=width, annotations=[ba])

def test_file_annotations_relative_path(dummy_file_annotations, add_root_path):
    rel_path = dummy_file_annotations.relative_path("/dummy")
    expected = os.path.relpath(dummy_file_annotations.path, "/dummy")
    assert rel_path == expected

def test_file_annotations_update_file(dummy_file_annotations, add_root_path):
    # Create a new File and update the file annotation.
    file_id = "file1"
    file_path = "/dummy/path/image1.jpg"
    height = 100
    width = 100
    dummy_file_annotations.update_file(id=file_id, path=file_path,height=height,width=width)
    assert dummy_file_annotations.id == "file1"
    assert dummy_file_annotations.height == 100
    assert dummy_file_annotations.width == 100

def test_file_annotations_delete_annotation(dummy_file_annotations, add_root_path):
    # Try deleting an annotation that exists.
    result = dummy_file_annotations.delete_annotation("a1", list_type="annotations")
    assert result is True
    # Try deleting again (should not be found).
    result = dummy_file_annotations.delete_annotation("a1", list_type="annotations")
    assert result is False

def test_file_annotations_update_annotations(dummy_file_annotations, add_root_path):
    # Update annotations list.
    new_ann = BoxAnnotation("a_new", "label1", Box(5, 5, 15, 15, 0))
    dummy_file_annotations.update_annotations([new_ann], list_type="annotations")
    assert len(dummy_file_annotations.annotations) == 1
    assert dummy_file_annotations.annotations[0].id == "a_new"

def test_file_annotations_assign_keypoints_error(add_root_path):
    # Create a FileAnnotations with a keypoint that does not fall inside any box.
    file_id = "file1"
    file_path = "/dummy/path/image1.jpg"
    height = 100
    width = 100
    p = Point2d(5, 5)  # Outside any box we will add.
    ka = KeypointAnnotation("kp1", "label1", p)
    # No box annotation provided.
    fa = FileAnnotations(id=file_id, path=file_path,height=height,width=width, annotations=[ka])
    with pytest.raises(Exception, match="not assigned"):
        fa.assign_keypoints()

def test_file_annotations_to_yolo(dummy_file_annotations, add_root_path):
    logger.warning(f"dummy_file_annotations: {dummy_file_annotations}")
    # logger.warning(f"yolo: {yolo}")
    yolo, label_ids = dummy_file_annotations.to_yolo(
        to_segmentation=False,
        to_object_detection=False,
        merge_boxes=False,
        target_classes=[],
        label_id_idx={"label1": 0}
    )
    assert isinstance(yolo, list)
    assert isinstance(label_ids, list)
    assert len(yolo) > 0
    
    

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
    file_ann = FileAnnotations(id=file_id, path=file_path,height=height,width=width, annotations=[ba, ka])
    return Dataset(labels, [file_ann])

def test_dataset_from_dict(dummy_dataset, add_root_path):
    data = {
        "labels": [
            {"id": "label1", "name": "Label One"},
            {"id": "label2", "name": "Label Two"}
        ],
        "files": [{
            "id": "file1",
            "path": "/dummy/path/image1.jpg",
            "height": 100,
            "width": 100,
            "annotations": [
                {"id": "a1", "label_id": "label1", "type": "Box", 
                 "value": {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 80, "angle": 0}},
                {"id": "a2", "label_id": "label2", "type": "Keypoint", 
                 "value": {"x": 30, "y": 40}}
            ],
            "predictions": []
        }]
    }
    ds = Dataset.from_dict(data)
    assert len(ds.labels) == 2
    assert len(ds.files) == 1

def test_dataset_label_to_index(dummy_dataset, add_root_path):
    idx = dummy_dataset.label_to_index("label1")
    assert isinstance(idx, int)
    with pytest.raises(ValueError):
        dummy_dataset.label_to_index("nonexistent")

def test_dataset_base_path(dummy_dataset, add_root_path):
    # Compute common prefix and ensure base_path is the directory.
    bp = dummy_dataset.base_path
    # In this dummy case, the base path should be the directory part of "/dummy/path/image1.jpg"
    expected = os.path.dirname("/dummy/path/image1.jpg")
    assert bp == expected

def test_dataset_to_yolo(dummy_dataset, add_root_path):
    yolo_data = dummy_dataset.to_yolo(to_segmentation=False, to_object_detection=False)
    assert "image_labels" in yolo_data
    assert "class_map" in yolo_data
    assert "n_kpts" in yolo_data
    for key, annotations in yolo_data["image_labels"].items():
        assert len(annotations) > 0

def test_dataset_save_and_load(tmp_path, dummy_dataset, add_root_path):
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

def test_base_to_dict_and_to_json(dummy_dataset, add_root_path):
    # Test that Base.to_dict and to_json work.
    d = dummy_dataset.to_dict()
    j = dummy_dataset.to_json()
    # Check that the JSON string is parseable.
    loaded = json.loads(j)
    assert isinstance(loaded, dict)
    