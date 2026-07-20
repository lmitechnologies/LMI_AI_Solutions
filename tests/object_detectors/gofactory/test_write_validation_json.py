import json
import os
import types

import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    BoxAnnotation,
    Dataset,
    FileAnnotations,
    KeypointAnnotation,
    Label,
    Mask,
    Point2d,
    Polygon,
)
from object_detectors.gofactory import write_validation_json as wvj

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
DOTA8_IMG_DIR = os.path.join(ASSETS, "images", "dota8")
COCO_IMG_DIR = os.path.join(ASSETS, "images", "coco")
OBB_WEIGHTS = os.path.join(ASSETS, "models", "od", "ultralytics", "yolo26n-obb.pt")
POSE_WEIGHTS = os.path.join(ASSETS, "models", "od", "ultralytics", "yolo26n-pose.pt")


class _StubModel:
    """Minimal stand-in for compute_ious: only .device and (optionally) .model.kpt_shape are used."""

    def __init__(self, nkpt=None):
        self.device = "cpu"
        if nkpt is not None:
            self.model = types.SimpleNamespace(kpt_shape=[nkpt, 3])


def _ann(label_id, ann_type, value, ann_id="0", confidence=1.0):
    return Annotation(id=ann_id, label_id=label_id, type=ann_type, value=value, confidence=confidence)


# ============================
#  parse_annotations
# ============================


def test_parse_box_object_detection():
    out = wvj.parse_annotations([_ann("a", AnnotationType.BOX, Box(10, 20, 30, 40, 0))], 100, 100, "ObjectDetection")
    assert out["boxes"].shape == (1, 5)
    np.testing.assert_array_equal(out["boxes"][0], [10, 20, 30, 40, 0])
    assert list(out["classes"]) == ["a"]
    assert out["masks"].size == 0 and out["points"].size == 0


def test_parse_box_oriented_uses_corners():
    out = wvj.parse_annotations([_ann("a", AnnotationType.BOX, Box(10, 20, 30, 40, 0))], 100, 100, "OrientedObjectDetection")
    assert out["boxes"].shape == (1, 4, 2)  # 4 corners (xyxyxyxy) for rotated iou


def test_parse_mask():
    m = np.zeros((100, 100), np.uint8)
    m[10:20, 10:20] = 1
    out = wvj.parse_annotations([_ann("a", AnnotationType.MASK, Mask(m))], 100, 100, "InstanceSegmentation")
    assert out["masks"].shape == (1, 100, 100)


def test_parse_polygon_instance_segmentation_becomes_mask():
    poly = Polygon(points=[[10, 10], [30, 10], [30, 30], [10, 30]])
    out = wvj.parse_annotations([_ann("a", AnnotationType.POLYGON, poly)], 100, 100, "InstanceSegmentation")
    assert out["masks"].shape == (1, 100, 100)


def test_parse_polygon_oriented_becomes_box():
    poly = Polygon(points=[[10, 10], [30, 10], [30, 30], [10, 30]])
    out = wvj.parse_annotations([_ann("a", AnnotationType.POLYGON, poly)], 100, 100, "OrientedObjectDetection")
    assert out["boxes"].shape == (1, 4, 2)


def test_parse_polygon_unsupported_is_skipped_but_class_recorded(caplog):
    # for ObjectDetection a polygon is skipped, yet its label is still recorded (label_id appended before type check)
    poly = Polygon(points=[[10, 10], [30, 10], [30, 30]])
    out = wvj.parse_annotations([_ann("a", AnnotationType.POLYGON, poly)], 100, 100, "ObjectDetection")
    assert out["boxes"].size == 0
    assert list(out["classes"]) == ["a"]


def test_parse_keypoint():
    out = wvj.parse_annotations([_ann("a", AnnotationType.KEYPOINT, Point2d(5, 6))], 100, 100, "KeypointDetection")
    assert out["points"].shape == (1, 2)
    np.testing.assert_array_equal(out["points"][0], [5, 6])


def test_parse_keypoints_uses_box_layout_and_visibility():
    annotations = [
        KeypointAnnotation("right", "right_eye", Point2d(30, 20), bounding_box_id="box"),
        BoxAnnotation("box", "person", Box(0, 0, 50, 50)),
        KeypointAnnotation("left", "left_eye", Point2d(10, 20, visibility=1), bounding_box_id="box"),
    ]
    out = wvj.parse_annotations(
        annotations,
        100,
        100,
        "KeypointDetection",
        keypoint_layouts={"person": ["left_eye", "nose", "right_eye"]},
        n_kpts=3,
    )
    np.testing.assert_array_equal(out["points"], [[[10, 20, 1], [0, 0, 0], [30, 20, 2]]])


def test_parse_unsupported_type_raises():
    bad = Annotation(id="0", label_id="a", value=Box(0, 0, 1, 1, 0))  # type left as None
    with pytest.raises(Exception, match="Not supported type"):
        wvj.parse_annotations([bad], 100, 100, "ObjectDetection")


# ============================
#  update_annotation_ids
# ============================


def test_update_annotation_ids():
    anns = [Annotation(id="x", label_id="a"), Annotation(id="y", label_id="b"), Annotation(id="z", label_id="c")]
    wvj.update_annotation_ids(anns, start_id=5)
    assert [a.id for a in anns] == ["5", "6", "7"]


def test_update_annotation_ids_updates_keypoint_links():
    annotations = [
        BoxAnnotation("box", "person", Box(0, 0, 10, 10)),
        KeypointAnnotation("point", "nose", Point2d(5, 5), bounding_box_id="box"),
    ]
    wvj.update_annotation_ids(annotations, start_id=5)
    assert annotations[1].bounding_box_id == "5"


# ============================
#  build_prediction_annotations
# ============================


def test_build_object_detection():
    preds = {
        "classes": np.array(["a", "b"]),
        "boxes": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float),
        "scores": np.array([0.9, 0.8]),
    }
    out = wvj.build_prediction_annotations(preds, "ObjectDetection", start_id=3)
    assert [a.id for a in out] == ["3", "4"]
    assert all(a.type == AnnotationType.BOX for a in out)
    assert out[0].confidence == pytest.approx(0.9)


def test_build_instance_segmentation():
    m = np.zeros((20, 20), np.uint8)
    m[2:5, 2:5] = 1
    preds = {
        "classes": np.array(["a"]),
        "boxes": np.array([[0, 0, 1, 1]], dtype=float),
        "scores": np.array([0.5]),
        "masks": np.array([m]),
    }
    out = wvj.build_prediction_annotations(preds, "InstanceSegmentation", 0)
    assert len(out) == 1 and out[0].type == AnnotationType.MASK


def test_build_keypoint_expands_box_plus_points():
    preds = {
        "classes": np.array(["a"]),
        "boxes": np.array([[1, 2, 3, 4]], dtype=float),
        "scores": np.array([0.7]),
        "points": np.array([[[10, 11, 1], [12, 13, 1], [14, 15, 1]]], dtype=float),  # 1 box, 3 kpts (x, y, vis)
    }
    out = wvj.build_prediction_annotations(preds, "KeypointDetection", 0)
    assert [a.id for a in out] == ["0", "1", "2", "3"]  # 1 box + 3 keypoints, contiguous ids
    assert out[0].type == AnnotationType.BOX
    assert [a.type for a in out[1:]] == [AnnotationType.KEYPOINT] * 3
    assert out[1].value.x == 10 and out[1].value.y == 11  # visibility dropped, (x, y) kept
    assert all(a.bounding_box_id == "0" for a in out[1:])


def test_build_oriented():
    corners = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=float)
    preds = {"classes": np.array(["a"]), "boxes": np.array([corners]), "scores": np.array([0.6])}
    out = wvj.build_prediction_annotations(preds, "OrientedObjectDetection", 0)
    assert len(out) == 1 and out[0].type == AnnotationType.BOX
    assert isinstance(out[0].value, Box)  # Polygon corners converted to a rotated Box


# ============================
#  compute_ious
# ============================


def _empty():
    return {"boxes": np.array([]), "masks": np.array([]), "points": np.array([]), "classes": np.array([])}


def test_compute_ious_object_detection():
    labels = _empty()
    labels["boxes"] = np.array([[0, 0, 10, 10, 0], [20, 20, 30, 30, 0]], dtype=float)
    preds = _empty()
    preds["boxes"] = np.array([[0, 0, 10, 10, 0]], dtype=float)
    res = wvj.compute_ious(labels, preds, "ObjectDetection", _StubModel())
    assert res["n_gt"] == 2 and res["n_pred"] == 1
    assert res["ious"].shape == (2, 1)
    assert res["ious"][0, 0].item() == pytest.approx(1.0)  # identical box -> iou 1


def test_compute_ious_empty_returns_none():
    res = wvj.compute_ious(_empty(), _empty(), "ObjectDetection", _StubModel())
    assert res["n_gt"] == 0 and res["n_pred"] == 0
    assert res["ious"] is None


def test_compute_ious_instance_segmentation():
    m = np.zeros((10, 10), np.uint8)
    m[0:5, 0:5] = 1
    labels = _empty()
    labels["masks"] = np.array([m])
    preds = _empty()
    preds["masks"] = np.array([m])
    res = wvj.compute_ious(labels, preds, "InstanceSegmentation", _StubModel())
    assert res["ious"].shape == (1, 1)
    assert res["ious"][0, 0].item() == pytest.approx(1.0)


def test_compute_ious_oriented():
    c = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    labels = _empty()
    labels["boxes"] = np.array([c])
    preds = _empty()
    preds["boxes"] = np.array([c])
    res = wvj.compute_ious(labels, preds, "OrientedObjectDetection", _StubModel())
    assert res["ious"].shape == (1, 1)
    assert res["ious"][0, 0].item() == pytest.approx(1.0, abs=1e-3)


def test_compute_ious_keypoint():
    labels = _empty()
    labels["boxes"] = np.array([[0, 0, 10, 10, 0]], dtype=float)
    labels["points"] = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)  # 1 box * 3 kpts, flat
    preds = _empty()
    preds["boxes"] = np.array([[0, 0, 10, 10, 0]], dtype=float)
    preds["points"] = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)
    res = wvj.compute_ious(labels, preds, "KeypointDetection", _StubModel(nkpt=3))
    assert res["n_gt"] == 1 and res["n_pred"] == 1
    assert res["ious"].shape == (1, 1)
    assert res["n_gt_kpt"] == 3 and res["n_pred_kpt"] == 3
    assert res["ious_kpt"].shape == (1, 1)


def test_compute_ious_keypoint_uses_ground_truth_visibility(monkeypatch):
    labels = _empty()
    labels["boxes"] = np.array([[0, 0, 10, 10, 0]], dtype=float)
    labels["points"] = np.array([[[2, 2, 1], [0, 0, 0]]], dtype=float)
    preds = _empty()
    preds["boxes"] = np.array([[0, 0, 10, 10, 0]], dtype=float)
    preds["points"] = np.array([[2, 2], [100, 100]], dtype=float)
    captured = {}

    def fake_kpt_iou(gt_points, pred_points, sigma, area):
        captured["gt_points"] = gt_points.cpu().numpy()
        return wvj.torch.ones((len(gt_points), len(pred_points)))

    monkeypatch.setattr(wvj, "kpt_iou", fake_kpt_iou)
    result = wvj.compute_ious(labels, preds, "KeypointDetection", _StubModel(nkpt=2))

    np.testing.assert_array_equal(captured["gt_points"], labels["points"])
    assert result["n_gt_kpt"] == 2


# ============================
#  MODEL_CLASSES dispatch
# ============================


def test_model_classes_keys():
    assert set(wvj.MODEL_CLASSES) == {"ObjectDetection", "InstanceSegmentation", "OrientedObjectDetection", "KeypointDetection"}


def test_write_json_rejects_unknown_model_type(tmp_path):
    # the model-type check runs before any IO, so bad types raise immediately
    with pytest.raises(Exception, match="Not supported model type"):
        wvj.write_json(
            "weights.pt",
            "BadType",
            None,
            str(tmp_path),
            str(tmp_path / "labels.json"),
            str(tmp_path / "preds.json"),
            str(tmp_path / "imgs"),
            str(tmp_path / "ious"),
            None,
        )


# ============================
#  Integration (end-to-end write_json on committed assets)
# ============================


def _save_dataset(path, img_name, h, w, annotations, labels):
    fa = FileAnnotations(id="0", path=img_name, height=h, width=w, annotations=annotations, predictions=[])
    Dataset(labels=labels, files=[fa]).save(str(path))


def _run_and_check_outputs(tmp_path, weights, model_type, img_dir, img_name, annotations, labels, image_size, expect_kpt=False, nkpt=0):
    label_path = tmp_path / "labels.json"
    _save_dataset(label_path, img_name, *cv2.imread(os.path.join(img_dir, img_name)).shape[:2], annotations, labels)
    out_pred = tmp_path / "out" / "preds.json"
    out_img_dir = tmp_path / "out" / "images"
    out_iou_dir = tmp_path / "out" / "ious"

    wvj.write_json(
        weights,
        model_type,
        None,
        img_dir,
        str(label_path),
        str(out_pred),
        str(out_img_dir),
        str(out_iou_dir),
        image_size=image_size,
        confidence=0.25,
    )

    ds = Dataset.load(str(out_pred))
    assert len(ds.files) == 1
    f = ds.files[0]
    assert max(f.width, f.height) == max(image_size)  # resized to fit image_size keeping aspect ratio
    assert os.path.exists(os.path.join(out_img_dir, img_name))

    iou_json = json.load(open(os.path.join(out_iou_dir, f.id + ".json")))
    assert {"n_gt", "n_pred", "iou"}.issubset(iou_json.keys())
    n_gt, n_pred = iou_json["n_gt"], iou_json["n_pred"]
    if n_gt and n_pred:
        assert len(iou_json["iou"]) == n_gt and len(iou_json["iou"][0]) == n_pred
    if expect_kpt:
        assert {"kpt_iou", "n_gt_kpt", "n_pred_kpt"}.issubset(iou_json.keys())
        assert iou_json["n_gt_kpt"] == n_gt * nkpt
        assert iou_json["n_pred_kpt"] == n_pred * nkpt
    return iou_json


@pytest.mark.skipif(not os.path.exists(OBB_WEIGHTS), reason=f"OBB weights not found: {OBB_WEIGHTS}")
def test_write_json_oriented_end_to_end(tmp_path):
    img_name = "P1470__1024__3296___1648.jpg"
    annotations = [_ann("soccer ball field", AnnotationType.BOX, Box(300, 300, 700, 700, 0))]  # central box, survives unpad
    labels = [Label(id="soccer ball field")]
    _run_and_check_outputs(
        tmp_path, OBB_WEIGHTS, "OrientedObjectDetection", DOTA8_IMG_DIR, img_name, annotations, labels, image_size=(1024, 1024)
    )


@pytest.mark.skipif(not os.path.exists(POSE_WEIGHTS), reason=f"pose weights not found: {POSE_WEIGHTS}")
def test_write_json_keypoint_end_to_end(tmp_path):
    img_name = "221872164_2e4d0bcc08_z.jpg"
    h, w = cv2.imread(os.path.join(COCO_IMG_DIR, img_name)).shape[:2]
    nkpt = 17  # yolo26n-pose is COCO 17-keypoint; GT must match to pass the keypoint count validation
    box = _ann("person", AnnotationType.BOX, Box(0.3 * w, 0.3 * h, 0.7 * w, 0.7 * h, 0))
    # 17 keypoints on a grid inside the central box so none are dropped during unpadding
    kpts = []
    for i in range(nkpt):
        x = (0.35 + 0.30 * (i % 5) / 4) * w
        y = (0.35 + 0.30 * (i // 5) / 3) * h
        kpts.append(_ann("kp", AnnotationType.KEYPOINT, Point2d(x, y), ann_id=str(i + 1), confidence=None))
    labels = [Label(id="person"), Label(id="kp")]
    _run_and_check_outputs(
        tmp_path,
        POSE_WEIGHTS,
        "KeypointDetection",
        COCO_IMG_DIR,
        img_name,
        [box] + kpts,
        labels,
        image_size=(640, 640),
        expect_kpt=True,
        nkpt=nkpt,
    )
