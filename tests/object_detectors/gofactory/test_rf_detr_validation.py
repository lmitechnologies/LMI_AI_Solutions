import json
import os

import cv2
import numpy as np
import pytest
import torch

from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    Dataset,
    FileAnnotations,
    Label,
    Mask,
    Polygon,
)
from object_detectors.gofactory import rf_detr_validation as rdv

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
COCO_IMG_DIR = os.path.join(ASSETS, "images", "coco")
SEG_WEIGHTS = os.path.join(ASSETS, "models", "od", "rf_detr", "rf-detr-seg-small.pth")


def _ann(label_id, ann_type, value, ann_id="0", confidence=1.0):
    return Annotation(id=ann_id, label_id=label_id, type=ann_type, value=value, confidence=confidence)


# ============================
#  parse_label_boxes / parse_label_masks
# ============================


def test_parse_label_boxes_box_and_polygon():
    poly = Polygon(points=[[10, 10], [30, 10], [30, 30], [10, 30]])
    out = rdv.parse_label_boxes(
        [
            _ann("a", AnnotationType.BOX, Box(10, 20, 30, 40, 0)),
            _ann("b", AnnotationType.POLYGON, poly, ann_id="1"),
        ]
    )
    assert out.shape == (2, 4)  # every annotation yields one xyxy row
    np.testing.assert_array_equal(out[0], [10, 20, 30, 40])
    np.testing.assert_array_equal(out[1], [10, 10, 30, 30])  # polygon reduced to its bounding box


def test_parse_label_masks_rasterizes_non_masks():
    m = np.zeros((100, 100), np.uint8)
    m[10:20, 10:20] = 1
    out = rdv.parse_label_masks(
        [
            _ann("a", AnnotationType.MASK, Mask(m)),
            _ann("b", AnnotationType.BOX, Box(10, 20, 30, 40, 0), ann_id="1"),
        ],
        100,
        100,
    )
    assert out.shape == (2, 100, 100)
    np.testing.assert_array_equal(out[0], m)
    assert out[1].sum() > 0  # box rasterized as a filled region


# ============================
#  build_prediction_annotations
# ============================


def test_build_prediction_annotations_od():
    preds = {
        "classes": np.array(["a", "b"]),
        "boxes": np.array([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 40.0, 40.0]]),
        "scores": np.array([0.9, 0.5]),
    }
    out = rdv.build_prediction_annotations(preds, rdv.MODEL_TYPE_OBJECT_DETECTION, start_id=3)
    assert [a.id for a in out] == ["3", "4"]  # ids sequential from start_id
    assert all(a.type == AnnotationType.BOX for a in out)
    assert out[0].label_id == "a" and out[0].confidence == pytest.approx(0.9)
    np.testing.assert_array_equal(out[0].value.to_numpy()[:4], [10, 10, 50, 50])


def test_build_prediction_annotations_seg():
    m = np.zeros((64, 64), np.float32)
    m[8:16, 8:16] = 1
    preds = {"classes": np.array(["a"]), "boxes": np.array([[8.0, 8.0, 16.0, 16.0]]), "scores": np.array([0.7]), "masks": [m]}
    out = rdv.build_prediction_annotations(preds, rdv.MODEL_TYPE_INSTANCE_SEGMENTATION, start_id=0)
    assert len(out) == 1 and out[0].type == AnnotationType.MASK
    np.testing.assert_array_equal(out[0].value.to_numpy(h=64, w=64), m.astype(np.uint8))


# ============================
#  IoU matrices
# ============================


def test_box_iou_matrix_values():
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])
    pred = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 5.0], [20.0, 20.0, 30.0, 30.0]])
    m = rdv.box_iou_matrix(gt, pred, "cpu")
    assert m.shape == (1, 3)
    assert m[0, 0].item() == pytest.approx(1.0)
    assert m[0, 1].item() == pytest.approx(0.5)
    assert m[0, 2].item() == pytest.approx(0.0)


def test_mask_iou_matrix_values():
    a = np.zeros((32, 32), np.uint8)
    a[0:8, 0:8] = 1
    b = np.zeros((32, 32), np.uint8)
    b[0:8, 0:4] = 1
    c = np.zeros((32, 32), np.uint8)
    c[16:24, 16:24] = 1
    m = rdv.mask_iou_matrix(np.array([a]), np.array([a, b, c]), "cpu")
    assert m.shape == (1, 3)
    assert m[0, 0].item() == pytest.approx(1.0)
    assert m[0, 1].item() == pytest.approx(0.5)
    assert m[0, 2].item() == pytest.approx(0.0)


def test_compute_ious_empty_sides():
    preds = {"classes": np.array([]), "boxes": np.zeros((0, 4))}
    n_gt, n_pred, ious = rdv.compute_ious(
        [_ann("a", AnnotationType.BOX, Box(0, 0, 10, 10, 0))], preds, rdv.MODEL_TYPE_OBJECT_DETECTION, 32, 32, "cpu"
    )
    assert (n_gt, n_pred, ious) == (1, 0, None)

    preds = {"classes": np.array(["a"]), "boxes": np.array([[0.0, 0.0, 10.0, 10.0]])}
    n_gt, n_pred, ious = rdv.compute_ious([], preds, rdv.MODEL_TYPE_OBJECT_DETECTION, 32, 32, "cpu")
    assert (n_gt, n_pred, ious) == (0, 1, None)


def test_compute_ious_od():
    annotations = [_ann("a", AnnotationType.BOX, Box(0, 0, 10, 10, 0))]
    preds = {"classes": np.array(["a"]), "boxes": np.array([[0.0, 0.0, 10.0, 10.0]])}
    n_gt, n_pred, ious = rdv.compute_ious(annotations, preds, rdv.MODEL_TYPE_OBJECT_DETECTION, 32, 32, "cpu")
    assert (n_gt, n_pred) == (1, 1)
    assert ious[0, 0].item() == pytest.approx(1.0)


# ============================
#  write_json (stubbed model)
# ============================


class _StubRfdetrModel:
    """Stand-in for RfdetrModel: one fixed box prediction per image, on cpu."""

    def __init__(self, model_path, model_type=None, image_size=None, **kwargs):
        self.device = "cpu"

    def warmup(self):
        pass

    def predict(self, image, configs=None, **kwargs):
        preds = {
            "boxes": [np.array([[10.0, 10.0, 50.0, 50.0]])],
            "scores": [np.array([0.9])],
            "classes": [np.array(["a"])],
        }
        return preds, None


def test_write_json_rejects_unknown_model_type(tmp_path):
    with pytest.raises(ValueError, match="Not supported model type"):
        rdv.write_json(
            model_path="weights.pth",
            model_type="BadType",
            variant="small",
            image_size=(512, 512),
            img_dir=str(tmp_path),
            label_path=str(tmp_path / "labels.json"),
            out_pred_json=str(tmp_path / "preds.json"),
            out_image_dir=str(tmp_path / "imgs"),
            out_iou_dir=str(tmp_path / "ious"),
            confidence=0.5,
        )


def test_write_json_end_to_end_stubbed(tmp_path, monkeypatch):
    import object_detectors.rf_detr_lmi.model as rf_detr_model

    monkeypatch.setattr(rf_detr_model, "RfdetrModel", _StubRfdetrModel)

    img_name = "img.png"
    im = np.zeros((64, 64, 3), np.uint8)
    cv2.imwrite(str(tmp_path / img_name), im)

    label_path = tmp_path / "labels.json"
    fa = FileAnnotations(
        id="0",
        path=img_name,
        height=64,
        width=64,
        annotations=[_ann("a", AnnotationType.BOX, Box(10, 10, 50, 50, 0))],
        predictions=[],
    )
    Dataset(labels=[Label(id="a")], files=[fa]).save(str(label_path))

    out_pred = tmp_path / "out" / "preds.json"
    out_img_dir = tmp_path / "out" / "images"
    out_iou_dir = tmp_path / "out" / "ious"
    rdv.write_json(
        model_path="weights.pth",
        model_type=rdv.MODEL_TYPE_OBJECT_DETECTION,
        variant="small",
        image_size=(512, 512),
        img_dir=str(tmp_path),
        label_path=str(label_path),
        out_pred_json=str(out_pred),
        out_image_dir=str(out_img_dir),
        out_iou_dir=str(out_iou_dir),
        confidence=0.5,
    )

    ds = Dataset.load(str(out_pred))
    assert len(ds.files) == 1
    f = ds.files[0]
    assert (f.width, f.height) == (64, 64)  # rf-detr passes images through unchanged
    assert len(f.predictions) == 1 and f.predictions[0].label_id == "a"
    assert os.path.exists(os.path.join(str(out_img_dir), img_name))

    iou_json = json.load(open(os.path.join(str(out_iou_dir), "0.json")))
    assert iou_json["n_gt"] == 1 and iou_json["n_pred"] == 1
    assert iou_json["iou"][0][0] == pytest.approx(1.0)


# ============================
#  Integration (end-to-end write_json on committed assets)
# ============================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(not os.path.exists(SEG_WEIGHTS), reason=f"seg weights not found: {SEG_WEIGHTS}")
def test_write_json_seg_end_to_end(tmp_path):
    img_name = "221872164_2e4d0bcc08_z.jpg"
    h, w = cv2.imread(os.path.join(COCO_IMG_DIR, img_name)).shape[:2]
    annotations = [_ann("person", AnnotationType.BOX, Box(0.3 * w, 0.3 * h, 0.7 * w, 0.7 * h, 0))]
    label_path = tmp_path / "labels.json"
    fa = FileAnnotations(id="0", path=img_name, height=h, width=w, annotations=annotations, predictions=[])
    Dataset(labels=[Label(id="person")], files=[fa]).save(str(label_path))

    out_pred = tmp_path / "out" / "preds.json"
    out_img_dir = tmp_path / "out" / "images"
    out_iou_dir = tmp_path / "out" / "ious"
    rdv.write_json(
        model_path=SEG_WEIGHTS,
        model_type=rdv.MODEL_TYPE_INSTANCE_SEGMENTATION,
        variant="seg-small",
        image_size=(384, 384),
        img_dir=COCO_IMG_DIR,
        label_path=str(label_path),
        out_pred_json=str(out_pred),
        out_image_dir=str(out_img_dir),
        out_iou_dir=str(out_iou_dir),
        confidence=0.25,
    )

    ds = Dataset.load(str(out_pred))
    f = ds.files[0]
    assert (f.width, f.height) == (w, h)  # input passes through unchanged
    assert all(p.type == AnnotationType.MASK for p in f.predictions)
    assert os.path.exists(os.path.join(str(out_img_dir), img_name))

    iou_json = json.load(open(os.path.join(str(out_iou_dir), "0.json")))
    assert iou_json["n_gt"] == 1
    assert iou_json["n_pred"] == len(f.predictions)
    if iou_json["n_pred"]:
        assert len(iou_json["iou"]) == 1 and len(iou_json["iou"][0]) == iou_json["n_pred"]
