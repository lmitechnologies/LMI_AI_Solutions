import logging
import os

import cv2
import numpy as np
import pytest
import torch
from rfdetr import RFDETRNano

from object_detectors.od_core.object_detector import ObjectDetector

logger = logging.getLogger(__name__)

COCO_DIR = "tests/assets/images/coco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PTH_FILE = "tests/assets/models/od/rf_detr/checkpoint.pth"
OD_MODEL = f"tests/assets/models/od/rf_detr/model_{DEVICE}.pt"
OUT_DIR = "tests/outputs/od/rf_detr"
IMAGE_SIZE = 384


def load_image(path):
    im = cv2.imread(path)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb


@pytest.fixture(scope="module")
def imgs_coco():
    paths = [os.path.join(COCO_DIR, img) for img in os.listdir(COCO_DIR)]
    images = []
    for p in paths:
        if "png" not in p and "jpg" not in p:
            continue
        rgb = load_image(p)
        h, w = rgb.shape[:2]
        images.append(rgb)
    return images


@pytest.fixture(scope="module")
def rf_model():
    model = RFDETRNano(pretrain_weights=PTH_FILE, device=DEVICE)
    return model


class Test_Rfdetr_Model:
    def test_compare_with_rfdetr(self, imgs_coco, tolerance=1e-4):
        "Use cpu to ensure consistency"
        rf_model = RFDETRNano(pretrain_weights=PTH_FILE, device="cpu")
        obj_detector = ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=OD_MODEL.replace("cuda", "cpu"),
            class_map=rf_model.class_names,
            image_size=[IMAGE_SIZE, IMAGE_SIZE],
            device="cpu",
        )

        # rf_model.optimize_for_inference()
        for img in imgs_coco:
            temp_image = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            outputs_od = obj_detector.predict(temp_image, configs=0.5)
            outputs_rfdetr = rf_model.predict(temp_image, threshold=0.5)
            assert "boxes" in outputs_od
            assert "scores" in outputs_od
            assert "classes" in outputs_od
            rf_detr_boxes = outputs_rfdetr.xyxy
            rf_detr_classes = [rf_model.class_names[c] for c in outputs_rfdetr.class_id]

            assert rf_detr_boxes.shape[0] == outputs_od["boxes"].shape[0]
            assert np.allclose(rf_detr_boxes, outputs_od["boxes"], rtol=tolerance, atol=tolerance)
            assert np.allclose(outputs_rfdetr.confidence, outputs_od["scores"], rtol=tolerance, atol=tolerance)
            assert rf_detr_classes == outputs_od["classes"].tolist()

    def test_warmup(self, imgs_coco, rf_model):
        obj_detector = ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=OD_MODEL,
            device=DEVICE,
            class_map=rf_model.class_names,
            image_size=[IMAGE_SIZE, IMAGE_SIZE],
        )
        obj_detector.warmup()

    def test_empty(self, imgs_coco, rf_model):
        object_detector = ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=OD_MODEL,
            device=DEVICE,
            class_map=rf_model.class_names,
            image_size=[IMAGE_SIZE, IMAGE_SIZE],
        )
        empty_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        outputs = object_detector.predict(empty_img, configs=0.5)
        assert len(outputs["boxes"]) == 0
        assert len(outputs["scores"]) == 0
        assert len(outputs["classes"]) == 0

    def test_confidence(self, imgs_coco, rf_model):
        object_detector = ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=OD_MODEL,
            device=DEVICE,
            class_map=rf_model.class_names,
            image_size=[IMAGE_SIZE, IMAGE_SIZE],
        )
        img = cv2.resize(imgs_coco[0], (IMAGE_SIZE, IMAGE_SIZE))
        outputs_05 = object_detector.predict(img, configs=1.0)
        assert len(outputs_05["boxes"]) == 0

    def test_operators(self, imgs_coco, rf_model):
        object_detector = ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=OD_MODEL.replace("cuda", "cpu"),
            device="cpu",
            class_map=rf_model.class_names,
            image_size=[IMAGE_SIZE, IMAGE_SIZE],
        )
        img = imgs_coco[0]
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        operators = [{"resize": [IMAGE_SIZE, IMAGE_SIZE, w, h]}]

        outputs_with_ops = object_detector.predict(img_resized, configs=0.5, operators=operators)
        outputs_without_ops = object_detector.predict(img_resized, configs=0.5)

        assert "boxes" in outputs_with_ops
        assert "scores" in outputs_with_ops
        assert "classes" in outputs_with_ops

        if len(outputs_with_ops["boxes"]) > 0:
            # boxes with operators should be scaled to the original image size
            assert np.all(outputs_with_ops["boxes"][:, 0] <= w)
            assert np.all(outputs_with_ops["boxes"][:, 1] <= h)
            assert np.all(outputs_with_ops["boxes"][:, 2] <= w)
            assert np.all(outputs_with_ops["boxes"][:, 3] <= h)

            # boxes without operators should be in the resized image space
            assert not np.allclose(outputs_with_ops["boxes"], outputs_without_ops["boxes"])

        annotated_image = object_detector.annotate_image(outputs_with_ops, img)
        out_name = "out_operators.jpg"
        os.makedirs(OUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUT_DIR, out_name), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
