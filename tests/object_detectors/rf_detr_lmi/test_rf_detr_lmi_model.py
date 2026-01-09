import logging
import os

import cv2
import numpy as np
import pytest
import torch
from od_core.object_detector import ObjectDetector
from rfdetr import RFDETRNano

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

COCO_DIR = "tests/assets/images/coco"
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
OD_MODEL = "tests/assets/models/od/rf_detr/checkpoint_best_regular.pth"


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


class Test_Rfdetr_Model:
    def test_compare_with_rfdetr(self, imgs_coco):
        obj_detector = ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=OD_MODEL,
            device=DEVICE,
            model_type="nano",
        )
        rf_model = RFDETRNano(pretrain_weights=OD_MODEL, device="cpu")
        rf_model.optimize_for_inference()

        for img in imgs_coco:
            outputs_od = obj_detector.predict(img, configs=0.5)
            outputs_rfdetr = rf_model.predict(img, threshold=0.5)
            assert "boxes" in outputs_od
            assert "scores" in outputs_od
            assert "classes" in outputs_od
            rf_detr_boxes = outputs_rfdetr.xyxy
            rf_detr_classes = [rf_model.class_names[c] for c in outputs_rfdetr.class_id]

            assert rf_detr_boxes.shape[0] == outputs_od["boxes"].shape[0]
            assert np.allclose(rf_detr_boxes, outputs_od["boxes"], rtol=1e-2, atol=1e-2)
            assert np.allclose(outputs_rfdetr.confidence, outputs_od["scores"], rtol=1e-2, atol=1e-2)
            assert rf_detr_classes == outputs_od["classes"].tolist()

    def test_warmup(self, imgs_coco):
        obj_detector = ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=OD_MODEL,
            device=DEVICE,
            model_type="nano",
        )
        obj_detector.warmup()

    def test_empty(self, imgs_coco):
        object_detector = ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=OD_MODEL,
            device=DEVICE,
            model_type="nano",
        )
        empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
        outputs = object_detector.predict(empty_img, configs=0.5)
        assert len(outputs["boxes"]) == 0
        assert len(outputs["scores"]) == 0
        assert len(outputs["classes"]) == 0

    def test_confidence(self, imgs_coco):
        object_detector = ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=OD_MODEL,
            device=DEVICE,
            model_type="nano",
        )
        img = imgs_coco[0]
        outputs_05 = object_detector.predict(img, configs=1.0)
        assert len(outputs_05["boxes"]) == 0
