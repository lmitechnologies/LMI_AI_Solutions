import logging
import os
import platform

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
IS_ARM = platform.machine().lower().startswith(("arm", "aarch"))
TOLERANCE = 1e-2 if IS_ARM else 1e-4


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


@pytest.fixture(scope="module")
def obj_detector(rf_model):
    obj_detector = ObjectDetector(
        metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
        model_path=OD_MODEL,
        device=DEVICE,
        class_map=rf_model.class_names,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
    )
    return obj_detector


@pytest.fixture(scope="module")
def cpu_models(rf_model):
    od_pt = ObjectDetector(
        metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
        model_path=OD_MODEL.replace("cuda", "cpu"),
        device="cpu",
        class_map=rf_model.class_names,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
    )

    od_pth = ObjectDetector(
        metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
        model_path=PTH_FILE,
        model_type="nano",
        device="cpu",
        class_map=rf_model.class_names,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
    )
    return od_pt, od_pth


class Test_Rfdetr_Model:
    def test_compare_with_rfdetr(self, imgs_coco, cpu_models, tolerance=TOLERANCE):
        "Use cpu to avoid gpu non-determinism issues."

        def compare_results(rf_preds, outputs, model_name):
            rf_boxes = rf_preds.xyxy
            rf_classes = [rf_model.class_names[c] for c in rf_preds.class_id]

            assert rf_boxes.shape[0] == outputs["boxes"].shape[0], f"{model_name}: Number of boxes mismatch"
            assert np.allclose(rf_boxes, outputs["boxes"], rtol=tolerance, atol=tolerance), f"{model_name}: Box coordinates mismatch"
            assert np.allclose(rf_preds.confidence, outputs["scores"], rtol=tolerance, atol=tolerance), (
                f"{model_name}: Confidence scores mismatch"
            )
            assert rf_classes == outputs["classes"].tolist(), f"{model_name}: Class labels mismatch"

        rf_model = RFDETRNano(pretrain_weights=PTH_FILE, device="cpu")
        pt_model, pth_model = cpu_models

        resized = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in imgs_coco]

        # rfdetr batch predict returns a list of Detections
        rf_batch_preds = rf_model.predict(resized, threshold=0.5)
        if not isinstance(rf_batch_preds, list):
            rf_batch_preds = [rf_batch_preds]

        batch_pt, _ = pt_model.predict(resized, configs=0.5)
        batch_pth, _ = pth_model.predict(resized, configs=0.5)

        assert len(rf_batch_preds) == len(resized)
        assert len(batch_pt["boxes"]) == len(resized)
        assert len(batch_pth["boxes"]) == len(resized)

        keys = ("boxes", "scores", "classes")
        for i, rf_preds in enumerate(rf_batch_preds):
            outputs_pt = {k: batch_pt[k][i] for k in keys}
            outputs_pth = {k: batch_pth[k][i] for k in keys}

            compare_results(rf_preds, outputs_pt, "pt_model")
            compare_results(rf_preds, outputs_pth, "pth_model")

    def test_warmup(self, obj_detector):
        obj_detector.warmup()

    def test_empty(self, obj_detector):
        empty_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        batch_outputs, _ = obj_detector.predict(empty_img, configs=0.5)
        outputs = {k: v[0] for k, v in batch_outputs.items()}
        assert len(outputs["boxes"]) == 0
        assert len(outputs["scores"]) == 0
        assert len(outputs["classes"]) == 0

    def test_confidence(self, imgs_coco, obj_detector):
        img = cv2.resize(imgs_coco[0], (IMAGE_SIZE, IMAGE_SIZE))
        batch_outputs, _ = obj_detector.predict(img, configs=1.0)
        outputs = {k: v[0] for k, v in batch_outputs.items()}
        assert len(outputs["boxes"]) == 0

    def test_operators(self, imgs_coco, obj_detector):
        for idx, img in enumerate(imgs_coco):
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            operators = [{"resize": [IMAGE_SIZE, IMAGE_SIZE, w, h]}]

            batch_outputs, _ = obj_detector.predict(img_resized, configs=0.5, operators=operators)
            outputs = {k: v[0] for k, v in batch_outputs.items()}
            assert len(outputs["boxes"]) > 0, "Expected detections with operators, but got none."

            if len(outputs["boxes"]) > 0:
                # boxes with operators should be scaled to the original image size
                assert np.all(outputs["boxes"][:, 0] <= w)
                assert np.all(outputs["boxes"][:, 1] <= h)
                assert np.all(outputs["boxes"][:, 2] <= w)
                assert np.all(outputs["boxes"][:, 3] <= h)

            annotated_image = obj_detector.annotate_image(outputs, img)
            out_name = f"out_operators_{idx}.jpg"
            os.makedirs(OUT_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(OUT_DIR, out_name), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    def test_operators_batch(self, imgs_coco, obj_detector):
        original_sizes = [(img.shape[1], img.shape[0]) for img in imgs_coco]  # (w, h)
        operators = [[{"resize": [IMAGE_SIZE, IMAGE_SIZE, w, h]}] for w, h in original_sizes]
        imgs_resized = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in imgs_coco]

        batch_outputs, _ = obj_detector.predict(imgs_resized, configs=0.5, operators=operators)

        assert len(batch_outputs["boxes"]) == len(imgs_coco)

        os.makedirs(OUT_DIR, exist_ok=True)
        for idx, img in enumerate(imgs_coco):
            w, h = original_sizes[idx]
            outputs = {k: v[idx] for k, v in batch_outputs.items()}
            assert len(outputs["boxes"]) > 0, f"Expected detections for image {idx}, but got none."

            # boxes with operators should be scaled to the original image size
            assert np.all(outputs["boxes"][:, 0] <= w)
            assert np.all(outputs["boxes"][:, 1] <= h)
            assert np.all(outputs["boxes"][:, 2] <= w)
            assert np.all(outputs["boxes"][:, 3] <= h)

            annotated_image = obj_detector.annotate_image(outputs, img)
            out_name = f"out_operators_batch_{idx}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, out_name), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
