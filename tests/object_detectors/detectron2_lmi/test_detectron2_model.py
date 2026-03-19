import json
import logging
import os

import cv2
import numpy as np
import pytest
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from object_detectors.detectron2_lmi.model import Detectron2Model
from object_detectors.od_core.object_detector import ObjectDetector

MASKRCNN_MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
COCO_CLASSMAP = "tests/assets/models/od/detectron2/class_map.json"
MODEL_PATH = "tests/assets/models/od/detectron2/model.pt"
OG_WEIGHTS_PATH = "tests/assets/models/od/detectron2/model_final_f10217.pkl"
SAMPLE_IMAGE = "tests/assets/images/detectron2/sample_image.jpg"
OUT_DIR = "tests/outputs/od/detectron2"

with open(COCO_CLASSMAP, "r") as f:
    class_map = json.load(f)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture(scope="module")
def og_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MASKRCNN_MODEL_CONFIG))
    model = build_model(cfg)
    DetectionCheckpointer(model).load(OG_WEIGHTS_PATH)
    model.eval()
    return model


@pytest.fixture(scope="module")
def detectron2_model():
    model = Detectron2Model(MODEL_PATH, class_map=class_map)
    return model


@pytest.fixture(scope="module")
def detectron2_model_api():
    model = ObjectDetector(
        metadata=dict(version="v0", model_name="mask_rcnn", task="seg", framework="detectron2"),
        model_path=MODEL_PATH,
        class_map=class_map,
    )
    return model


class TestDetectron2ModelPT:
    def test_model(self, og_model, detectron2_model):
        img = torch.as_tensor(cv2.imread(SAMPLE_IMAGE).transpose(2, 0, 1).astype("float32"))
        inputs = [{"image": img}]
        with torch.no_grad():
            orginal_preds = og_model.inference(inputs, do_postprocess=False)[0]

        confs = {v: 0.00 for k, v in class_map.items()}
        image = cv2.imread(SAMPLE_IMAGE)
        preds = detectron2_model.predict(image, confs=confs, process_masks=False)
        assert orginal_preds.pred_boxes.tensor.shape == preds.get("boxes")[0].shape
        assert orginal_preds.pred_classes.shape == preds.get("classes")[0].shape
        assert orginal_preds.scores.shape == preds.get("scores")[0].shape
        assert orginal_preds.pred_masks.shape == preds.get("masks")[0].shape

        # check if the scores are all close
        assert np.allclose(orginal_preds.scores.cpu().numpy(), preds.get("scores"))
        assert np.allclose(orginal_preds.pred_masks.cpu().numpy(), preds.get("masks"))

    def test_annotations(self, detectron2_model):
        confs = {v: 0.95 for k, v in class_map.items()}
        image = cv2.imread(SAMPLE_IMAGE)
        outputs = detectron2_model.predict(image, confs=confs, return_segments=True, process_masks=True)
        outputs["boxes"] = outputs["boxes"][0]
        outputs["classes"] = outputs["classes"][0]
        outputs["scores"] = outputs["scores"][0]
        outputs["masks"] = outputs["masks"][0]
        outputs["segments"] = outputs["segments"][0]

        assert (
            len(outputs["boxes"]) == len(outputs["classes"]) == len(outputs["scores"]) == len(outputs["masks"]) == len(outputs["segments"])
        )

        annotated_image = detectron2_model.annotate_image(outputs, image, show_segments=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        out_name = os.path.basename(SAMPLE_IMAGE).split(".")[0] + "_raw.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), annotated_image)

    def test_operators(self, detectron2_model):
        confs = {v: 0.95 for k, v in class_map.items()}
        image = cv2.imread(SAMPLE_IMAGE)
        h, w = image.shape[:2]
        image2 = cv2.resize(image, (512, 512))
        operators = [{"resize": [512, 512, w, h]}]
        outputs = detectron2_model.predict(
            image2,
            confs=confs,
            return_segments=True,
            process_masks=True,
            operators=operators,
        )
        outputs["boxes"] = outputs["boxes"][0]
        outputs["classes"] = outputs["classes"][0]
        outputs["scores"] = outputs["scores"][0]
        outputs["masks"] = outputs["masks"][0]
        outputs["segments"] = outputs["segments"][0]
        assert (
            len(outputs["boxes"]) == len(outputs["classes"]) == len(outputs["scores"]) == len(outputs["masks"]) == len(outputs["segments"])
        )
        assert outputs["masks"].shape[1] == h
        assert outputs["masks"].shape[2] == w

        annotated_image = detectron2_model.annotate_image(outputs, image, show_segments=True)
        out_name = os.path.basename(SAMPLE_IMAGE).split(".")[0] + "_operators.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), annotated_image)

    def test_operators_no_masks(self, detectron2_model: Detectron2Model):
        confs = {v: 1.0 for k, v in class_map.items()}
        image = cv2.imread(SAMPLE_IMAGE)
        h, w = image.shape[:2]
        image2 = cv2.resize(image, (512, 512))
        operators = [{"resize": [512, 512, w, h]}]
        outputs = detectron2_model.predict(
            image2,
            confs=confs,
            return_segments=True,
            process_masks=True,
            operators=operators,
        )
        outputs["boxes"] = outputs["boxes"][0]
        outputs["classes"] = outputs["classes"][0]
        outputs["scores"] = outputs["scores"][0]
        outputs["masks"] = outputs["masks"][0]
        outputs["segments"] = outputs["segments"][0]
        assert (
            len(outputs["boxes"]) == len(outputs["classes"]) == len(outputs["scores"]) == len(outputs["masks"]) == len(outputs["segments"])
        )
        assert len(outputs["boxes"]) == 0


class TestDetectron2ModelPT_API:
    def test_model(self, og_model, detectron2_model_api):
        img = torch.as_tensor(cv2.imread(SAMPLE_IMAGE).transpose(2, 0, 1).astype("float32"))
        inputs = [{"image": img}]
        with torch.no_grad():
            orginal_preds = og_model.inference(inputs, do_postprocess=False)[0]

        confs = {v: 0.00 for k, v in class_map.items()}

        image = cv2.imread(SAMPLE_IMAGE)
        preds = detectron2_model_api.predict(image, confs=confs, process_masks=False)
        assert orginal_preds.pred_boxes.tensor.shape == preds.get("boxes")[0].shape
        assert orginal_preds.pred_classes.shape == preds.get("classes")[0].shape
        assert orginal_preds.scores.shape == preds.get("scores")[0].shape
        assert orginal_preds.pred_masks.shape == preds.get("masks")[0].shape

        # check if the scores are all close
        assert np.allclose(orginal_preds.scores.cpu().numpy(), preds.get("scores"))
        assert np.allclose(orginal_preds.pred_masks.cpu().numpy(), preds.get("masks"))

    def test_annotations(self, detectron2_model_api):
        confs = {v: 0.95 for k, v in class_map.items()}

        image = cv2.imread(SAMPLE_IMAGE)
        outputs = detectron2_model_api.predict(image, confs=confs, return_segments=True, process_masks=True)
        outputs["boxes"] = outputs["boxes"][0]
        outputs["classes"] = outputs["classes"][0]
        outputs["scores"] = outputs["scores"][0]
        outputs["masks"] = outputs["masks"][0]
        outputs["segments"] = outputs["segments"][0]

        assert (
            len(outputs["boxes"]) == len(outputs["classes"]) == len(outputs["scores"]) == len(outputs["masks"]) == len(outputs["segments"])
        )

        annotated_image = detectron2_model_api.annotate_image(outputs, image, show_segments=True)
        out_name = os.path.basename(SAMPLE_IMAGE).split(".")[0] + "_raw_api.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), annotated_image)

    def test_operators(self, detectron2_model_api):
        confs = {v: 0.95 for k, v in class_map.items()}

        image = cv2.imread(SAMPLE_IMAGE)
        h, w = image.shape[:2]
        image2 = cv2.resize(image, (512, 512))
        operators = [{"resize": [512, 512, w, h]}]
        outputs = detectron2_model_api.predict(
            image2,
            confs=confs,
            return_segments=True,
            process_masks=True,
            operators=operators,
        )
        outputs["boxes"] = outputs["boxes"][0]
        outputs["classes"] = outputs["classes"][0]
        outputs["scores"] = outputs["scores"][0]
        outputs["masks"] = outputs["masks"][0]
        outputs["segments"] = outputs["segments"][0]
        assert (
            len(outputs["boxes"]) == len(outputs["classes"]) == len(outputs["scores"]) == len(outputs["masks"]) == len(outputs["segments"])
        )
        assert outputs["masks"].shape[1] == h
        assert outputs["masks"].shape[2] == w

        annotated_image = detectron2_model_api.annotate_image(outputs, image, show_segments=True)
        out_name = os.path.basename(SAMPLE_IMAGE).split(".")[0] + "_operators_api.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), annotated_image)

    def test_operators_no_masks(self, detectron2_model_api):
        confs = {v: 1.0 for k, v in class_map.items()}

        image = cv2.imread(SAMPLE_IMAGE)
        h, w = image.shape[:2]
        image2 = cv2.resize(image, (512, 512))
        operators = [{"resize": [512, 512, w, h]}]
        outputs = detectron2_model_api.predict(
            image2,
            confs=confs,
            return_segments=True,
            process_masks=True,
            operators=operators,
        )
        outputs["boxes"] = outputs["boxes"][0]
        outputs["classes"] = outputs["classes"][0]
        outputs["scores"] = outputs["scores"][0]
        outputs["masks"] = outputs["masks"][0]
        outputs["segments"] = outputs["segments"][0]
        assert (
            len(outputs["boxes"]) == len(outputs["classes"]) == len(outputs["scores"]) == len(outputs["masks"]) == len(outputs["segments"])
        )
        assert len(outputs["boxes"]) == 0
