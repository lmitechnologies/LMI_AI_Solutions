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

COCO_DIR = "tests/assets/images/coco"
MASKRCNN_MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
COCO_CLASSMAP = "tests/assets/models/od/detectron2/class_map.json"
MODEL_PATH = "tests/assets/models/od/detectron2/model.pt"
OG_WEIGHTS_PATH = "tests/assets/models/od/detectron2/model_final_f10217.pkl"
OUT_DIR = "tests/outputs/od/detectron2"
USE_CUDA = torch.cuda.is_available()
KEYS = ["boxes", "classes", "scores", "masks", "segments"]

with open(COCO_CLASSMAP, "r") as f:
    class_map = json.load(f)

logger = logging.getLogger(__name__)


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
        images.append(rgb)
    return images


@pytest.fixture(scope="module")
def og_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MASKRCNN_MODEL_CONFIG))
    cfg.MODEL.DEVICE = "cuda" if USE_CUDA else "cpu"
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


@pytest.fixture(scope="module", params=["detectron2_model", "detectron2_model_api"])
def model(request, detectron2_model, detectron2_model_api):
    return detectron2_model if request.param == "detectron2_model" else detectron2_model_api


def test_model(og_model, model, imgs_coco):
    confs = {v: 0.00 for v in class_map.values()}
    for image in imgs_coco:
        img = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        inputs = [{"image": img}]
        with torch.no_grad():
            orginal_preds = og_model.inference(inputs, do_postprocess=False)[0]

        preds = model.predict(image, confs=confs, process_masks=False)
        assert orginal_preds.pred_boxes.tensor.shape == preds.get("boxes")[0].shape
        assert orginal_preds.pred_classes.shape == preds.get("classes")[0].shape
        assert orginal_preds.scores.shape == preds.get("scores")[0].shape
        assert orginal_preds.pred_masks.shape == preds.get("masks")[0].shape

        # check if the scores are all close
        assert np.allclose(orginal_preds.scores.cpu().numpy(), preds.get("scores"))
        assert np.allclose(orginal_preds.pred_masks.cpu().numpy(), preds.get("masks"))


def test_operators(model, imgs_coco):
    confs = {v: 0.95 for v in class_map.values()}
    image = imgs_coco[0]
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, (512, 512))
    operators = [{"resize": [512, 512, w, h]}]
    outputs = model.predict(
        image_resized,
        confs=confs,
        return_segments=True,
        process_masks=True,
        operators=operators,
    )
    for key in KEYS:
        outputs[key] = outputs[key][0]

    assert len(outputs["boxes"]) == len(outputs["classes"]) == len(outputs["scores"]) == len(outputs["masks"]) == len(outputs["segments"])
    assert outputs["masks"].shape[1] == h
    assert outputs["masks"].shape[2] == w


def test_operators_no_masks(model, imgs_coco):
    confs = {v: 1.0 for v in class_map.values()}
    image = imgs_coco[0]
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, (512, 512))
    operators = [{"resize": [512, 512, w, h]}]
    outputs = model.predict(
        image_resized,
        confs=confs,
        return_segments=True,
        process_masks=True,
        operators=operators,
    )
    for key in KEYS:
        outputs[key] = outputs[key][0]

    assert len(outputs["boxes"]) == len(outputs["classes"]) == len(outputs["scores"]) == len(outputs["masks"]) == len(outputs["segments"])
    assert len(outputs["boxes"]) == 0


def test_batch_operators(model, imgs_coco):
    confs = {v: 0.8 for v in class_map.values()}
    images = imgs_coco
    original_sizes = [img.shape[:2] for img in images]
    images_resized = [cv2.resize(img, (512, 512)) for img in images]
    operators = [[{"resize": [512, 512, w, h]}] for h, w in original_sizes]
    outputs = model.predict(images_resized, confs=confs, process_masks=True, operators=operators)
    assert len(outputs["boxes"]) == len(images)
    assert len(outputs["scores"]) == len(images)
    assert len(outputs["classes"]) == len(images)
    assert len(outputs["masks"]) == len(images)
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, (h, w) in enumerate(original_sizes):
        assert len(outputs["boxes"][i]) == len(outputs["scores"][i]) == len(outputs["classes"][i]) == len(outputs["masks"][i])
        if len(outputs["masks"][i]) > 0:
            assert outputs["masks"][i].shape[1] == h
            assert outputs["masks"][i].shape[2] == w
        per_image = {k: v[i] for k, v in outputs.items()}
        annotated = model.annotate_image(per_image, images[i].copy())
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT_DIR, f"coco_{i}_batch_operators.jpg"), bgr)
