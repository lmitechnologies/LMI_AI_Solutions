import logging
import os

import cv2
import pytest
import torch

from classifiers.cls_core.classifier import Classifier
from classifiers.ultralytics_lmi.yolo.model import YoloCls

logger = logging.getLogger(__name__)


IMG_DIR = "tests/assets/images/coco"
OUT_DIR = "tests/outputs/cls/yolov8"
MODEL_SZ = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLS_MODELS = [
    ("tests/assets/models/cls/yolo26n-cls.pt", "yolov26"),
    ("tests/assets/models/cls/yolo11n-cls.pt", "yolov11"),
]


@pytest.fixture
def model_det():
    return [YoloCls(model_path, device=DEVICE, image_size=[MODEL_SZ, MODEL_SZ]) for model_path, _ in CLS_MODELS]


def _make_api_model(model_path, model_name):
    return Classifier(
        metadata=dict(
            version="v1",
            model_name=model_name,
            task="classification",
            framework="ultralytics",
            model_path=model_path,
            image_size=[MODEL_SZ, MODEL_SZ],
        ),
        device=DEVICE,
    )


def load_image(path):
    im = cv2.imread(path)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb


@pytest.fixture
def imgs_coco():
    im_dim = MODEL_SZ
    paths = [os.path.join(IMG_DIR, img) for img in os.listdir(IMG_DIR)]
    images = []
    resized_images = []
    ops = []
    for p in paths:
        if "png" not in p and "jpg" not in p:
            continue
        rgb = load_image(p)
        h, w = rgb.shape[:2]
        im2 = cv2.resize(rgb, (im_dim, im_dim))
        images.append(rgb)
        resized_images.append(im2)
        ops.append([{"resize": (im_dim, im_dim, w, h)}])
    return images, resized_images, ops


def _assert_results(out, n):
    assert len(out["classes"]) == n
    assert len(out["scores"]) == n
    for sc in out["scores"]:
        assert sc > 0


def test_model_class_comparison():
    for model_path, model_name in CLS_MODELS:
        direct = YoloCls(model_path, device=DEVICE, image_size=[MODEL_SZ, MODEL_SZ])
        api = _make_api_model(model_path, model_name)
        assert type(direct) is type(api), f"direct={type(direct).__name__}, api={type(api).__name__}"


def test_warmup(model_det):
    for model in model_det:
        model.warmup()


def test_predict(model_det, imgs_coco):
    i = 0
    for model in model_det:
        for img, resized, _op in zip(*imgs_coco):
            out, time_info = model.predict(resized)
            _assert_results(out, 1)
            label = f"{out['classes'][0]}:{out['scores'][0]:.2f}"
            im_out = cv2.putText(
                img.copy(),
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            os.makedirs(OUT_DIR, exist_ok=True)
            im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(OUT_DIR, f"cls-{i}.png"), im_out)
            i += 1


def test_predict_batch(model_det, imgs_coco):
    _, resized_images, _ = imgs_coco
    for model in model_det:
        out, time_info = model.predict(resized_images)
        _assert_results(out, len(resized_images))


def test_predict_batch_size(model_det, imgs_coco):
    _, resized_images, _ = imgs_coco
    for model in model_det:
        out, time_info = model.predict(resized_images, batch_size=2)
        _assert_results(out, len(resized_images))
