import logging
import os

import cv2
import pytest

from classifiers.cls_core.classifier import Classifier
from classifiers.ultralytics_lmi.yolo.model import YoloCls

logger = logging.getLogger(__name__)


IMG_DIR = "tests/assets/images/coco"
OUT_DIR = "tests/outputs/cls/yolov8"
MODEL_SZ = 224

CLS_MODELS = [
    "tests/assets/models/cls/yolo26n-cls.pt",
    "tests/assets/models/cls/yolo11n-cls.pt",
]


@pytest.fixture
def model_det():
    return [YoloCls(model) for model in CLS_MODELS]


@pytest.fixture
def model_det_api():
    return [
        Classifier(
            metadata=dict(
                version="v1",
                model_name="yolov8",
                task="classification",
                framework="ultralytics",
                model_path=model,
                image_size=[MODEL_SZ, MODEL_SZ],
            )
        )
        for model in CLS_MODELS
    ]


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


class Test_Yolo_Det:
    def test_warmup(self, model_det):
        for model in model_det:
            model.warmup()

    def test_predict(self, model_det, imgs_coco):
        i = 0
        for model in model_det:
            for img, resized, _op in zip(*imgs_coco):
                out, time_info = model.predict(resized)
                assert len(out["classes"]) > 0
                for sc in out["scores"]:
                    assert sc > 0
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


class Test_Yolo_Det_API:
    def test_warmup(self, model_det_api):
        for model in model_det_api:
            model.warmup()

    def test_predict(self, model_det_api, imgs_coco):
        i = 0
        for model in model_det_api:
            for img, resized, _op in zip(*imgs_coco):
                out, time_info = model.predict(resized)
                assert len(out["classes"]) > 0
                for sc in out["scores"]:
                    assert sc > 0

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
