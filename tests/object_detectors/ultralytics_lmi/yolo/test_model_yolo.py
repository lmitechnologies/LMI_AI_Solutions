import logging
import os

import cv2
import numpy as np
import pytest
import torch
from od_core.object_detector import ObjectDetector
from ultralytics import YOLO
from ultralytics_lmi.yolo.model import Yolo, YoloObb, YoloPose, YoloSeg

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


COCO_DIR = "tests/assets/images/coco"
DOTA8_DIR = "tests/assets/images/dota8"
DOTA_DIR = "tests/assets/images/dota"
OUT_DIR = "tests/outputs/od/ultralytics/yolo"
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

OD_DET_MODELS = ["tests/assets/models/od/yolo11n.pt", "tests/assets/models/od/yolov8n.pt"]

OD_SEG_MODELS = [
    "tests/assets/models/od/yolo11n-seg.pt",
    "tests/assets/models/od/yolov8n-seg.pt",
]

OD_OBB_DOTA_8 = [
    "tests/assets/models/od/yolo11n-obb.pt",
]

OD_OBB_DOTA = [
    "tests/assets/models/od/yolov8n-obb.pt",
]

OD_POSE_MODELS = [
    "tests/assets/models/od/yolo11n-pose.pt",
    "tests/assets/models/od/yolov8n-pose.pt",
]


@pytest.fixture(scope="module")
def yolo_models():
    models = {}
    keys = ["det", "seg", "obb_dota", "obb_dota8", "pose"]
    model_lists = [
        OD_DET_MODELS,
        OD_SEG_MODELS,
        OD_OBB_DOTA,
        OD_OBB_DOTA_8,
        OD_POSE_MODELS,
    ]
    model_classes = [Yolo, YoloSeg, YoloObb, YoloObb, YoloPose]
    for k, ml, mc in zip(keys, model_lists, model_classes):
        models[k] = [mc(model, device=DEVICE) for model in ml]
    return models


@pytest.fixture(scope="module")
def yolo_models_api():
    models = {}
    keys = ["det", "seg", "obb_dota", "obb_dota8", "pose"]
    tasks = ["od", "seg", "obb", "obb", "pose"]
    model_lists = [
        OD_DET_MODELS,
        OD_SEG_MODELS,
        OD_OBB_DOTA,
        OD_OBB_DOTA_8,
        OD_POSE_MODELS,
    ]
    for k, ml, task in zip(keys, model_lists, tasks):
        models[k] = [
            ObjectDetector(
                metadata=dict(
                    version="v1",
                    model_name="yolov8" if "yolov8n" in model else "yolov11",
                    task=task,
                    framework="ultralytics",
                    model_path=model,
                    image_size=[640, 640],
                ),
                device=DEVICE,
            )
            for model in ml
        ]
    return models


def load_image(path):
    im = cv2.imread(path)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb


@pytest.fixture(scope="module")
def imgs_coco():
    im_dim = 640
    paths = [os.path.join(COCO_DIR, img) for img in os.listdir(COCO_DIR)]
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


@pytest.fixture(scope="module")
def imgs_dota():
    im_dim = 1024
    paths = [os.path.join(DOTA_DIR, img) for img in os.listdir(DOTA_DIR)]
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


@pytest.fixture(scope="module")
def imgs_dota8():
    im_dim = 1024
    paths = [os.path.join(DOTA8_DIR, img) for img in os.listdir(DOTA8_DIR)]
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
    def test_compare_with_ultralytics(self, imgs_coco):
        device = "cpu"
        for model_path in OD_DET_MODELS:
            ults_model = YOLO(model_path)
            our_model = Yolo(model_path, device=device)
            for _img, resized, _op in zip(*imgs_coco):
                resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                results = ults_model(
                    resized_bgr,
                    conf=0.5,
                    iou=0.4,
                    max_det=300,
                    device=device,
                )
                ults_out = results[0].cpu().numpy()

                out, time_info = our_model.predict(resized, configs=0.5, iou=0.4, max_det=300)
                assert np.array_equal(np.array(out["boxes"]), ults_out.boxes.xyxy)
                assert np.array_equal(np.array(out["scores"]), ults_out.boxes.conf)

    def test_warmup(self, yolo_models, yolo_models_api):
        for model in yolo_models["det"] + yolo_models_api["det"]:
            model.warmup()

    def test_predict_empty(self, yolo_models, yolo_models_api):
        for model in yolo_models["det"] + yolo_models_api["det"]:
            out, time_info = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), configs=0.5)
            assert len(out["boxes"]) == 0 and len(out["scores"]) == 0

    def test_predict(self, yolo_models, yolo_models_api, imgs_coco):
        i = 0
        for model in yolo_models["det"] + yolo_models_api["det"]:
            for img, resized, op in zip(*imgs_coco):
                out, time_info = model.predict(resized, configs=0.5, operators=op)
                assert len(out["boxes"]) > 0
                for sc in out["scores"]:
                    assert sc >= 0.5
                im_out = model.annotate_image(out, img)

                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out, time_info = model.predict(resized, configs=0.5, operators=op)
                    for b, sc in zip(out["boxes"], out["scores"]):
                        assert b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(OUT_DIR, f"det-{i}.png"), im_out)
                i += 1


class Test_Yolo_Seg:
    def test_compare_with_ultralytics(self, imgs_coco):
        device = "cpu"
        for model_path in OD_SEG_MODELS:
            ults_model = YOLO(model_path)
            our_model = YoloSeg(model_path, device=device)
            for _img, resized, _op in zip(*imgs_coco):
                resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                results = ults_model(
                    resized_bgr,
                    conf=0.5,
                    iou=0.4,
                    max_det=300,
                    retina_masks=True,
                    device=device,
                )
                ults_out = results[0].cpu().numpy()

                out, time_info = our_model.predict(resized, configs=0.5, iou=0.4, max_det=300)
                assert np.array_equal(np.array(out["boxes"]), ults_out.boxes.xyxy)
                assert np.array_equal(np.array(out["scores"]), ults_out.boxes.conf)
                assert np.array_equal(np.array(out["masks"]), ults_out.masks.data)
                assert len(out["segments"]) == len(results[0].masks.xy)
                for s1, s2 in zip(out["segments"], results[0].masks.xy):
                    assert np.array_equal(s1, s2)

    def test_warmup(self, yolo_models, yolo_models_api):
        for model in yolo_models["seg"] + yolo_models_api["seg"]:
            model.warmup()

    def test_predict_empty(self, yolo_models, yolo_models_api):
        for model in yolo_models["seg"] + yolo_models_api["seg"]:
            out, time_info = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), configs=0.5)
            assert len(out["boxes"]) == 0 and len(out["masks"]) == 0 and len(out["segments"]) == 0 and len(out["scores"]) == 0

    def test_predict(self, yolo_models, yolo_models_api, imgs_coco):
        i = 0
        for model in yolo_models["seg"] + yolo_models_api["seg"]:
            for img, resized, op in zip(*imgs_coco):
                out, time_info = model.predict(resized, configs=0.5, operators=op)
                assert len(out["masks"]) > 0 and len(out["segments"]) > 0
                for sc in out["scores"]:
                    assert sc >= 0.5
                im_out = model.annotate_image(out, img)

                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out, time_info = model.predict(resized, configs=0.5, operators=op)
                    for seg, m, b, sc in zip(out["segments"], out["masks"], out["boxes"], out["scores"]):
                        assert seg.is_cuda and m.is_cuda and b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f"seg-{i}.png"), im_out)
                i += 1


class Test_Yolo_Obb:
    def compare_with_ultralytics(self, imgs, model_paths):
        device = "cpu"
        for model_path in model_paths:
            ults_model = YOLO(model_path)
            our_model = YoloObb(model_path, device=device)
            for _img, resized, _op in zip(*imgs):
                resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                results = ults_model(
                    resized_bgr,
                    conf=0.5,
                    iou=0.4,
                    max_det=300,
                    device=device,
                )
                ults_out = results[0].cpu().numpy()

                out, time_info = our_model.predict(resized, configs=0.5, iou=0.4, max_det=300)
                assert np.allclose(np.array(out["boxes"]), ults_out.obb.xyxyxyxy, atol=1e-5)  # for floating point precision issue
                assert np.array_equal(np.array(out["scores"]), ults_out.obb.conf)

    def test_compare_with_ultralytics_dota8(self, imgs_dota8):
        self.compare_with_ultralytics(imgs_dota8, OD_OBB_DOTA_8)

    def test_compare_with_ultralytics_dota(self, imgs_dota):
        self.compare_with_ultralytics(imgs_dota, OD_OBB_DOTA)

    def test_warmup_dota8(self, yolo_models, yolo_models_api):
        for model in yolo_models["obb_dota8"] + yolo_models_api["obb_dota8"]:
            model.warmup()

    def test_warmup_dota(self, yolo_models, yolo_models_api):
        for model in yolo_models["obb_dota"] + yolo_models_api["obb_dota"]:
            model.warmup()

    def test_predict_empty(self, yolo_models, yolo_models_api):
        for model in yolo_models["obb_dota8"] + yolo_models["obb_dota"] + yolo_models_api["obb_dota8"] + yolo_models_api["obb_dota"]:
            out, time_info = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), configs=0.5)
            assert len(out["boxes"]) == 0 and len(out["scores"]) == 0

    def test_predict_dota8(self, yolo_models, yolo_models_api, imgs_dota8):
        i = 0
        for model in yolo_models["obb_dota8"] + yolo_models_api["obb_dota8"]:
            for img, resized, op in zip(*imgs_dota8):
                out, time_info = model.predict(resized, configs=0.5, operators=op)
                assert len(out["boxes"]) > 0
                for sc in out["scores"]:
                    assert sc >= 0.5
                im_out = model.annotate_image(out, img)

                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out, time_info = model.predict(resized, configs=0.5, operators=op)
                    for b, sc in zip(out["boxes"], out["scores"]):
                        assert b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f"obb-{i}.png"), im_out)
                i += 1

    def test_predict_dota(self, yolo_models, yolo_models_api, imgs_dota):
        i = 0
        for model in yolo_models["obb_dota"] + yolo_models_api["obb_dota"]:
            for img, resized, op in zip(*imgs_dota):
                out, time_info = model.predict(resized, configs=0.5, operators=op)
                assert len(out["boxes"]) > 0
                for sc in out["scores"]:
                    assert sc >= 0.5
                im_out = model.annotate_image(out, img)

                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out, time_info = model.predict(resized, configs=0.5, operators=op)
                    for b, sc in zip(out["boxes"], out["scores"]):
                        assert b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f"obb-{i}.png"), im_out)
                i += 1


class Test_Yolo_Pose:
    def test_compare_with_ultralytics(self, imgs_coco):
        device = "cpu"
        for model_path in OD_POSE_MODELS:
            ults_model = YOLO(model_path)
            our_model = YoloPose(model_path, device=device)
            for _img, resized, _op in zip(*imgs_coco):
                resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                results = ults_model(
                    resized_bgr,
                    conf=0.5,
                    iou=0.4,
                    max_det=300,
                    device=device,
                )
                ults_out = results[0].cpu().numpy()

                out, time_info = our_model.predict(resized, configs=0.5, iou=0.4, max_det=300)
                assert np.array_equal(np.array(out["boxes"]), ults_out.boxes.xyxy)
                assert np.array_equal(np.array(out["scores"]), ults_out.boxes.conf)
                assert np.array_equal(np.array(out["points"]), ults_out.keypoints.data)

    def test_warmup(self, yolo_models, yolo_models_api):
        for model in yolo_models["pose"] + yolo_models_api["pose"]:
            model.warmup()

    def test_predict_empty(self, yolo_models, yolo_models_api):
        for model in yolo_models["pose"] + yolo_models_api["pose"]:
            out, time_info = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), configs=0.5)
            assert len(out["boxes"]) == 0 and len(out["points"]) == 0 and len(out["scores"]) == 0

    def test_predict(self, yolo_models, yolo_models_api, imgs_coco):
        i = 0
        for model in yolo_models["pose"] + yolo_models_api["pose"]:
            for img, resized, op in zip(*imgs_coco):
                out, time_info = model.predict(resized, configs=0.5, operators=op)
                assert len(out["boxes"]) > 0
                for sc in out["scores"]:
                    assert sc >= 0.5
                im_out = model.annotate_image(out, img)

                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out, time_info = model.predict(resized, configs=0.5, operators=op)
                    for b, sc, kp in zip(out["boxes"], out["scores"], out["points"]):
                        assert b.is_cuda and sc.is_cuda and kp.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f"pose-{i}.png"), im_out)
                i += 1
