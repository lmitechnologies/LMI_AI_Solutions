import logging
import os

import cv2
import numpy as np
import pytest
import torch
from ultralytics import YOLO

from object_detectors.od_core.object_detector import ObjectDetector
from object_detectors.ultralytics_lmi.yolo.model import Yolo, YoloObb, YoloPose, YoloSeg

logger = logging.getLogger(__name__)


COCO_DIR = "tests/assets/images/coco"
DOTA8_DIR = "tests/assets/images/dota8"
OUT_DIR = "tests/outputs/od/ultralytics/yolo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = [640, 640]
OBB_IMGSZ = [1024, 1024]
OFF_SIZES = [(500, 661), (576, 704), (704, 512)]  # (h, w), non-square

OD_DET_MODELS = [
    "tests/assets/models/od/ultralytics/yolo26n.pt",
    "tests/assets/models/od/ultralytics/yolo11n.pt",
    "tests/assets/models/od/ultralytics/yolov8n.pt",
]

OD_SEG_MODELS = [
    "tests/assets/models/od/ultralytics/yolo26n-seg.pt",
    "tests/assets/models/od/ultralytics/yolo11n-seg.pt",
    "tests/assets/models/od/ultralytics/yolov8n-seg.pt",
]

OD_OBB_DOTA_8 = [
    "tests/assets/models/od/ultralytics/yolo26n-obb.pt",
    "tests/assets/models/od/ultralytics/yolo11n-obb.pt",
    "tests/assets/models/od/ultralytics/yolov8n-obb.pt",
]

OD_POSE_MODELS = [
    "tests/assets/models/od/ultralytics/yolo26n-pose.pt",
    "tests/assets/models/od/ultralytics/yolo11n-pose.pt",
    "tests/assets/models/od/ultralytics/yolov8n-pose.pt",
]


def _model_name(path):
    return os.path.splitext(os.path.basename(path))[0]


@pytest.fixture(scope="module")
def yolo_models():
    models = {}
    keys = ["det", "seg", "obb_dota8", "pose"]
    model_lists = [
        OD_DET_MODELS,
        OD_SEG_MODELS,
        OD_OBB_DOTA_8,
        OD_POSE_MODELS,
    ]
    model_classes = [Yolo, YoloSeg, YoloObb, YoloPose]
    image_sizes = [IMGSZ, IMGSZ, OBB_IMGSZ, IMGSZ]
    for k, ml, mc, imsz in zip(keys, model_lists, model_classes, image_sizes):
        instances = []
        for path in ml:
            m = mc(path, device=DEVICE, image_size=imsz)
            m.test_name = _model_name(path)
            instances.append(m)
        models[k] = instances
    return models


@pytest.fixture(scope="module")
def yolo_models_api():
    models = {}
    keys = ["det", "seg", "obb_dota8", "pose"]
    tasks = ["od", "seg", "obb", "pose"]
    model_lists = [
        OD_DET_MODELS,
        OD_SEG_MODELS,
        OD_OBB_DOTA_8,
        OD_POSE_MODELS,
    ]
    image_sizes = [IMGSZ, IMGSZ, OBB_IMGSZ, IMGSZ]
    for k, ml, task, imsz in zip(keys, model_lists, tasks, image_sizes):
        instances = []
        for path in ml:
            m = ObjectDetector(
                metadata=dict(
                    version="v1",
                    model_name="yolov8" if "yolov8n" in path else "yolov11",
                    task=task,
                    framework="ultralytics",
                    model_path=path,
                    image_size=imsz,
                ),
                device=DEVICE,
            )
            m.test_name = _model_name(path)
            instances.append(m)
        models[k] = instances
    return models


@pytest.fixture(scope="module")
def all_models(yolo_models_api):
    return yolo_models_api


def test_model_class_comparison(yolo_models, yolo_models_api):
    for key in yolo_models:
        for d, a in zip(yolo_models[key], yolo_models_api[key]):
            assert type(d) is type(a), f"{key} [{d.test_name}]: direct={type(d).__name__}, api={type(a).__name__}"


def load_image(path):
    im = cv2.imread(path)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb


@pytest.fixture(scope="module")
def imgs_coco():
    return _load_images(COCO_DIR, 640)


@pytest.fixture(scope="module")
def imgs_dota8():
    return _load_images(DOTA8_DIR, 1024)


def _load_images(directory, im_dim):
    """Load and resize images from a directory, returning (images, resized, ops)."""
    paths = [os.path.join(directory, img) for img in os.listdir(directory)]
    images, resized_images, ops = [], [], []
    for p in paths:
        if "png" not in p and "jpg" not in p:
            continue
        rgb = load_image(p)
        h, w = rgb.shape[:2]
        resized_images.append(cv2.resize(rgb, (im_dim, im_dim)))
        images.append(rgb)
        ops.append([{"resize": (im_dim, im_dim, w, h)}])
    return images, resized_images, ops


def _nonsquare_batch(images):
    """Resize each image to a cycling non-square size and build matching per-image operators."""
    resized, ops = [], []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        rh, rw = OFF_SIZES[i % len(OFF_SIZES)]
        resized.append(cv2.resize(img, (rw, rh)))
        ops.append([{"resize": (rw, rh, w, h)}])
    return resized, ops


def _assert_empty_output(out, keys, batch_size=1):
    """Assert each key in out has batch_size items and all are empty."""
    for key in keys:
        assert len(out[key]) == batch_size
        for i in range(batch_size):
            assert len(out[key][i]) == 0


def _assert_batch_counts(out, keys, n=None):
    """Assert each key in out has exactly n items."""
    if n is None:
        return
    for key in keys:
        assert len(out[key]) == n


def _assert_batch_scores(out, key, min_conf):
    """Assert all scores in a batch output are >= min_conf."""
    for sc_list in out[key]:
        for sc in sc_list:
            assert sc >= min_conf


def _assert_batch_nonempty(out, keys):
    """Assert each per-image entry for every key is non-empty."""
    for key in keys:
        for i, item in enumerate(out[key]):
            assert len(item) > 0, f"out['{key}'][{i}] is empty"


def _assert_batch_output(out, keys, n=None, min_conf=0.5):
    """Assert batch size, minimum scores, and non-empty entries for every key."""
    _assert_batch_counts(out, keys, n)
    _assert_batch_scores(out, "scores", min_conf)
    _assert_batch_nonempty(out, keys)


def _write_annotated_images(model, out, images, filename_prefix):
    """Annotate each image with detection results and write to OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for img_idx, img in enumerate(images):
        per_img = {k: v[img_idx] for k, v in out.items()}
        im_out = model.annotate_image(per_img, img)
        im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT_DIR, f"{filename_prefix}-{img_idx}.png"), im_out)


def _assert_batch_cuda(out, keys, idx=0):
    """Assert all tensors at the given image index across keys are on CUDA."""
    for tensors in zip(*[out[k][idx] for k in keys]):
        for t in tensors:
            assert t.is_cuda


class Test_Yolo_Det:
    KEYS = ["boxes", "scores", "classes"]

    def test_compare_with_ultralytics_nonsquare(self, imgs_coco):
        images, _, _ = imgs_coco
        resized_images, _ = _nonsquare_batch(images)
        for model_path in OD_DET_MODELS:
            ults_model = YOLO(model_path)
            our_model = Yolo(model_path, device="cpu", image_size=IMGSZ)
            batch_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in resized_images]

            results = ults_model(batch_bgr, conf=0.5, iou=0.4, max_det=300, device="cpu")
            out, _ = our_model.predict(resized_images, configs=0.5, iou=0.4, max_det=300)
            for ults_result, our_boxes, our_scores in zip(results, out["boxes"], out["scores"]):
                ults_out = ults_result.cpu().numpy()
                assert np.array_equal(np.array(our_boxes), ults_out.boxes.xyxy)
                assert np.array_equal(np.array(our_scores), ults_out.boxes.conf)

    def test_warmup(self, all_models):
        for model in all_models["det"]:
            model.warmup()

    def test_predict_batch_empty(self, all_models):
        batch = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2
        for model in all_models["det"]:
            out, _ = model.predict(batch, configs=0.5)
            _assert_empty_output(out, self.KEYS, batch_size=2)

    def test_predict_batch(self, all_models, imgs_coco):
        images, _, _ = imgs_coco
        if len(images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(images)
        resized_images, ops_list = _nonsquare_batch(images)

        for model in all_models["det"]:
            # per-image operators
            model.predict(resized_images, configs=0.5, operators=ops_list)

            # shared operators (list[dict] applied to all images)
            model.predict(resized_images, configs=0.5, operators=ops_list[0])

            # no operators
            model.predict(resized_images, configs=0.5)

            if torch.cuda.is_available():
                tensor_batch = [torch.from_numpy(img).cuda() for img in resized_images]
                out_gpu, _ = model.predict(tensor_batch, configs=0.5, operators=ops_list)
                for img_idx in range(num_imgs):
                    _assert_batch_cuda(out_gpu, self.KEYS[:-1], img_idx)
                _write_annotated_images(model, out_gpu, images, filename_prefix=model.test_name)

    def test_predict_batch_square(self, all_models, imgs_coco):
        _, resized_images, ops_list = imgs_coco
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(resized_images)
        for model in all_models["det"]:
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)
            _assert_batch_output(out, self.KEYS, num_imgs)

    def test_predict_batch_invalid_operators(self, yolo_models, imgs_coco):
        _, resized_images, ops_list = imgs_coco
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        model = yolo_models["det"][0]
        # operators length (1) doesn't match batch size (N > 1)
        with pytest.raises(ValueError):
            model.predict(resized_images, configs=0.5, operators=[ops_list[0]])

    def test_insize_input_no_warning(self, imgs_coco, caplog):
        """An in-size input (== image_size) is passed through with no resize warning."""
        _, resized_images, _ = imgs_coco
        model = Yolo(OD_DET_MODELS[0], device="cpu", image_size=IMGSZ)
        with caplog.at_level(logging.WARNING):
            model.predict([resized_images[0]], configs=0.5)
        assert not [r for r in caplog.records if "model input" in r.message]


class Test_Yolo_Seg:
    KEYS = ["boxes", "masks", "scores", "segments", "classes"]

    def test_compare_with_ultralytics_nonsquare(self, imgs_coco):
        images, _, _ = imgs_coco
        resized_images, _ = _nonsquare_batch(images)
        for model_path in OD_SEG_MODELS:
            ults_model = YOLO(model_path)
            our_model = YoloSeg(model_path, device="cpu", image_size=IMGSZ)
            batch_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in resized_images]
            results = ults_model(batch_bgr, conf=0.5, iou=0.4, max_det=300, retina_masks=True, device="cpu")
            out, _ = our_model.predict(resized_images, configs=0.5, iou=0.4, max_det=300, return_segments=True)
            for ults_result, our_boxes, our_scores, our_masks, our_segs in zip(
                results, out["boxes"], out["scores"], out["masks"], out["segments"]
            ):
                ults_out = ults_result.cpu().numpy()
                assert np.array_equal(np.array(our_boxes), ults_out.boxes.xyxy)
                assert np.array_equal(np.array(our_scores), ults_out.boxes.conf)
                assert np.array_equal(np.array(our_masks), ults_out.masks.data)
                assert len(our_segs) == len(ults_result.masks.xy)
                for s1, s2 in zip(our_segs, ults_result.masks.xy):
                    assert np.array_equal(s1, s2)

    def test_warmup(self, all_models):
        for model in all_models["seg"]:
            model.warmup()

    def test_predict_batch_empty(self, all_models):
        batch = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2
        for model in all_models["seg"]:
            out, _ = model.predict(batch, configs=0.5)
            _assert_empty_output(out, self.KEYS, batch_size=2)

    def test_predict_batch(self, all_models, imgs_coco):
        images, _, _ = imgs_coco
        if len(images) < 2:
            pytest.skip("Not enough images for batch test")

        num_images = len(images)
        resized_images, batch_ops = _nonsquare_batch(images)
        for model in all_models["seg"]:
            # per-image operators
            out, _ = model.predict(resized_images, configs=0.5, operators=batch_ops, return_segments=False)
            for img_idx in range(num_images):
                assert len(out["segments"][img_idx]) == 0

            # shared operators
            model.predict(resized_images, configs=0.5, operators=batch_ops[0])

            # no operators
            model.predict(resized_images, configs=0.5)

            if torch.cuda.is_available():
                tensor_batch = [torch.from_numpy(img).cuda() for img in resized_images]
                out_gpu, _ = model.predict(tensor_batch, configs=0.5, operators=batch_ops)
                for img_idx in range(num_images):
                    _assert_batch_cuda(out_gpu, self.KEYS[:-1], img_idx)
                _write_annotated_images(model, out_gpu, images, filename_prefix=model.test_name)

    def test_predict_batch_square(self, all_models, imgs_coco):
        _, resized_images, ops_list = imgs_coco
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(resized_images)
        for model in all_models["seg"]:
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)
            _assert_batch_output(out, self.KEYS, num_imgs)


class Test_Yolo_Obb:
    KEYS = ["boxes", "scores", "classes"]

    def test_compare_with_ultralytics_nonsquare(self, imgs_dota8):
        images, _, _ = imgs_dota8
        for model_path in OD_OBB_DOTA_8:
            ults_model = YOLO(model_path)
            our_model = YoloObb(model_path, device="cpu", image_size=OBB_IMGSZ)
            resized_images, _ = _nonsquare_batch(images)
            batch_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in resized_images]
            results = ults_model(batch_bgr, conf=0.5, iou=0.4, max_det=300, device="cpu")
            out, _ = our_model.predict(resized_images, configs=0.5, iou=0.4, max_det=300)
            for ults_result, our_boxes, our_scores in zip(results, out["boxes"], out["scores"]):
                ults_out = ults_result.cpu().numpy()
                assert np.allclose(np.array(our_boxes), ults_out.obb.xyxyxyxy, atol=1e-5)
                assert np.array_equal(np.array(our_scores), ults_out.obb.conf)

    def test_warmup_dota8(self, all_models):
        for model in all_models["obb_dota8"]:
            model.warmup()

    def test_predict_batch_empty(self, all_models):
        batch = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2
        for model in all_models["obb_dota8"]:
            out, _ = model.predict(batch, configs=0.5)
            _assert_empty_output(out, self.KEYS, batch_size=2)

    def test_predict_batch(self, all_models, imgs_dota8):
        images, _, _ = imgs_dota8
        if len(images) < 2:
            pytest.skip("Not enough images for batch test")
        resized_images, ops_list = _nonsquare_batch(images)

        for model in all_models["obb_dota8"]:
            # per-image operators
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)

            # shared operators
            out2, _ = model.predict(resized_images, configs=0.5, operators=ops_list[0])

            # no operators
            out3, _ = model.predict(resized_images, configs=0.5)

            if torch.cuda.is_available():
                tensor_batch = [torch.from_numpy(img).cuda() for img in resized_images]
                out_gpu, _ = model.predict(tensor_batch, configs=0.5, operators=ops_list)
                _write_annotated_images(model, out_gpu, images, filename_prefix=model.test_name)

    def test_predict_batch_square(self, all_models, imgs_dota8):
        _, resized_images, ops_list = imgs_dota8
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(resized_images)
        for model in all_models["obb_dota8"]:
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)
            _assert_batch_output(out, self.KEYS, num_imgs)


class Test_Yolo_Pose:
    KEYS = ["boxes", "scores", "points", "classes"]

    def test_compare_with_ultralytics_nonsquare(self, imgs_coco):
        images, _, _ = imgs_coco
        resized_images, _ = _nonsquare_batch(images)
        for model_path in OD_POSE_MODELS:
            ults_model = YOLO(model_path)
            our_model = YoloPose(model_path, device="cpu", image_size=IMGSZ)
            batch_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in resized_images]
            results = ults_model(batch_bgr, conf=0.5, iou=0.4, max_det=300, device="cpu")
            out, _ = our_model.predict(resized_images, configs=0.5, iou=0.4, max_det=300)
            for ults_result, our_boxes, our_scores, our_points in zip(results, out["boxes"], out["scores"], out["points"]):
                ults_out = ults_result.cpu().numpy()
                assert np.array_equal(np.array(our_boxes), ults_out.boxes.xyxy)
                assert np.array_equal(np.array(our_scores), ults_out.boxes.conf)
                assert np.array_equal(np.array(our_points), ults_out.keypoints.data)

    def test_warmup(self, all_models):
        for model in all_models["pose"]:
            model.warmup()

    def test_predict_batch_empty(self, all_models):
        batch = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2
        for model in all_models["pose"]:
            out, _ = model.predict(batch, configs=0.5)
            _assert_empty_output(out, self.KEYS, batch_size=2)

    def test_predict_batch(self, all_models, imgs_coco):
        # No _assert_batch_output here because pose is more sensitive to distortion and drop some detections.
        images, _, _ = imgs_coco
        if len(images) < 2:
            pytest.skip("Not enough images for batch test")
        resized_images, ops_list = _nonsquare_batch(images)
        for model in all_models["pose"]:
            # per-image operators
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)

            # shared operators
            out2, _ = model.predict(resized_images, configs=0.5, operators=ops_list[0])

            # no operators
            out3, _ = model.predict(resized_images, configs=0.5)

            if torch.cuda.is_available():
                tensor_batch = [torch.from_numpy(img).cuda() for img in resized_images]
                out_gpu, _ = model.predict(tensor_batch, configs=0.5, operators=ops_list)
                _write_annotated_images(model, out_gpu, images, filename_prefix=model.test_name)

    def test_predict_batch_square(self, all_models, imgs_coco):
        _, resized_images, ops_list = imgs_coco
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(resized_images)
        for model in all_models["pose"]:
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)
            _assert_batch_output(out, self.KEYS, num_imgs)
