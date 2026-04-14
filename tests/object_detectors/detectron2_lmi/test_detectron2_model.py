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
ENGINE_PATH = "tests/assets/models/od/detectron2/model.engine"
OUT_DIR = "tests/outputs/od/detectron2"
USE_CUDA = torch.cuda.is_available()
KEYS = ["boxes", "classes", "scores", "masks", "segments"]

with open(COCO_CLASSMAP, "r") as f:
    class_map = json.load(f)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def imgs_coco():
    paths = [os.path.join(COCO_DIR, img) for img in os.listdir(COCO_DIR)]
    images = []
    for p in paths:
        if "png" not in p and "jpg" not in p:
            continue
        im = cv2.imread(p)
        images.append(im)
    return images


def _make_og_model(device):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MASKRCNN_MODEL_CONFIG))
    cfg.MODEL.DEVICE = device
    model = build_model(cfg)
    DetectionCheckpointer(model).load(OG_WEIGHTS_PATH)
    model.eval()
    return model


@pytest.fixture(scope="module")
def og_model():
    return _make_og_model("cuda" if USE_CUDA else "cpu")


@pytest.fixture(scope="module")
def og_cpu_model():
    return _make_og_model("cpu")


def _make_model(device):
    return {
        "direct": lambda: Detectron2Model(MODEL_PATH, class_map=class_map, device=device),
        "api": lambda: ObjectDetector(
            metadata=dict(version="v0", model_name="mask_rcnn", task="seg", framework="detectron2"),
            model_path=MODEL_PATH,
            class_map=class_map,
            device=device,
        ),
    }


@pytest.fixture(scope="module", params=["direct", "api"])
def model(request):
    device = "cuda" if USE_CUDA else "cpu"
    return _make_model(device)[request.param]()


@pytest.fixture(scope="module", params=["direct", "api"])
def model_cpu(request):
    return _make_model("cpu")[request.param]()


def _assert_empty_out(out, keys=None):
    """Assert all specified per-image output keys are empty."""
    if keys is None:
        keys = KEYS
    for k in keys:
        assert len(out[k]) == 0, f"Expected empty out['{k}'], got {len(out[k])}"


def _assert_lengths_equal(out, keys=None):
    """Assert all specified per-image output keys have equal lengths."""
    if keys is None:
        keys = KEYS
    lengths = [len(out[k]) for k in keys]
    assert len(set(lengths)) == 1, f"Unequal lengths: {dict(zip(keys, lengths))}"


def _assert_nonempty_out(out, keys=None):
    """Assert all specified per-image output keys are non-empty and have equal lengths."""
    if keys is None:
        keys = KEYS
    _assert_lengths_equal(out, keys)
    for k in keys:
        assert len(out[k]) > 0, f"Expected non-empty out['{k}']"


def _assert_scores_geq(out, min_conf):
    """Assert all per-image scores are >= min_conf."""
    for sc in out["scores"]:
        assert sc >= min_conf, f"Score {sc} < {min_conf}"


def _assert_batch_counts(outputs, keys, n):
    """Assert each key in a batch output has exactly n items."""
    for k in keys:
        assert len(outputs[k]) == n, f"Expected {n} items for outputs['{k}'], got {len(outputs[k])}"


def _assert_batch_empty(outputs, keys, n):
    """Assert each key has exactly n items and every item is empty."""
    _assert_batch_counts(outputs, keys, n)
    for k in keys:
        for item in outputs[k]:
            assert len(item) == 0, f"Expected empty item in outputs['{k}']"


def _trt_available():
    try:
        import tensorrt  # noqa: F401
        from cuda import cudart  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def detectron2_trt_model():
    if not _trt_available():
        pytest.skip("TensorRT / cuda-python not available")
    if not os.path.exists(ENGINE_PATH):
        pytest.skip(f"Engine file not found: {ENGINE_PATH}")
    try:
        return Detectron2Model(ENGINE_PATH, class_map=class_map)
    except Exception as e:
        pytest.skip(f"Failed to load TRT engine: {e}")


def test_compare_with_original_model(og_cpu_model, model_cpu, imgs_coco):
    confs = {v: 0.00 for v in class_map.values()}
    for image in imgs_coco:
        img = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        inputs = [{"image": img}]
        with torch.no_grad():
            orginal_preds = og_cpu_model.inference(inputs, do_postprocess=True)[0]
        preds, _ = model_cpu.predict(image, configs=confs)

        # check if the outputs are all close
        instances = orginal_preds["instances"]
        assert np.array_equal(instances.scores.cpu().numpy(), preds.get("scores")[0])
        assert np.array_equal(instances.pred_boxes.tensor.cpu().numpy(), preds.get("boxes")[0])
        assert np.array_equal(instances.pred_masks.cpu().numpy(), preds.get("masks")[0])


def test_operators(model, imgs_coco):
    confs = {v: 0.95 for v in class_map.values()}
    image = imgs_coco[0]
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, (512, 512))
    operators = [{"resize": [512, 512, w, h]}]
    outputs, _ = model.predict(image_resized, configs=confs, return_segments=False, operators=operators)
    outputs = {k: v[0] for k, v in outputs.items()}

    _assert_nonempty_out(outputs, ["boxes", "classes", "scores", "masks"])
    _assert_empty_out(outputs, ["segments"])
    _assert_scores_geq(outputs, 0.95)
    assert outputs["masks"].shape[1] == h
    assert outputs["masks"].shape[2] == w


def test_warmup(model):
    model.warmup()


def test_empty(model):
    blank_images = [np.zeros((512, 512, 3), dtype=np.uint8) for _ in range(2)]
    confs = {v: 0.95 for v in class_map.values()}
    outputs, _ = model.predict(blank_images, configs=confs)
    _assert_batch_empty(outputs, KEYS, len(blank_images))


def test_operators_no_masks(model, imgs_coco):
    image = imgs_coco[0]
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, (512, 512))
    operators = [{"resize": [512, 512, w, h]}]
    outputs, _ = model.predict(image_resized, configs=1, operators=operators)
    outputs = {k: v[0] for k, v in outputs.items()}

    _assert_empty_out(outputs)


def test_batch_operators(model, imgs_coco):
    confs = {v: 0.8 for v in class_map.values()}
    images = imgs_coco
    th, tw = 640, 640
    original_sizes = [img.shape[:2] for img in images]
    images_resized = [cv2.resize(img, (tw, th)) for img in images]
    operators = [[{"resize": [tw, th, w, h]}] for h, w in original_sizes]
    outputs, _ = model.predict(images_resized, configs=confs, operators=operators)
    _assert_batch_counts(outputs, KEYS, len(images))
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, (h, w) in enumerate(original_sizes):
        out = {k: v[i] for k, v in outputs.items()}
        _assert_nonempty_out(out)
        _assert_scores_geq(out, 0.8)
        assert out["masks"].shape[1] == h
        assert out["masks"].shape[2] == w
        annotated = model.annotate_image(out, images[i].copy())
        cv2.imwrite(os.path.join(OUT_DIR, f"coco_{i}_batch_operators.jpg"), annotated)


def test_trt_batch_operators(detectron2_trt_model, imgs_coco):
    model = detectron2_trt_model
    confs = {v: 0.8 for v in class_map.values()}
    th, tw = model.image_size

    images = imgs_coco
    original_sizes = [img.shape[:2] for img in images]
    resized = [cv2.resize(img, (tw, th)) for img in images]
    operators = [[{"resize": [tw, th, w, h]}] for h, w in original_sizes]

    outputs, _ = model.predict(resized, configs=confs, operators=operators)
    _assert_batch_counts(outputs, KEYS, len(images))
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, (h, w) in enumerate(original_sizes):
        out = {k: v[i] for k, v in outputs.items()}
        _assert_nonempty_out(out)
        _assert_scores_geq(out, 0.8)
        assert out["masks"].shape[1] == h
        assert out["masks"].shape[2] == w
        annotated = model.annotate_image(out, images[i].copy())
        cv2.imwrite(os.path.join(OUT_DIR, f"coco_{i}_trt_batch_operators.jpg"), annotated)


def test_trt_empty(detectron2_trt_model):
    model = detectron2_trt_model
    th, tw = model.image_size
    empty_img = np.zeros((th, tw, 3), dtype=np.uint8)
    batch_outputs, _ = model.predict(empty_img, configs=0.5)
    outputs = {k: v[0] for k, v in batch_outputs.items()}
    _assert_empty_out(outputs)


def test_trt_warmup(detectron2_trt_model):
    detectron2_trt_model.warmup()
