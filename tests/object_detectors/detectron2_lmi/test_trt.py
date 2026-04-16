import json
import logging
import os

import cv2
import numpy as np
import pytest
import torch

from object_detectors.detectron2_lmi.model import Detectron2Model

from .test_model import _assert_batch_counts, _assert_empty_out, _assert_nonempty_out, _assert_scores_geq

COCO_DIR = "tests/assets/images/coco"
COCO_CLASSMAP = "tests/assets/models/od/detectron2/class_map.json"
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


def _trt_available():
    try:
        import tensorrt  # noqa: F401
        from cuda import cudart  # noqa: F401

        return USE_CUDA
    except ImportError:
        return False


@pytest.fixture(scope="module")
def trt_model():
    if not _trt_available():
        pytest.skip("TensorRT / cuda-python not available")
    if not os.path.exists(ENGINE_PATH):
        pytest.skip(f"Engine file not found: {ENGINE_PATH}")
    try:
        return Detectron2Model(ENGINE_PATH, class_map=class_map)
    except Exception as e:
        pytest.skip(f"Failed to load TRT engine: {e}")


def _assert_all_cuda(outputs, keys=KEYS):
    for k, v in outputs.items():
        assert k in keys
        if k == "classes":
            assert isinstance(v, np.ndarray)
        elif k == "segments":
            assert isinstance(v, list)
            for seg in v:
                assert isinstance(seg, torch.Tensor)
                assert seg.is_cuda
        else:
            assert isinstance(v, torch.Tensor)
            assert v.is_cuda


def test_warmup(trt_model):
    trt_model.warmup()


def test_batch_operators(trt_model, imgs_coco):
    model = trt_model
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


def test_batch_operators_cuda(trt_model, imgs_coco):
    model = trt_model
    confs = {v: 0.8 for v in class_map.values()}
    th, tw = model.image_size

    images = imgs_coco
    original_sizes = [img.shape[:2] for img in images]
    resized = [torch.from_numpy(cv2.resize(img, (tw, th))).cuda() for img in images]
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
        _assert_all_cuda(out)
        annotated = model.annotate_image(out, images[i].copy())
        cv2.imwrite(os.path.join(OUT_DIR, f"coco_{i}_trt_batch_operators_cuda.jpg"), annotated)


def test_empty(trt_model):
    model = trt_model
    th, tw = model.image_size
    empty_img = np.zeros((th, tw, 3), dtype=np.uint8)
    batch_outputs, _ = model.predict(empty_img, configs=0.5)
    outputs = {k: v[0] for k, v in batch_outputs.items()}
    _assert_empty_out(outputs)


def test_empty_cuda(trt_model):
    model = trt_model
    th, tw = model.image_size
    empty_img = torch.zeros((th, tw, 3), dtype=torch.uint8).cuda()
    batch_outputs, _ = model.predict(empty_img, configs=0.5)
    outputs = {k: v[0] for k, v in batch_outputs.items()}
    _assert_empty_out(outputs)
