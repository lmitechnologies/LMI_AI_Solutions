import logging
import os

import cv2
import numpy as np
import pytest
import torch
from rfdetr.assets.coco_classes import COCO_CLASSES

from object_detectors.od_core.object_detector import ObjectDetector

from .test_model import _assert_empty_out, _assert_nonempty_out, _assert_scores_geq

logger = logging.getLogger(__name__)

COCO_DIR = "tests/assets/images/coco"
KEYS = ["boxes", "scores", "masks", "segments", "classes"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRT_MODEL = "tests/assets/models/od/rf_detr/inference_model.engine"
OUT_DIR = "tests/outputs/od/rf_detr"
IMAGE_SIZE = 384


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
def trt_model():
    if DEVICE != "cuda":
        pytest.skip("TensorRT model can only be tested on CUDA device.")
    try:
        return ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=TRT_MODEL,
            class_map=COCO_CLASSES,
            image_size=[IMAGE_SIZE, IMAGE_SIZE],
        )
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


def test_trt_warmup(trt_model):
    trt_model.warmup()


def test_operators_batch(imgs_coco, trt_model):
    original_sizes = [(img.shape[1], img.shape[0]) for img in imgs_coco]  # (w, h)
    from lmi_utils.preprocess_utils.ops import ResizeMeta

    operators = [
        ResizeMeta(
            src_sizes=[[w, h] for w, h in original_sizes],
            dst_sizes=[[IMAGE_SIZE, IMAGE_SIZE] for _ in original_sizes],
            pads=[[0, 0, 0, 0] for _ in original_sizes],
        )
    ]
    imgs_resized = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in imgs_coco]

    batch_outputs, _ = trt_model.predict(imgs_resized, configs=0.5, operators=operators)
    assert len(batch_outputs["boxes"]) == len(imgs_coco)

    os.makedirs(OUT_DIR, exist_ok=True)
    for idx, img in enumerate(imgs_coco):
        w, h = original_sizes[idx]
        out = {k: v[idx] for k, v in batch_outputs.items()}
        _assert_nonempty_out(out)
        _assert_scores_geq(out, 0.5)

        # boxes with operators should be scaled to the original image size
        assert np.all(out["boxes"][:, 0] <= w)
        assert np.all(out["boxes"][:, 1] <= h)
        assert np.all(out["boxes"][:, 2] <= w)
        assert np.all(out["boxes"][:, 3] <= h)

        annotated_image = trt_model.annotate_image(out, img)
        out_name = f"out_trt_operators_batch_{idx}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


def test_empty(trt_model):
    empty_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    batch_outputs, _ = trt_model.predict(empty_img, configs=0.5)
    out = {k: v[0] for k, v in batch_outputs.items()}
    _assert_empty_out(out)


def test_operators_batch_cuda(imgs_coco, trt_model):
    original_sizes = [(img.shape[1], img.shape[0]) for img in imgs_coco]  # (w, h)
    from lmi_utils.preprocess_utils.ops import ResizeMeta

    operators = [
        ResizeMeta(
            src_sizes=[[w, h] for w, h in original_sizes],
            dst_sizes=[[IMAGE_SIZE, IMAGE_SIZE] for _ in original_sizes],
            pads=[[0, 0, 0, 0] for _ in original_sizes],
        )
    ]
    imgs_resized = [torch.from_numpy(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))).cuda() for img in imgs_coco]

    batch_outputs, _ = trt_model.predict(imgs_resized, configs=0.5, operators=operators)
    assert len(batch_outputs["boxes"]) == len(imgs_coco)

    os.makedirs(OUT_DIR, exist_ok=True)
    for idx, img in enumerate(imgs_coco):
        w, h = original_sizes[idx]
        out = {k: v[idx] for k, v in batch_outputs.items()}
        _assert_nonempty_out(out)
        _assert_scores_geq(out, 0.5)

        # boxes with operators should be scaled to the original image size
        assert (out["boxes"][:, 0] <= w).all()
        assert (out["boxes"][:, 1] <= h).all()
        assert (out["boxes"][:, 2] <= w).all()
        assert (out["boxes"][:, 3] <= h).all()

        _assert_all_cuda(out)
        annotated_image = trt_model.annotate_image(out, img)
        out_name = f"out_trt_operators_batch_cuda_{idx}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


def test_empty_cuda(trt_model):
    empty_img = torch.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8).cuda()
    batch_outputs, _ = trt_model.predict(empty_img, configs=0.5)
    out = {k: v[0] for k, v in batch_outputs.items()}
    _assert_empty_out(out)


def test_no_cross_chunk_contamination(imgs_coco, trt_model):
    """
    The test creates a batch of interleaved rich/blank images large enough to force
    multiple chunks, then verifies blank images never acquire ghost detections from
    adjacent chunks' buffer state.
    """
    CONF = 0.5
    blank = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    rich = cv2.resize(imgs_coco[0], (IMAGE_SIZE, IMAGE_SIZE))

    # Interleave rich and blank to make contamination detectable in either direction.
    # Use enough images to guarantee chunking regardless of fixed_batch_size.
    n_pairs = 8
    batch = []
    for _ in range(n_pairs):
        batch.append(rich)
        batch.append(blank)

    # Run several times — a race condition failure is probabilistic, repetition helps
    for _ in range(10):
        batch_out, _ = trt_model.predict(batch, configs=CONF)
        for idx in range(len(batch)):
            out = {k: v[idx] for k, v in batch_out.items()}
            if idx % 2 == 0:
                _assert_nonempty_out(out, ["boxes", "scores", "classes"])
            else:
                _assert_empty_out(out, ["boxes", "scores", "classes"])
