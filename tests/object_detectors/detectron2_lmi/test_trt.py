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
SIZE_OFFSETS = [(128, 64), (64, 128), (192, 96)]

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

        return USE_CUDA
    except ImportError:
        return False


@pytest.fixture(scope="module")
def trt_model():
    if not _trt_available():
        pytest.skip("TensorRT not available")
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
    for _ in range(1):
        trt_model.warmup()


def test_batch_operators(trt_model, imgs_coco):
    model = trt_model
    confs = {v: 0.8 for v in class_map.values()}
    th, tw = model.image_size

    images = imgs_coco
    original_sizes = [img.shape[:2] for img in images]
    resized = [cv2.resize(img, (tw, th)) for img in images]
    from lmi_utils.preprocess_utils.ops import ResizeMeta

    operators = [
        ResizeMeta(
            src_sizes=[[w, h] for h, w in original_sizes],
            dst_sizes=[[tw, th] for _ in original_sizes],
            pads=[[0, 0, 0, 0] for _ in original_sizes],
        )
    ]

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
    from lmi_utils.preprocess_utils.ops import ResizeMeta

    # Per-image varied off-sizes (!= engine input) passed as CUDA tensors: the guard stretches each
    # on-device to the engine size while the resize operator reverts boxes/masks to the original frame.
    resized_dims = [(th + dh, tw + dw) for dh, dw in (SIZE_OFFSETS[i % len(SIZE_OFFSETS)] for i in range(len(images)))]
    resized = [torch.from_numpy(cv2.resize(img, (rw, rh))).cuda() for img, (rh, rw) in zip(images, resized_dims)]
    operators = [
        ResizeMeta(
            src_sizes=[[w, h] for h, w in original_sizes],
            dst_sizes=[[rw, rh] for (rh, rw) in resized_dims],
            pads=[[0, 0, 0, 0] for _ in original_sizes],
        )
    ]

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


def test_clamp_boxes_to_image():
    """TRT postprocess must clamp out-of-bounds boxes to the image (the engine drops Boxes.clip).

    Built without an engine so it runs everywhere: feed synthetic normalized predictions
    (one box spilling past every edge, one fully inside) straight through postprocess.
    """
    from object_detectors.detectron2_lmi.model import Detectron2TRT

    model = object.__new__(Detectron2TRT)
    model.device = torch.device("cpu")
    model.batch_size = 1
    model._setup_class_map({0: "person"})

    image_h, image_w = 100, 200
    images = [np.zeros((image_h, image_w, 3), dtype=np.uint8)]
    # Normalized xyxy: first box out of bounds on all sides, second well inside.
    boxes = torch.tensor([[[-0.1, -0.2, 1.2, 1.3], [0.5, 0.4, 0.9, 0.8]]])
    num_preds = torch.tensor([2])
    scores = torch.tensor([[0.9, 0.9]])
    classes = torch.tensor([[0, 0]])

    results = model.postprocess((num_preds, boxes, scores, classes), images=images, configs=0.0)
    out = results[0].boxes

    assert (out[:, 0::2] >= 0).all() and (out[:, 0::2] <= image_w).all()
    assert (out[:, 1::2] >= 0).all() and (out[:, 1::2] <= image_h).all()
    assert torch.allclose(out[0], torch.tensor([0.0, 0.0, float(image_w), float(image_h)]))
    assert torch.allclose(out[1], torch.tensor([100.0, 40.0, 180.0, 80.0]))


def test_no_cross_chunk_contamination(trt_model, imgs_coco):
    """
    The test creates a batch of interleaved rich/blank images large enough to force
    multiple chunks, then verifies blank images never acquire ghost detections from
    adjacent chunks' buffer state.
    """
    model = trt_model
    confs = {v: 0.8 for v in class_map.values()}
    th, tw = model.image_size

    blank = np.zeros((th, tw, 3), dtype=np.uint8)
    rich = cv2.resize(imgs_coco[0], (tw, th))

    # Interleave rich and blank to make contamination detectable in either direction.
    # Use enough images to guarantee chunking regardless of fixed_batch_size.
    n_pairs = 8
    batch = []
    for _ in range(n_pairs):
        batch.append(rich)
        batch.append(blank)

    # Run several times — a race condition failure is probabilistic, repetition helps.
    for _ in range(10):
        batch_out, _ = model.predict(batch, configs=confs)
        for idx in range(len(batch)):
            out = {k: v[idx] for k, v in batch_out.items()}
            if idx % 2 == 0:
                _assert_nonempty_out(out)
            else:
                _assert_empty_out(out)
