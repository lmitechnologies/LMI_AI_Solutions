import glob
import os

import cv2
import pytest
import torch

from object_detectors.ultralytics_lmi.yolo.model import Yolo, YoloSeg
from tests.object_detectors.match_detections import assert_detections_match

COCO_DIR = "tests/assets/images/coco"
MODEL_DIR = "tests/assets/models/od/ultralytics"
IMGSZ = [640, 640]
CASES = [(Yolo, "yolo26n"), (YoloSeg, "yolo11n-seg")]


@pytest.fixture(scope="module")
def imgs_coco():
    paths = sorted(p for p in glob.glob(os.path.join(COCO_DIR, "*")) if p.endswith((".png", ".jpg")))
    return [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (IMGSZ[1], IMGSZ[0])) for p in paths]


@pytest.mark.parametrize("cls,stem", CASES, ids=[c[1] for c in CASES])
def test_engine_matches_pt(cls, stem, imgs_coco):
    """The engine built by tests/build_test_engines.py finds the .pt model's detections."""
    if not torch.cuda.is_available():
        pytest.skip("TensorRT model can only be tested on CUDA device.")
    pytest.importorskip("tensorrt")
    engine = os.path.join(MODEL_DIR, f"{stem}.engine")
    if not os.path.exists(engine):
        pytest.skip(f"Engine file not found: {engine}")

    ref, _ = cls(os.path.join(MODEL_DIR, f"{stem}.pt"), device="cuda", image_size=IMGSZ).predict(imgs_coco, configs=0.4)
    out, _ = cls(engine, device="cuda", image_size=IMGSZ).predict(imgs_coco, configs=0.4)
    assert sum(len(s) for s in ref["scores"]) > 0
    for i in range(len(imgs_coco)):
        # a borderline yolo26n kite moves 0.05 between the .pt on CPU and CUDA alone, hence the wide score tolerance
        assert_detections_match(
            {k: v[i] for k, v in ref.items()},
            {k: v[i] for k, v in out.items()},
            min_score=0.5,
            score_tol=0.1,
            min_box_iou=0.95,
            min_mask_iou=0.95 if cls is YoloSeg else None,
            label=f"{stem} image {i}",
        )
