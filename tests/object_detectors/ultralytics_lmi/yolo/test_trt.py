import glob
import os

import cv2
import pytest
import torch

from object_detectors.ultralytics_lmi.yolo.model import Yolo, YoloObb, YoloPose, YoloSeg
from tests.object_detectors.match_detections import assert_detections_match

MODEL_DIR = "tests/assets/models/od/ultralytics"
# (class, model stem, image dir, image size); sizes match YOLO_MODELS in tests/build_test_engines.py
CASES = [
    (Yolo, "yolo26n", "tests/assets/images/coco", 640),
    (YoloSeg, "yolo11n-seg", "tests/assets/images/coco", 640),
    (YoloPose, "yolo26n-pose", "tests/assets/images/coco", 640),
    (YoloPose, "yolo11n-pose", "tests/assets/images/coco", 640),
    (YoloObb, "yolo26n-obb", "tests/assets/images/dota8", 1024),
    (YoloObb, "yolo11n-obb", "tests/assets/images/dota8", 1024),
]


def _load(image_dir, size):
    paths = sorted(p for p in glob.glob(os.path.join(image_dir, "*")) if p.endswith((".png", ".jpg")))
    return [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (size, size)) for p in paths]


@pytest.mark.parametrize("cls,stem,image_dir,size", CASES, ids=[c[1] for c in CASES])
def test_engine_matches_pt(cls, stem, image_dir, size):
    """The engine built by tests/build_test_engines.py finds the .pt model's detections."""
    if not torch.cuda.is_available():
        pytest.skip("TensorRT model can only be tested on CUDA device.")
    pytest.importorskip("tensorrt")
    engine = os.path.join(MODEL_DIR, f"{stem}.engine")
    if not os.path.exists(engine):
        pytest.skip(f"Engine file not found: {engine}")

    imgs = _load(image_dir, size)
    ref, _ = cls(os.path.join(MODEL_DIR, f"{stem}.pt"), device="cuda", image_size=[size, size]).predict(imgs, configs=0.4)
    out, _ = cls(engine, device="cuda", image_size=[size, size]).predict(imgs, configs=0.4)
    assert sum(len(s) for s in ref["scores"]) > 0
    for i in range(len(imgs)):
        # yolo26n's kite is borderline: scores shift >=0.05 from CPU->CUDA alone, and FP16 adds another ~0.05-0.12
        score_tol = 0.15 if stem == "yolo26n" else 0.1
        assert_detections_match(
            {k: v[i] for k, v in ref.items()},
            {k: v[i] for k, v in out.items()},
            min_score=0.5,
            score_tol=score_tol,
            min_box_iou=0.95,
            min_mask_iou=0.95 if cls is YoloSeg else None,
            max_point_dist=3.0 if cls is YoloPose else None,
            label=f"{stem} image {i}",
        )
