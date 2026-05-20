import logging
import os

import cv2
import numpy as np
import pytest
import torch
from rfdetr import RFDETRSegSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

from object_detectors.od_core.object_detector import ObjectDetector
from object_detectors.rf_detr_lmi.model import RfdetrModel

logger = logging.getLogger(__name__)

COCO_DIR = "tests/assets/images/coco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PTH_FILE = "tests/assets/models/od/rf_detr/rf-detr-seg-small.pth"
OD_MODEL = f"tests/assets/models/od/rf_detr/model_{DEVICE}.pt"
TRT_MODEL = "tests/assets/models/od/rf_detr/inference_model.engine"
OUT_DIR = "tests/outputs/od/rf_detr"
IMAGE_SIZE = 384
MODEL_TYPE = "seg-small"


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
        h, w = rgb.shape[:2]
        images.append(rgb)
    return images


@pytest.fixture(scope="module")
def rf_model():
    model = RFDETRSegSmall(pretrain_weights=PTH_FILE, device=DEVICE)
    return model


@pytest.fixture(scope="module")
def obj_detector():
    obj_detector = ObjectDetector(
        metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
        model_path=OD_MODEL,
        device=DEVICE,
        class_map=COCO_CLASSES,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
    )
    return obj_detector


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


@pytest.fixture(scope="module")
def cpu_models():
    rf_model = RFDETRSegSmall(pretrain_weights=PTH_FILE, device="cpu")
    # rf_model.optimize_for_inference()

    od_pt = ObjectDetector(
        metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
        model_path=OD_MODEL.replace("cuda", "cpu"),
        device="cpu",
        class_map=COCO_CLASSES,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
    )

    od_pth = ObjectDetector(
        metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
        model_path=PTH_FILE,
        model_type=MODEL_TYPE,
        device="cpu",
        class_map=COCO_CLASSES,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
    )
    return rf_model, od_pt, od_pth


KEYS = ["boxes", "scores", "masks", "segments", "classes"]


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


def assert_outputs_match_rf(rf_preds, outputs, label):
    rf_boxes = rf_preds.xyxy
    rf_masks = rf_preds.mask
    rf_classes = [COCO_CLASSES[c] for c in rf_preds.class_id]
    assert rf_boxes.shape[0] == outputs["boxes"].shape[0], f"{label}: Number of boxes mismatch"
    assert np.array_equal(rf_boxes, outputs["boxes"]), f"{label}: Box coordinates mismatch"
    assert np.array_equal(rf_preds.confidence, outputs["scores"]), f"{label}: Confidence scores mismatch"
    assert np.array_equal(rf_classes, outputs["classes"]), f"{label}: Class labels mismatch"
    if rf_masks is not None:
        assert np.array_equal(rf_masks, outputs["masks"]), f"{label}: Masks mismatch"


def test_model_class_comparison(obj_detector):
    direct = RfdetrModel(OD_MODEL, device=DEVICE, class_map=COCO_CLASSES, image_size=[IMAGE_SIZE, IMAGE_SIZE])
    api = obj_detector
    assert type(direct) is type(api), f"direct={type(direct).__name__}, api={type(api).__name__}"


class Test_Rfdetr_Model:
    def test_compare_with_rfdetr(self, imgs_coco, cpu_models):
        "Use cpu to avoid gpu non-determinism issues."

        rf_model, pt_model, pth_model = cpu_models

        for img in imgs_coco:
            resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            rf_preds = rf_model.predict(resized, threshold=0.5)

            batch_pt, _ = pt_model.predict(resized, configs=0.5)
            batch_pth, _ = pth_model.predict(resized, configs=0.5)
            outputs_pt = {k: batch_pt[k][0] for k in ("boxes", "scores", "classes", "masks")}
            outputs_pth = {k: batch_pth[k][0] for k in ("boxes", "scores", "classes", "masks")}

            assert_outputs_match_rf(rf_preds, outputs_pt, "pt_model")
            assert_outputs_match_rf(rf_preds, outputs_pth, "pth_model")

    def test_warmup(self, obj_detector):
        obj_detector.warmup()

    def test_empty(self, obj_detector):
        empty_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        batch_outputs, _ = obj_detector.predict(empty_img, configs=0.5)
        out = {k: v[0] for k, v in batch_outputs.items()}
        _assert_empty_out(out)

    def test_confidence(self, imgs_coco, obj_detector):
        img = cv2.resize(imgs_coco[0], (IMAGE_SIZE, IMAGE_SIZE))
        batch_outputs, _ = obj_detector.predict(img, configs=1.0)
        out = {k: v[0] for k, v in batch_outputs.items()}
        _assert_empty_out(out)

    def test_operators(self, imgs_coco, obj_detector):
        for idx, img in enumerate(imgs_coco):
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            operators = [{"type": "resize", "metadata": [{"src_size": [w, h], "dst_size": [IMAGE_SIZE, IMAGE_SIZE]}]}]

            batch_outputs, _ = obj_detector.predict(img_resized, configs=0.5, operators=operators, return_segments=False)
            out = {k: v[0] for k, v in batch_outputs.items()}
            _assert_nonempty_out(out, ["boxes", "scores", "classes", "masks"])
            _assert_empty_out(out, ["segments"])
            _assert_scores_geq(out, 0.5)

            # boxes with operators should be scaled to the original image size
            assert np.all(out["boxes"][:, 0] <= w)
            assert np.all(out["boxes"][:, 1] <= h)
            assert np.all(out["boxes"][:, 2] <= w)
            assert np.all(out["boxes"][:, 3] <= h)

            annotated_image = obj_detector.annotate_image(out, img)
            out_name = f"out_operators_{idx}.jpg"
            os.makedirs(OUT_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(OUT_DIR, out_name), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    def test_operators_batch(self, imgs_coco, obj_detector):
        original_sizes = [(img.shape[1], img.shape[0]) for img in imgs_coco]  # (w, h)
        operators = [
            {
                "type": "resize",
                "metadata": [{"src_size": [w, h], "dst_size": [IMAGE_SIZE, IMAGE_SIZE]} for w, h in original_sizes],
            }
        ]
        imgs_resized = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in imgs_coco]

        batch_outputs, _ = obj_detector.predict(imgs_resized, configs=0.5, operators=operators)

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

            annotated_image = obj_detector.annotate_image(out, img)
            out_name = f"out_operators_batch_{idx}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, out_name), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    def test_tensor_input(self, imgs_coco, obj_detector):
        """predict() accepts uint8 CUDA HWC tensors and returns the same detections as numpy."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img_np = cv2.resize(imgs_coco[0], (IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = torch.from_numpy(img_np).cuda()

        out_np, _ = obj_detector.predict(img_np, configs=0.5)
        out_tensor, _ = obj_detector.predict(img_tensor, configs=0.5)

        assert out_np.keys() == out_tensor.keys()
        assert len(out_np["boxes"][0]) == len(out_tensor["boxes"][0])

    def test_tensor_input_batch(self, imgs_coco, obj_detector):
        """predict() accepts a list of uint8 CUDA HWC tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        imgs_resized = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in imgs_coco]
        tensor_batch = [torch.from_numpy(img).cuda() for img in imgs_resized]

        out, _ = obj_detector.predict(tensor_batch, configs=0.5)
        assert len(out["boxes"]) == len(imgs_coco)
