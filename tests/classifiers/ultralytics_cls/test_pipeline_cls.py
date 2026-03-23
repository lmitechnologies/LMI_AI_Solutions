import logging
import os

import cv2
import pytest
import torch

from classifiers.ultralytics_lmi.yolo.model import YoloCls

# Asset paths
MODEL_PATH = os.path.abspath("tests/assets/models/cls/yolo11n-cls.pt")
IMAGE_DIR = os.path.abspath("tests/assets/images/coco")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = [224, 224]

logger = logging.getLogger(__name__)


@pytest.fixture
def classifier():
    return YoloCls(MODEL_PATH, device=DEVICE, image_size=IMGSZ)


@pytest.fixture
def sample_image():
    image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(image_files) > 0, "No images found in assets"
    # Load first image and resize to model's expected size for consistency
    img = cv2.imread(image_files[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMGSZ[0], IMGSZ[1]))
    return img


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_yolo_cls_predict(classifier, sample_image, input_type):
    """
    Test that YoloCls.predict works with both numpy and torch tensor inputs.
    """
    if input_type == "numpy":
        input_img = sample_image
    else:
        # Convert to torch tensor (H, W, C)
        input_img = torch.from_numpy(sample_image)

    # Run predict
    results, time_info = classifier.predict(input_img)

    # Verify results
    assert "classes" in results
    assert "scores" in results
    assert len(results["classes"]) > 0
    assert len(results["classes"]) == len(results["scores"])

    # Verify time info
    assert "preproc" in time_info
    assert "proc" in time_info
    assert "postproc" in time_info


def test_yolo_cls_grayscale(classifier, sample_image):
    """
    Test that YoloCls.predict handles grayscale (2D) inputs.
    """
    gray_img = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
    assert gray_img.ndim == 2

    results, _ = classifier.predict(gray_img)
    assert len(results["classes"]) > 0
