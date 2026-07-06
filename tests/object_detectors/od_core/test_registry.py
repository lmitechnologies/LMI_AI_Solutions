import logging

from object_detectors.od_core.object_detector_registry import ObjectDetectorRegistry

logger = logging.getLogger(__name__)


def test_backends_table_covers_known_models():
    """The BACKENDS table expands to the known lookup keys — no backend imports needed."""
    assert len(ObjectDetectorRegistry._key_map) > 0, "ObjectDetectorRegistry should have registered keys."

    to_be_tested_keys = [
        ("detectron2", "mask_rcnn", "objectdetection", "v0"),
        ("ultralytics", "yolov5", "objectdetection", "v0"),
        ("ultralytics", "yolov8", "objectdetection", "v1"),
        ("ultralytics", "yolov11", "instancesegmentation", "v1"),
        ("ultralytics", "yolo", "keypointdetection", "v1"),
        ("ultralytics", "yolo", "orientedobjectdetection", "v1"),
        ("rfdetr", "rfdetr", "objectdetection", "v1"),
        ("rfdetr", "rfdetr", "instancesegmentation", "v1"),
    ]

    for key in to_be_tested_keys:
        key2 = ObjectDetectorRegistry._generate_key(*key)
        assert key2 in ObjectDetectorRegistry._key_map, f"Model {key} should be registered."
