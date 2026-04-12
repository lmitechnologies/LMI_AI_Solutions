import logging

from object_detectors.od_core.object_detector_registry import ObjectDetectorRegistry

logger = logging.getLogger(__name__)


def test_auto_registration():
    """
    Test that the auto-registration of object detector models works correctly.
    This will check if the models are registered in the ObjectDetectorRegistry.
    """
    ObjectDetectorRegistry.auto_register_models()
    assert len(ObjectDetectorRegistry._registry) > 0, "ObjectDetectorRegistry should have registered models."

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
        key2 = ObjectDetectorRegistry._generate_key(*key, info={})
        assert key2 in ObjectDetectorRegistry._registry, f"Model {key} should be registered."
