from lmi_common.model_registry import ModelRegistry


class ObjectDetectorRegistry(ModelRegistry):
    PACKAGES = {
        "ultralytics": ["object_detectors.ultralytics_lmi.yolo.model", "object_detectors.yolov5_lmi.model"],
        "ultralytics8": ["object_detectors.ultralytics_lmi.yolo.model"],
        "detectron2": ["object_detectors.detectron2_lmi.model"],
        "rfdetr": ["object_detectors.rf_detr_lmi.model"],
    }
    _registry = {}
