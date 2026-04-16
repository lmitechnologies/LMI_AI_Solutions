from lmi_common.model_registry import ModelRegistry


class ObjectDetectorRegistry(ModelRegistry):
    PACKAGES = [
        "object_detectors.ultralytics_lmi",
        "object_detectors.detectron2_lmi",
        "object_detectors.yolov5_lmi",
        "object_detectors.rf_detr_lmi",
    ]
    TARGET_MODULE_SUFFIXES = [".model"]
    _registry = {}
