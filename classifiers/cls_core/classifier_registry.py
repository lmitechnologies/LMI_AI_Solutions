from lmi_common.model_registry import ModelRegistry


class ClassifierRegistry(ModelRegistry):
    PACKAGES = {
        "ultralytics": ["classifiers.ultralytics_lmi.yolo.model"],
    }
    _registry = {}
