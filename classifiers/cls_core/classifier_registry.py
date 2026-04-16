from lmi_common.model_registry import ModelRegistry


class ClassifierRegistry(ModelRegistry):
    PACKAGES = ["classifiers.ultralytics_lmi"]
    TARGET_MODULE_SUFFIXES = [".model"]
    _registry = {}
