from lmi_common.model_registry import ModelRegistry


class ClassifierRegistry(ModelRegistry):
    BACKENDS = [
        {
            "frameworks": ["ultralytics"],
            "model_names": ["yolo", "yolov8", "yolov11"],
            "tasks": ["classification"],
            "versions": ["v1"],
            "class_path": "classifiers.ultralytics_lmi.yolo.model:YoloCls",
        },
    ]
