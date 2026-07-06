from typing import Any, Dict

from lmi_common.model_registry import ModelRegistry


class ObjectDetectorRegistry(ModelRegistry):
    BACKENDS = [
        {
            "frameworks": ["ultralytics", "ultralytics8"],
            "model_names": ["yolo", "yolov8", "yolov11"],
            "tasks": ["od", "objectdetection"],
            "versions": ["v1"],
            "class_path": "object_detectors.ultralytics_lmi.yolo.model:Yolo",
        },
        {
            "frameworks": ["ultralytics", "ultralytics8"],
            "model_names": ["yolo", "yolov8", "yolov11"],
            "tasks": ["seg", "instancesegmentation"],
            "versions": ["v1"],
            "class_path": "object_detectors.ultralytics_lmi.yolo.model:YoloSeg",
        },
        {
            "frameworks": ["ultralytics", "ultralytics8"],
            "model_names": ["yolo", "yolov8", "yolov11"],
            "tasks": ["obb", "orientedobjectdetection"],
            "versions": ["v1"],
            "class_path": "object_detectors.ultralytics_lmi.yolo.model:YoloObb",
        },
        {
            "frameworks": ["ultralytics", "ultralytics8"],
            "model_names": ["yolo", "yolov8", "yolov11"],
            "tasks": ["pose", "keypointdetection"],
            "versions": ["v1"],
            "class_path": "object_detectors.ultralytics_lmi.yolo.model:YoloPose",
        },
        {
            "frameworks": ["ultralytics"],
            "model_names": ["yolov5"],
            "tasks": ["od", "seg", "instancesegmentation", "objectdetection"],
            "versions": ["v0"],
            "class_path": "object_detectors.yolov5_lmi.model:Yolov5",
        },
        {
            "frameworks": ["detectron2"],
            "model_names": ["mask_rcnn", "faster_rcnn"],
            "tasks": ["od", "seg", "instancesegmentation", "objectdetection"],
            "versions": ["v0"],
            "class_path": "object_detectors.detectron2_lmi.model:Detectron2Model",
        },
        {
            "frameworks": ["rfdetr"],
            "model_names": ["rfdetr"],
            "tasks": ["od", "seg", "instancesegmentation", "objectdetection"],
            "versions": ["v1"],
            "class_path": "object_detectors.rf_detr_lmi.model:RfdetrModel",
        },
    ]

    @classmethod
    def _get_version(cls, metadata: Dict[str, Any], framework: str) -> str:
        version = metadata.get("version")
        if version:
            return version
        # legacy backends predate version metadata; trailing digits in OD framework names
        # are product names (detectron2, ultralytics8), not versions
        model_name = (metadata.get("model_name") or metadata.get("algorithm") or "").lower()
        if framework.lower() == "detectron2" or model_name == "yolov5":
            return "v0"
        return "v1"
