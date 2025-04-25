from typing import Dict, Any
from .od_base import ODBase
from .object_detector_registry import ObjectDetectorRegistry

class ObjectDetector(ODBase):

    def __new__(cls, metadata: Dict[str, Any], *args, **kwargs):
        model_path = metadata.get('model_path') 

        try:
            wrapper_cls = ObjectDetectorRegistry.get_class(metadata)
        except ValueError as e:
            raise ValueError(f"Failed to find a registered detector for metadata: {metadata}") from e

        if model_path is not None:
            instance = wrapper_cls(model_path, *args, **kwargs)
        else:
            instance = wrapper_cls(*args, **kwargs)

        return instance