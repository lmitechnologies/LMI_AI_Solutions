from typing import Dict, Any
from .anomaly_detector_registry import AnomalyDetectorRegistry
import logging

class AnomalyDetector:

    def __new__(cls, metadata: Dict[str, Any], *args, **kwargs):
        logger = logging.getLogger(__name__)
        model_path = metadata.get('model_path')
        image_size = metadata.get('image_size')
        tile_size = metadata.get('tile_size', args[0] if len(args) > 0 else None)
        stride = metadata.get('stride', args[1] if len(args) > 1 else None)
        tile_mode = metadata.get('tile_mode', args[2] if len(args) > 2 else "padding")

        if image_size:
            if 'image_size' in kwargs and kwargs['image_size'] is not None:
                logger.warning(
                    "Both 'image_size' in metadata and kwargs provided. "
                    "Using the one from metadata."
                )
            
        kwargs['image_size'] = image_size
        
        try:
            wrapper_cls = AnomalyDetectorRegistry.get_class(metadata)
        except ValueError as e:
            raise ValueError(f"Failed to find a registered detector for metadata: {metadata}") from e

        if model_path is not None:
            if len(args) > 4:
                logger.warning(
                    "Both 'model_path' in metadata and positional arguments provided. All positional arguments will be ignored."
                )
            instance = wrapper_cls(
                model_path,
                tile_size=tile_size,
                stride=stride,
                tile_mode=tile_mode,
                **kwargs
            )
        else:
            instance = wrapper_cls(*args, **kwargs)

        return instance