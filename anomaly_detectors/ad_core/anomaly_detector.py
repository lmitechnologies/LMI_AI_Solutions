from typing import Dict, Any
from anomalib_lmi.base import Anomalib_Base
from .anomaly_detector_registry import AnomalyDetectorRegistry
import logging

class AnomalyDetector(Anomalib_Base):

    def __new__(cls, metadata: Dict[str, Any], *args, **kwargs):
        logger = logging.getLogger(__name__)
        model_path = metadata.get('model_path')
        image_size = metadata.get('image_size')
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
            instance = wrapper_cls(model_path, *args, **kwargs)
        else:
            instance = wrapper_cls(*args, **kwargs)

        return instance