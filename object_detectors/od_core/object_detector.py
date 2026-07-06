import logging
from typing import Any, Dict

from .object_detector_registry import ObjectDetectorRegistry


class ObjectDetector:
    """Factory that instantiates the correct object-detector backend from a metadata dict.

    Only two fields are extracted from ``metadata`` and handled specially:

    - ``model_path`` (str): Forwarded as the first positional argument to the backend
      constructor. If also present in ``kwargs``, the ``kwargs`` copy is dropped to avoid
      a duplicate-argument ``TypeError``.
    - ``image_size`` (int | tuple): Forwarded via ``kwargs["image_size"]``. If supplied in
      both ``metadata`` and ``kwargs``, the ``metadata`` value takes precedence (a warning
      is logged).

    All other metadata fields (e.g. ``framework``, ``model_name``, ``task``, ``version``)
    are used only for registry lookup and are **not** forwarded to the backend constructor.
    Any additional constructor arguments must be passed explicitly via ``*args`` or ``**kwargs``.
    """

    def __new__(cls, metadata: Dict[str, Any], *args, **kwargs):
        logger = logging.getLogger(__name__)
        model_path = metadata.get("model_path")
        image_size = metadata.get("image_size")
        if "image_size" in kwargs and kwargs["image_size"] is not None:
            if image_size is not None:
                logger.warning("Both 'image_size' in metadata and kwargs provided. Using the one from metadata.")
            else:
                image_size = kwargs["image_size"]

        try:
            wrapper_cls = ObjectDetectorRegistry.get_class(metadata)
        except ValueError as e:
            raise ValueError(f"Failed to find a registered detector for metadata: {metadata}") from e

        if image_size is not None:
            kwargs["image_size"] = image_size
        else:
            logger.warning(f"No 'image_size' provided in metadata or kwargs for {wrapper_cls.__name__}. Using its default image size.")

        if model_path is not None:
            kwargs.pop("model_path", None)
            instance = wrapper_cls(model_path, *args, **kwargs)
        else:
            instance = wrapper_cls(*args, **kwargs)

        return instance
