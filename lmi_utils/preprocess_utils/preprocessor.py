from typing import Any, Callable, Dict, List, Tuple

from lmi_utils.image_utils.types import ImageBatch, ImageLike

from .base import BaseProcessor
from .handlers import resize, tile


class Preprocessor(BaseProcessor):
    """
    A class to run a dynamic pipeline of preprocessing steps on an image.

    Handlers (processing functions) are registered with the instance and called based on a list of processing steps.
    """

    def __init__(self):
        """
        Initializes the preprocessor and registers default handlers.
        """
        self._handlers = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Registers the built-in processing functions."""
        self.register_handler("resize", resize)
        self.register_handler("tile", tile)

    def register_handler(self, name: str, handler_func: Callable) -> None:
        """
        Registers a new handler function or overwrites an existing one.

        Args:
            name (str): Unique identifier for this handler (e.g., "resize", "tile").
            handler_func (callable): Processing function with signature:
                (images: list[torch.Tensor], config: dict) -> tuple[list[torch.Tensor], dict]

                - images: List of (H, W, C) torch tensors
                - config: Handler-specific configuration dictionary
                - Returns: (processed_images, metadata_dict)
                    - processed_images: List of (H, W, C) tensors
                    - metadata_dict: Must be a dict with a "metadata" key (enforced by validate_handler_metadata)

        """
        if not callable(handler_func):
            raise TypeError(f"Preprocess handler for '{name}' must be a callable function.")
        self._handlers[name] = handler_func

    def preprocess(self, images: ImageBatch, processing_steps: List[Dict[str, Any]]) -> Tuple[List[ImageLike], List[Dict[str, Any]]]:
        """
        Runs the preprocessing pipeline.

        Args:
            images: A single HW/HWC image, list of HW/HWC images, or a BHWC batch (numpy array or torch tensor). Any dtype is accepted.
            processing_steps: List of step dicts, each with keys:
                - "type" (str): Registered handler name (e.g. "resize", "tile").
                - "configuration" (dict): Handler-specific config passed as-is.

        Returns:
            processed_imgs: List of (H, W, C) images, same type as input.
            history: List of step records for reconstruction, each with keys:
                - "type" (str): Handler name.
                - "metadata" (list): Per-image metadata returned by the handler.
        """
        if isinstance(images, list):
            pass
        elif hasattr(images, "ndim") and images.ndim == 4:
            images = list(images)
        else:
            images = [images]
        self.validate_image_list(images, stage="preprocessing")
        self.validate_steps(processing_steps)

        # convert to tensor if needed
        processed_imgs, is_numpy = self.to_tensor_list(images)

        history = []
        for step in processing_steps:
            op_name = step["type"]
            config = step["configuration"]
            if op_name not in self._handlers:
                raise ValueError(f"Handler for '{op_name}' is not registered.")

            # Call the handler
            handler = self._handlers[op_name]
            new_images, metadata = handler(processed_imgs, config)

            # Validate handler output
            self.validate_image_handler_output(new_images, op_name, expected_type="preprocess handler")
            self.validate_handler_metadata(metadata, op_name)

            # Save Metadata for reconstruction
            step_record = {"type": op_name, "metadata": metadata["metadata"]}
            history.append(step_record)
            processed_imgs = new_images

        return self.from_tensor_list(processed_imgs, is_numpy), history
