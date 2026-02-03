from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch

from .base import BaseProcessor
from .handlers import resize, tile


class Preprocessor(BaseProcessor):
    """
    A class to run a dynamic pipeline of preprocessing steps on an image.

    Handlers (processing functions) are registered with the instance and
    called based on a list of processing steps.
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
                    - metadata_dict: Dictionary containing operation metadata

        """
        if not callable(handler_func):
            raise TypeError(f"Handler for '{name}' must be a callable function.")
        self._handlers[name] = handler_func

    def preprocess(
        self, images: Union[List[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor], processing_steps: List[Dict[str, Any]]
    ) -> Tuple[List[Union[np.ndarray, torch.Tensor]], List[Dict[str, Any]]]:
        """
        Runs the preprocessing pipeline.

        Args:
            images (np.ndarray | torch.Tensor | list): Input image(s) in format (H, W, C).
            processing_steps (list): List of config dictionaries.

        Returns:
            processed_imgs (list[np.ndarray | torch.Tensor]): list of (H, W, C).
            history (list[dict]): Metadata chain for reconstruction.
        """
        if not isinstance(images, list):
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
            self.validate_handler_output(new_images, op_name, expected_type="handler")
            if not isinstance(metadata, dict):
                raise TypeError(f"Handler '{op_name}' must return metadata as dict, got {type(metadata)}")

            # Save Metadata for reconstruction
            step_record = {"type": op_name, "configuration": metadata}
            history.append(step_record)
            processed_imgs = new_images

        return self.from_tensor_list(processed_imgs, is_numpy), history
