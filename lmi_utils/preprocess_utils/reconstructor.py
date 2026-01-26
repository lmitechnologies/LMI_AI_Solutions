from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch

from .base import BaseProcessor
from .handlers import undo_resize, undo_tile


class Reconstructor(BaseProcessor):
    """
    A class to reconstruct original images from processed images using undo handlers.
    """

    _STEP_REQUIRED_KEYS = {"op", "metadata"}

    def __init__(self):
        self._undo_handlers = {}
        self._register_default_undo_handlers()

    def _register_default_undo_handlers(self):
        """Registers built-in undo handlers."""
        self.register_undo_handler("tile", undo_tile)
        self.register_undo_handler("resize", undo_resize)

    def register_undo_handler(self, name: str, undo_func: Callable) -> None:
        """
        Register an undo handler for a specific operation.

        Args:
            name (str): Operation name (must match forward handler name).
            undo_func (callable): Undo function with signature:
                (images: list[torch.Tensor], metadata: dict) -> list[torch.Tensor]

                - images: List of processed (H, W, C) tensors
                - metadata: Dictionary containing operation metadata
                - Returns: List of (H, W, C) tensors
        """
        if not callable(undo_func):
            raise TypeError(f"Undo handler for '{name}' must be callable.")
        self._undo_handlers[name] = undo_func

    def reconstruct(
        self, processed_images: List[Union[torch.Tensor, np.ndarray]], history: List[Dict[str, Any]]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstructs the original image from processed images and metadata.

        Args:
            processed_images: List of (H, W, C) tensors or numpy arrays.
            history: List of metadata dicts.

        Returns:
            torch.Tensor | np.ndarray: The reconstructed image (H, W, C).
        """
        self.validate_image_list(processed_images, stage="reconstruction")

        current_images, is_numpy = self.to_tensor_list(processed_images)

        # Iterate BACKWARDS through history
        for step in reversed(history):
            self.validate_step_keys(step, self._STEP_REQUIRED_KEYS)

            op_name = step["op"]
            meta = step["metadata"]
            if op_name not in self._undo_handlers:
                raise ValueError(f"Undo handler for '{op_name}' is not registered.")

            # run the undo operation
            undo_func = self._undo_handlers[op_name]
            current_images = undo_func(current_images, meta)

            self.validate_handler_output(current_images, op_name, expected_type="undo handler")

        if len(current_images) != 1:
            raise RuntimeError(f"Reconstruction expected 1 image, got {len(current_images)}")

        # Return the single root image
        return self.from_tensor_list(current_images, is_numpy)[0]
