from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch

from .handlers import undo_resize_handler, undo_tile_handler


class Reconstructor:
    def __init__(self):
        self._undo_handlers = {}
        self.register_default_undo_handlers()

    def register_default_undo_handlers(self):
        """Registers built-in undo handlers."""
        self.register_undo_handler("tile", undo_tile_handler)
        self.register_undo_handler("resize", undo_resize_handler)

    def register_undo_handler(self, name: str, undo_func: Callable) -> None:
        """Register an undo handler with validation"""
        if not callable(undo_func):
            raise TypeError(f"Undo handler for '{name}' must be callable.")
        self._undo_handlers[name] = undo_func

    def reconstruct(
        self, processed_images: List[Union[torch.Tensor, np.ndarray]], history: List[Dict[str, Any]]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        reconstructs the original image from processed images and metadata.

        Args:
            processed_images: List of (H, W, C) tensors or numpy arrays.
            history: List of metadata dicts.

        Returns:
            torch.Tensor | np.ndarray: The reconstructed image (H, W, C).
        """
        if not processed_images:
            raise ValueError("No input images provided for reconstruction.")

        if not isinstance(processed_images, list):
            raise TypeError("input images must be a list.")

        first_type = type(processed_images[0])
        if first_type not in (torch.Tensor, np.ndarray):
            raise TypeError("Images must be torch.Tensors or np.ndarrays")
        if not all(isinstance(img, first_type) for img in processed_images):
            raise TypeError("All images must be the same type")

        # convert to tensors if needed
        is_numpy = isinstance(processed_images[0], np.ndarray)
        current_images = processed_images
        if is_numpy:
            current_images = [torch.from_numpy(img) for img in processed_images]

        # Iterate BACKWARDS through history
        required_keys = {"op", "metadata"}
        for step in reversed(history):
            if not required_keys.issubset(step.keys()):
                raise ValueError(f"Each operation step must contain keys: {required_keys}")

            op_name = step["op"]
            meta = step["metadata"]

            if op_name in self._undo_handlers:
                undo_func = self._undo_handlers[op_name]
                current_images = undo_func(current_images, meta)
            else:
                raise ValueError(f"No undo handler for {op_name}")

        if len(current_images) != 1:
            raise RuntimeError(f"Reconstruction expected returning 1 image, got {len(current_images)}")

        # Return the single root image
        if is_numpy:
            return current_images[0].cpu().numpy()
        return current_images[0]
