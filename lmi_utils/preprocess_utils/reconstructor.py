from typing import Any, Callable, Dict, List

from lmi_utils.image_utils.types import ImageLike

from .base import BaseProcessor
from .handlers import revert_resize, revert_tile


class Reconstructor(BaseProcessor):
    """
    A class to reconstruct original images from processed images using undo handlers.
    """

    def __init__(self):
        self._undo_handlers = {}
        self._register_default_undo_handlers()

    def _register_default_undo_handlers(self):
        """Registers built-in undo handlers."""
        self.register_undo_handler("tile", revert_tile)
        self.register_undo_handler("resize", revert_resize)

    def register_undo_handler(self, name: str, undo_func: Callable) -> None:
        """
        Register an undo handler for a specific operation.

        Args:
            name (str): Operation name (must match forward handler name).
            undo_func (callable): Undo function with signature:
                (images: list[torch.Tensor], metadata: list) -> list[torch.Tensor]

                - images: List of processed (H, W, C) tensors
                - metadata: Per-image metadata list from the history step
                - Returns: List of (H, W, C) tensors
        """
        if not callable(undo_func):
            raise TypeError(f"Undo handler for '{name}' must be callable.")
        self._undo_handlers[name] = undo_func

    def reconstruct(self, images: List[ImageLike], steps: List[Dict[str, Any]]) -> List[ImageLike]:
        """
        Reconstructs the original image from images and history.

        Args:
            images: List of (H, W, C) tensors or numpy arrays.
            steps: List of configuration dicts.

        Returns:
            List: The reconstructed image(s) (H, W, C).
        """
        self.validate_image_list(images, stage="reconstruction")
        self.validate_history_steps(steps)

        restored_images, is_numpy = self.to_tensor_list(images)

        # Iterate BACKWARDS through steps
        for step in reversed(steps):
            op_name = step["type"]
            meta = step["metadata"]
            if op_name not in self._undo_handlers:
                raise ValueError(f"Undo handler for '{op_name}' is not registered.")

            # run the undo operation
            undo_func = self._undo_handlers[op_name]
            restored_images = undo_func(restored_images, meta)
            self.validate_handler_output(restored_images, op_name, expected_type="undo handler")

        return self.from_tensor_list(restored_images, is_numpy)
