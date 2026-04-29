from typing import Any, Callable, Dict, List

from lmi_utils.image_utils.types import ImageLike

from .base import BaseProcessor
from .handlers import revert_resize, revert_resize_coords, revert_tile, revert_tile_coords


class Reconstructor(BaseProcessor):
    """
    A class to reconstruct original images/coordinates from preprocessed data.
    """

    def __init__(self):
        self._revert_images_handlers = {}
        self._revert_coord_handlers = {}
        self._register_default_images_handlers()
        self._register_default_coord_handlers()

    def _register_default_images_handlers(self):
        """Registers built-in revert images handlers."""
        self.register_images_handler("tile", revert_tile)
        self.register_images_handler("resize", revert_resize)

    def _register_default_coord_handlers(self):
        """Registers built-in revert coordinate handlers."""
        self.register_coord_handler("resize", revert_resize_coords)
        self.register_coord_handler("tile", revert_tile_coords)

    def register_images_handler(self, name: str, revert_func: Callable) -> None:
        """
        Register a revert images handler for a specific operation.

        Args:
            name (str): Operation name (must match forward handler name).
            revert_func (callable): revert function with signature:
                (images: list[torch.Tensor], metadata: list) -> list[torch.Tensor]

                - images: List of processed (H, W, C) tensors
                - metadata: Per-image metadata list from the history step
                - Returns: List of (H, W, C) tensors
        """
        if not callable(revert_func):
            raise TypeError(f"Revert image handler for '{name}' must be callable.")
        self._revert_images_handlers[name] = revert_func

    def register_coord_handler(self, name: str, revert_func: Callable) -> None:
        """
        Register a revert coordinate handler for a specific operation.

        Args:
            name (str): Operation name (must match forward handler name).
            revert_func (callable): revert function with signature:
                (results: list[dict], metadata: list) -> list[dict]

                - results: List of per-image result dicts (boxes, masks, segments, points)
                - metadata: Per-image metadata list from the history step
                - Returns: List of per-image result dicts with reverted coordinates
        """
        if not callable(revert_func):
            raise TypeError(f"Revert coordinate handler for '{name}' must be callable.")
        self._revert_coord_handlers[name] = revert_func

    def reconstruct_coordinates(self, results: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reverts predicted coordinates to the original pre-preprocessing space.

        Args:
            results: Batch results dict where each value is a per-image list.
                     Keys: boxes, scores, classes, masks, segments, points.
            steps: Preprocessing history returned by Preprocessor.preprocess.

        Returns:
            Results dict with coordinates reverted to original image space.
        """
        self.validate_history_steps(steps)
        if not results or not steps:
            return results

        first_val = next(iter(results.values()))
        n = len(first_val)
        per_image = [{k: v[i] for k, v in results.items()} for i in range(n)]
        per_image, is_numpy = self.to_tensor_results(per_image)

        for step in reversed(steps):
            op_name = step["type"]
            if op_name not in self._revert_coord_handlers:
                raise ValueError(f"Revert coordinate handler for '{op_name}' is not registered.")
            per_image = self._revert_coord_handlers[op_name](per_image, step["metadata"])
            self.validate_coord_handler_output(per_image, op_name)

        per_image = self.from_tensor_results(per_image, is_numpy)
        keys = per_image[0].keys()
        return {k: [d[k] for d in per_image] for k in keys}

    def reconstruct_images(self, images: List[ImageLike], steps: List[Dict[str, Any]]) -> List[ImageLike]:
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
            if op_name not in self._revert_images_handlers:
                raise ValueError(f"Revert image handler for '{op_name}' is not registered.")

            # run the revert operation
            revert_func = self._revert_images_handlers[op_name]
            restored_images = revert_func(restored_images, meta)
            self.validate_image_handler_output(restored_images, op_name, expected_type="revert image handler")

        return self.from_tensor_list(restored_images, is_numpy)
