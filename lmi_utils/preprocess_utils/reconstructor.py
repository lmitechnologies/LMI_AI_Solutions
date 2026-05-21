from typing import Any, Dict, List

from lmi_utils.image_utils.types import ImageLike

from .base import BaseProcessor
from .operation import Operation
from .ops import (
    CropOperation,
    FlipOperation,
    PadOperation,
    ResizeOperation,
    RotateOperation,
    TileOperation,
)


class Reconstructor(BaseProcessor):
    """
    Reconstructs original images and coordinates from preprocessed data.

    Mirrors the `Preprocessor`: register the same `Operation` instance on
    both, and image / coordinate reverts come from a single object.

    Device contract:
        Output tensors live on the same device as the input tensors. Every
        Operation must allocate any internal tensors with `device=<input>.device`
        and avoid implicit `.cpu()` / `.cuda()` moves.
    """

    def __init__(self):
        self._ops: Dict[str, Operation] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(ResizeOperation())
        self.register(PadOperation())
        self.register(FlipOperation())
        self.register(TileOperation())
        self.register(CropOperation())
        self.register(RotateOperation())

    def register(self, op: Operation) -> None:
        """
        Register an Operation. Pairs with `Preprocessor.register(op)`.

        Image-space-only ops can rely on the default identity revert_coords;
        config-only ops can rely on the default identity revert_images.
        """
        if not isinstance(op, Operation):
            raise TypeError(f"Expected Operation, got {type(op)}")
        if not op.name:
            raise ValueError("Operation must define a non-empty `name`")
        self._ops[op.name] = op

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
            if op_name not in self._ops:
                raise ValueError(f"Revert coordinate handler for '{op_name}' is not registered.")
            input_populated = self._populated_coord_fields(per_image)
            per_image = self._ops[op_name].revert_coords(per_image, step["metadata"])
            self.validate_coord_handler_output(per_image, op_name, input_populated=input_populated)

        per_image = self.from_tensor_results(per_image, is_numpy)
        keys = per_image[0].keys()
        return {k: [d[k] for d in per_image] for k in keys}

    def apply_coordinates(self, results: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Forward-applies a history of geometric ops to original-space coordinates.

        Inverse of :meth:`reconstruct_coordinates`: takes coords in original image space
        and returns coords in the post-preprocessing space.

        Args:
            results: Batch results dict where each value is a per-image list.
            steps: Preprocessing history (same shape as for reconstruct_coordinates).

        Returns:
            Results dict with coordinates mapped into the preprocessed space.
        """
        self.validate_history_steps(steps)
        if not results or not steps:
            return results

        first_val = next(iter(results.values()))
        n = len(first_val)
        per_image = [{k: v[i] for k, v in results.items()} for i in range(n)]
        per_image, is_numpy = self.to_tensor_results(per_image)

        for step in steps:
            op_name = step["type"]
            if op_name not in self._ops:
                raise ValueError(f"Apply coordinate handler for '{op_name}' is not registered.")
            input_populated = self._populated_coord_fields(per_image)
            per_image = self._ops[op_name].apply_coords(per_image, step["metadata"])
            self.validate_coord_handler_output(per_image, op_name, input_populated=input_populated)

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

        if not steps:
            return list(images)

        restored_images, is_numpy = self.to_tensor_list(images)

        for step in reversed(steps):
            op_name = step["type"]
            if op_name not in self._ops:
                raise ValueError(f"Revert image handler for '{op_name}' is not registered.")

            restored_images = self._ops[op_name].revert_images(restored_images, step["metadata"])
            self.validate_image_handler_output(restored_images, op_name, expected_type="revert image handler")

        return self.from_tensor_list(restored_images, is_numpy)
