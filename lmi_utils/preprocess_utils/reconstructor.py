from typing import Any, Dict, List, Tuple, Type

from lmi_utils.image_utils.types import ImageLike

from .base import BaseProcessor
from .operation import Meta, Operation
from .ops import (
    CropBoxOperation,
    FlipOperation,
    PadOperation,
    ResizeOperation,
    TileOperation,
)


class Reconstructor(BaseProcessor):
    """Reconstructs original images and coordinates from preprocessed data.

    Operations are dispatched by Meta type. The Reconstructor mirrors the
    Preprocessor: register the same Operation classes on both.

    Device contract:
        Output tensors live on the same device as the input tensors.
    """

    _DEFAULT_OPS: Tuple[Type[Operation], ...] = (
        ResizeOperation,
        PadOperation,
        FlipOperation,
        TileOperation,
        CropBoxOperation,
    )

    @classmethod
    def default_ops(cls) -> Dict[Type[Meta], Type[Operation]]:
        return {op_cls.meta_cls: op_cls for op_cls in cls._DEFAULT_OPS}

    def __init__(self):
        self._ops: Dict[Type[Meta], Operation] = {}
        for op_cls in self._DEFAULT_OPS:
            self.register(op_cls())

    def register(self, op: Operation) -> None:
        if not isinstance(op, Operation):
            raise TypeError(f"Expected Operation, got {type(op)}")
        if not getattr(op, "meta_cls", None):
            raise ValueError("Operation must define `meta_cls`")
        self._ops[op.meta_cls] = op

    def reconstruct_coordinates(self, results: Dict[str, Any], history: List[Meta]) -> Dict[str, Any]:
        """Revert predicted coordinates to original pre-preprocessing space."""
        return self._transform_coordinates(results, history, reverse=True)

    def apply_coordinates(self, results: Dict[str, Any], history: List[Meta]) -> Dict[str, Any]:
        """Forward-apply geometric ops to original-space coordinates."""
        return self._transform_coordinates(results, history, reverse=False)

    def _transform_coordinates(self, results: Dict[str, Any], history: List[Meta], *, reverse: bool) -> Dict[str, Any]:
        """Run coord ops over ``history`` (reversed for revert, forward for apply)."""
        self._validate_history(history)
        if not results or not history:
            return results

        first_val = next(iter(results.values()))
        n = len(first_val)
        per_image = [{k: v[i] for k, v in results.items()} for i in range(n)]
        per_image, is_numpy = self.to_tensor_results(per_image)

        ordered = reversed(history) if reverse else history
        input_populated = self._populated_coord_fields(per_image)
        for meta in ordered:
            op = self._ops.get(type(meta))
            if op is None:
                raise ValueError(f"No Operation registered for meta {type(meta).__name__}")
            transform = op.revert_coords if reverse else op.apply_coords
            per_image = transform(per_image, meta)
            input_populated = self.validate_coord_handler_output(per_image, type(meta).__name__, input_populated=input_populated)

        per_image = self.from_tensor_results(per_image, is_numpy)
        keys = per_image[0].keys()
        return {k: [d[k] for d in per_image] for k in keys}

    def reconstruct_images(self, images: List[ImageLike], history: List[Meta]) -> List[ImageLike]:
        """Reconstruct original images from preprocessed images and history."""
        self.validate_image_list(images, stage="reconstruction")
        self._validate_history(history)

        if not history:
            return list(images)

        restored, is_numpy = self.to_tensor_list(images)

        for meta in reversed(history):
            op = self._ops.get(type(meta))
            if op is None:
                raise ValueError(f"No Operation registered for meta {type(meta).__name__}")
            restored = op.revert_images(restored, meta)
            self.validate_image_handler_output(restored, type(meta).__name__, expected_type="revert image handler")

        return self.from_tensor_list(restored, is_numpy)

    @staticmethod
    def _validate_history(history: List[Meta]) -> None:
        if not isinstance(history, list):
            raise TypeError(f"history must be a list, got {type(history)}")
        for i, m in enumerate(history):
            if not isinstance(m, Meta):
                raise TypeError(f"history[{i}] must be a Meta instance, got {type(m)}")
