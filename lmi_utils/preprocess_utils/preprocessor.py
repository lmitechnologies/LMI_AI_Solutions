from typing import Any, Dict, List, Tuple

from lmi_utils.image_utils.types import ImageBatch, ImageLike

from .base import BaseProcessor
from .operation import Operation
from .ops import ResizeOperation, TileOperation


class Preprocessor(BaseProcessor):
    """
    Runs a dynamic pipeline of preprocessing Operations on an image.

    Operations are registered as `Operation` instances and selected per-step
    by name. Each step's metadata is recorded so a `Reconstructor` can later
    invert the pipeline.

    Device contract:
        Output tensors live on the same device as the input tensors. Every
        Operation must allocate any internal tensors with `device=<input>.device`
        and avoid implicit `.cpu()` / `.cuda()` moves. Numpy inputs are bridged
        through CPU tensors (numpy is CPU-only by definition).
    """

    def __init__(self):
        self._ops: Dict[str, Operation] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(ResizeOperation())
        self.register(TileOperation())

    def register(self, op: Operation) -> None:
        """
        Register an Operation. Pairs with `Reconstructor.register(op)`.

        Args:
            op: An `Operation` instance with a non-empty `name`.
        """
        if not isinstance(op, Operation):
            raise TypeError(f"Expected Operation, got {type(op)}")
        if not op.name:
            raise ValueError("Operation must define a non-empty `name`")
        self._ops[op.name] = op

    def preprocess(self, images: ImageBatch, processing_steps: List[Dict[str, Any]]) -> Tuple[List[ImageLike], List[Dict[str, Any]]]:
        """
        Runs the preprocessing pipeline.

        Args:
            images: A single HW/HWC image, list of HW/HWC images, or a BHWC batch (numpy array or torch tensor). Any dtype is accepted.
            processing_steps: List of step dicts, each with keys:
                - "type" (str): Registered Operation name (e.g. "resize", "tile").
                - "configuration" (dict): Op-specific config passed as-is.

        Returns:
            processed_imgs: List of (H, W, C) images, same type as input.
            history: List of step records for reconstruction, each with keys:
                - "type" (str): Operation name.
                - "metadata" (list): Per-image metadata returned by the op.
        """
        if isinstance(images, list):
            pass
        elif hasattr(images, "ndim") and images.ndim == 4:
            images = list(images)
        else:
            images = [images]
        self.validate_image_list(images, stage="preprocessing")
        self.validate_steps(processing_steps)

        processed_imgs, is_numpy = self.to_tensor_list(images)

        history = []
        for step in processing_steps:
            op_name = step["type"]
            config = step["configuration"]
            if op_name not in self._ops:
                raise ValueError(f"Operation '{op_name}' is not registered.")

            op = self._ops[op_name]
            new_images, metadata = op.forward(processed_imgs, config)

            self.validate_image_handler_output(new_images, op_name, expected_type="preprocess handler")
            self.validate_handler_metadata(metadata, op_name)

            history.append({"type": op_name, "metadata": metadata})
            processed_imgs = new_images

        return self.from_tensor_list(processed_imgs, is_numpy), history
