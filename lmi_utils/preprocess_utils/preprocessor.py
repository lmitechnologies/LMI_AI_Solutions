from typing import Any, Dict, List, Optional, Tuple

from lmi_utils.image_utils.types import ImageBatch, ImageLike

from .base import BaseProcessor
from .operation import Operation
from .ops import (
    CropOperation,
    CropToLabelOperation,
    FlipOperation,
    PadOperation,
    ResizeOperation,
    TileOperation,
)


class Preprocessor(BaseProcessor):
    """
    Runs a dynamic pipeline of preprocessing Operations on an image.

    Operations are registered as `Operation` instances and selected per-step
    by name. Each step's metadata is recorded so a `Reconstructor` can later
    invert the pipeline.

    Runtime channel:
        Ops that need caller-supplied data (e.g. crop-to-label needs the box
        from an upstream detector) receive it through the optional `runtime`
        argument to `preprocess`. `runtime` is a `{id: value}` dict whose
        keys must match the `id` of the target manifest step; the matched op's
        `bind` method resolves the value into a concrete step before forward
        dispatch.

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
        self.register(PadOperation())
        self.register(FlipOperation())
        self.register(TileOperation())
        self.register(CropOperation())
        self.register(CropToLabelOperation())

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

    def preprocess(
        self,
        images: ImageBatch,
        processing_steps: List[Dict[str, Any]],
        runtime: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[List[ImageLike], List[Dict[str, Any]]]:
        """
        Runs the preprocessing pipeline.

        Args:
            images: A single HW/HWC image, list of HW/HWC images, or a BHWC batch (numpy array or torch tensor). Any dtype is accepted.
            processing_steps: List of step dicts, each with keys:
                - "type" (str): Registered Operation name (e.g. "resize", "tile", "crop-to-label").
                - "configuration" (dict): Op-specific config passed as-is.
                - "id" (str, optional): Unique step identifier. Required for steps targeted by `runtime`.
            runtime: Optional `{id: value}` dict. Each key must match a step's `id`;
                Supported values by op type:
                  - "crop-to-label": {"boxes": [[x1, y1, x2, y2], ...]} —
                    per-image boxes in original-image coordinates supplied by an upstream detector.

        Returns:
            processed_imgs: List of (H, W, C) images, same type as input.
            history: List of step records for reconstruction, each with keys:
                - "type" (str): Resolved Operation name (after bind).
                - "metadata" (list): Per-image metadata returned by the op.
                - "id" (str, optional): Preserved from the manifest step.
        """
        if isinstance(images, list):
            pass
        elif hasattr(images, "ndim") and images.ndim == 4:
            images = list(images)
        else:
            images = [images]
        self.validate_image_list(images, stage="preprocessing")
        self.validate_steps(processing_steps)
        self._validate_runtime(runtime, processing_steps)

        if not processing_steps:
            return list(images), []

        processed_imgs, is_numpy = self.to_tensor_list(images)

        history = []
        for step in processing_steps:
            op_name = step["type"]
            if op_name not in self._ops:
                raise ValueError(f"Operation '{op_name}' is not registered.")
            op = self._ops[op_name]

            step_id = step.get("id")
            runtime_value = runtime.get(step_id, {}) if runtime and step_id is not None else {}
            resolved = op.bind(step, runtime_value)
            resolved_name = resolved.get("type")
            if resolved_name not in self._ops:
                raise ValueError(f"Op '{op_name}'.bind produced unknown type '{resolved_name}'.")
            resolved_op = self._ops[resolved_name]
            config = resolved.get("configuration", {})

            new_images, metadata = resolved_op.forward(processed_imgs, config)

            self.validate_image_handler_output(new_images, resolved_name, expected_type="preprocess handler")
            self.validate_handler_metadata(metadata, resolved_name)

            history_entry: Dict[str, Any] = {"type": resolved_name, "metadata": metadata}
            if resolved.get("id") is not None:
                history_entry["id"] = resolved["id"]
            history.append(history_entry)
            processed_imgs = new_images

        return self.from_tensor_list(processed_imgs, is_numpy), history

    @staticmethod
    def _validate_runtime(
        runtime: Optional[Dict[str, Dict[str, Any]]],
        processing_steps: List[Dict[str, Any]],
    ) -> None:
        if runtime is None:
            return
        if not isinstance(runtime, dict):
            raise TypeError(f"runtime must be a dict keyed by step id, got {type(runtime)}")

        step_ids = {step["id"] for step in processing_steps if step.get("id") is not None}
        if len(step_ids) != sum(1 for step in processing_steps if step.get("id") is not None):
            raise ValueError("processing_steps contains duplicate 'id' values; each step's id must be unique.")

        for key, value in runtime.items():
            if not isinstance(key, str):
                raise TypeError(f"runtime keys must be strings (step ids), got {type(key)}")
            if not isinstance(value, dict):
                raise TypeError(f"runtime[{key!r}] must be a dict, got {type(value)}")
            if key not in step_ids:
                raise ValueError(f"runtime key {key!r} does not match any preprocessing step's 'id'.")
