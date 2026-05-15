from typing import Any, Dict, List, Optional, Tuple

from lmi_utils.image_utils.types import ImageBatch, ImageLike

from .base import BaseProcessor
from .operation import Operation
from .ops import CropOperation, CropToLabelOperation, ResizeOperation, TileOperation


class Preprocessor(BaseProcessor):
    """
    Runs a dynamic pipeline of preprocessing Operations on an image.

    Operations are registered as `Operation` instances and selected per-step
    by name. Each step's metadata is recorded so a `Reconstructor` can later
    invert the pipeline.

    Runtime channel:
        Ops that need caller-supplied data (e.g. crop-to-label needs the box
        from an upstream detector) receive it through the optional `runtime`
        argument to `preprocess`. Each runtime patch is a step record
        `{"type": <op>, "instance"?: <name>, "runtime": {...}}` matched to a
        manifest step by `(type, instance)`. The matched op's `bind` method
        resolves the patch into a concrete step before forward dispatch.

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
        runtime: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[ImageLike], List[Dict[str, Any]]]:
        """
        Runs the preprocessing pipeline.

        Args:
            images: A single HW/HWC image, list of HW/HWC images, or a BHWC batch (numpy array or torch tensor). Any dtype is accepted.
            processing_steps: List of step dicts, each with keys:
                - "type" (str): Registered Operation name (e.g. "resize", "tile", "crop-to-label").
                - "configuration" (dict): Op-specific config passed as-is.
                - "instance" (str, optional): Disambiguates multiple instances of the same op type.
            runtime: Optional list of runtime patches, each:
                - "type" (str): Must match a step's type.
                - "instance" (str, optional): Must match a step's instance when present.
                - "runtime" (dict): Op-specific payload consumed by the op's `bind`.

        Returns:
            processed_imgs: List of (H, W, C) images, same type as input.
            history: List of step records for reconstruction, each with keys:
                - "type" (str): Resolved Operation name (after bind).
                - "metadata" (list): Per-image metadata returned by the op.
                - "instance" (str, optional): Preserved from the manifest step.
        """
        if isinstance(images, list):
            pass
        elif hasattr(images, "ndim") and images.ndim == 4:
            images = list(images)
        else:
            images = [images]
        self.validate_image_list(images, stage="preprocessing")
        self.validate_steps(processing_steps)
        self._validate_runtime(runtime)

        runtime_index = self._index_runtime(processing_steps, runtime)

        processed_imgs, is_numpy = self.to_tensor_list(images)

        history = []
        for idx, step in enumerate(processing_steps):
            op_name = step["type"]
            if op_name not in self._ops:
                raise ValueError(f"Operation '{op_name}' is not registered.")
            op = self._ops[op_name]

            patch_payload = runtime_index.get(idx, {})
            resolved = op.bind(step, patch_payload)
            resolved_name = resolved.get("type")
            if resolved_name not in self._ops:
                raise ValueError(f"Op '{op_name}'.bind produced unknown type '{resolved_name}'.")
            resolved_op = self._ops[resolved_name]
            config = resolved.get("configuration", {})

            new_images, metadata = resolved_op.forward(processed_imgs, config)

            self.validate_image_handler_output(new_images, resolved_name, expected_type="preprocess handler")
            self.validate_handler_metadata(metadata, resolved_name)

            history_entry: Dict[str, Any] = {"type": resolved_name, "metadata": metadata}
            if resolved.get("instance") is not None:
                history_entry["instance"] = resolved["instance"]
            history.append(history_entry)
            processed_imgs = new_images

        return self.from_tensor_list(processed_imgs, is_numpy), history

    @staticmethod
    def _validate_runtime(runtime: Optional[List[Dict[str, Any]]]) -> None:
        if runtime is None:
            return
        if not isinstance(runtime, list):
            raise TypeError(f"runtime must be a list of patches, got {type(runtime)}")
        for i, patch in enumerate(runtime):
            if not isinstance(patch, dict):
                raise TypeError(f"runtime[{i}] must be a dict, got {type(patch)}")
            if "type" not in patch:
                raise ValueError(f"runtime[{i}] missing required key 'type'")
            if "runtime" in patch and not isinstance(patch["runtime"], dict):
                raise TypeError(f"runtime[{i}]['runtime'] must be a dict, got {type(patch['runtime'])}")

    @staticmethod
    def _index_runtime(
        processing_steps: List[Dict[str, Any]],
        runtime: Optional[List[Dict[str, Any]]],
    ) -> Dict[int, Dict[str, Any]]:
        """Match each runtime patch to its step index. Returns {step_idx: payload_dict}."""
        if not runtime:
            return {}

        # Bucket step indices by (type, instance) and by type alone.
        by_key: Dict[Tuple[str, Optional[str]], List[int]] = {}
        by_type: Dict[str, List[int]] = {}
        for idx, step in enumerate(processing_steps):
            key = (step["type"], step.get("instance"))
            by_key.setdefault(key, []).append(idx)
            by_type.setdefault(step["type"], []).append(idx)

        result: Dict[int, Dict[str, Any]] = {}
        consumed: set = set()
        for i, patch in enumerate(runtime):
            ptype = patch["type"]
            pinstance = patch.get("instance")
            payload = patch.get("runtime", {})

            if pinstance is not None:
                candidates = [idx for idx in by_key.get((ptype, pinstance), []) if idx not in consumed]
            else:
                candidates = [idx for idx in by_type.get(ptype, []) if idx not in consumed]

            if not candidates:
                raise ValueError(f"runtime[{i}] (type='{ptype}', instance={pinstance!r}) does not match any preprocessing step.")
            if pinstance is None and len(candidates) > 1:
                raise ValueError(
                    f"runtime[{i}] (type='{ptype}') is ambiguous: matches {len(candidates)} steps. Add 'instance' to disambiguate."
                )

            target = candidates[0]
            consumed.add(target)
            result[target] = payload

        return result
