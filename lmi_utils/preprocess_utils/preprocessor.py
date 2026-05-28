from typing import Any, Dict, List, Optional, Tuple, Type

from lmi_utils.image_utils.types import ImageBatch, ImageLike

from .base import BaseProcessor
from .operation import Config, Meta, Operation
from .ops import (
    CropOperation,
    FlipOperation,
    PadOperation,
    ResizeOperation,
    RotateOperation,
    TileOperation,
)


class Preprocessor(BaseProcessor):
    """Runs a typed pipeline of preprocessing Operations on an image.

    Operations are dispatched by Config type. Configs are constructed via
    ``lmi_utils.preprocess_utils.steps`` (recommended) or directly as dataclasses.
    JSON manifests should be converted via
    ``lmi_utils.preprocess_utils._parser.parse_steps`` before being passed in.

    Runtime channel:
        Ops that need caller-supplied data (e.g. crop-to-label) receive it through
        the optional ``runtime`` argument keyed by the Config's ``runtime_key`` —
        for crop-to-label that is its ``label``. Each Config's ``bind`` resolves the
        value into a concrete (executable) Config before forward dispatch.

    Device contract:
        Output tensors live on the same device as the input tensors.
    """

    _DEFAULT_OPS: Tuple[Type[Operation], ...] = (
        ResizeOperation,
        PadOperation,
        FlipOperation,
        TileOperation,
        CropOperation,
        RotateOperation,
    )

    @classmethod
    def default_ops(cls) -> Dict[Type[Config], Type[Operation]]:
        return {op_cls.config_cls: op_cls for op_cls in cls._DEFAULT_OPS}

    def __init__(self):
        self._ops: Dict[Type[Config], Operation] = {}
        for op_cls in self._DEFAULT_OPS:
            self.register(op_cls())

    def register(self, op: Operation) -> None:
        """Register an Operation under its ``config_cls``."""
        if not isinstance(op, Operation):
            raise TypeError(f"Expected Operation, got {type(op)}")
        if not getattr(op, "config_cls", None):
            raise ValueError("Operation must define `config_cls`")
        self._ops[op.config_cls] = op

    def preprocess(
        self,
        images: ImageBatch,
        configs: List[Config],
        runtime: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[List[ImageLike], List[Meta]]:
        """Run the preprocessing pipeline.

        Args:
            images: HW/HWC image, list of HW/HWC images, or BHWC batch (numpy or torch).
            configs: List of typed Config objects (one per step).
            runtime: Optional ``{label: value}`` dict for ops that consume runtime data
                (crop-to-label is keyed by its ``label``).

        Returns:
            (processed_images, history): the processed image list and a per-step
            list of typed Meta objects suitable for Reconstructor.
        """
        self._validate_configs(configs)
        self._validate_runtime(runtime, configs)

        if isinstance(images, list):
            pass
        elif hasattr(images, "ndim") and images.ndim == 4:
            images = list(images)
        else:
            images = [images]
        self.validate_image_list(images, stage="preprocessing")

        if not configs:
            return list(images), []

        processed_imgs, is_numpy = self.to_tensor_list(images)

        history: List[Meta] = []
        for cfg in configs:
            key = cfg.runtime_key
            rt = runtime.get(key, {}) if (runtime and key) else {}
            resolved = cfg.bind(rt)
            op = self._ops.get(type(resolved))
            if op is None:
                raise ValueError(f"No Operation registered for {type(resolved).__name__}")

            new_images, meta = op.forward(processed_imgs, resolved)
            self.validate_image_handler_output(new_images, type(resolved).__name__, expected_type="preprocess handler")
            if not isinstance(meta, Meta):
                raise TypeError(f"{type(op).__name__}.forward returned {type(meta)}, expected Meta")
            history.append(meta)
            processed_imgs = new_images

        return self.from_tensor_list(processed_imgs, is_numpy), history

    @staticmethod
    def _validate_configs(configs: List[Config]) -> None:
        if not isinstance(configs, list):
            raise TypeError(f"configs must be a list, got {type(configs)}")
        for i, c in enumerate(configs):
            if not isinstance(c, Config):
                raise TypeError(f"configs[{i}] must be a Config, got {type(c)}")

        keys = [c.runtime_key for c in configs if c.is_runtime]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"Duplicate crop-to-label label(s) in one chain: {keys}. Each crop-to-label 'label' must be unique within a model role."
            )

    @staticmethod
    def _validate_runtime(runtime: Optional[Dict[str, Dict[str, Any]]], configs: List[Config]) -> None:
        if runtime is None:
            return
        if not isinstance(runtime, dict):
            raise TypeError(f"runtime must be a dict keyed by step label, got {type(runtime)}")

        key_set = {c.runtime_key for c in configs if c.is_runtime}
        for key, value in runtime.items():
            if not isinstance(key, str):
                raise TypeError(f"runtime keys must be strings (step labels), got {type(key)}")
            if not isinstance(value, dict):
                raise TypeError(f"runtime[{key!r}] must be a dict, got {type(value)}")
            if key not in key_set:
                raise ValueError(f"runtime key {key!r} does not match any runtime step's label.")
