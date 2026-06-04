from typing import Dict, List, Tuple, Type

from lmi_utils.image_utils.types import ImageBatch, ImageLike

from .base import BaseProcessor
from .operation import Config, Meta, Operation
from .ops import (
    CropOperation,
    FlipOperation,
    PadOperation,
    ResizeOperation,
    TileOperation,
)


class Preprocessor(BaseProcessor):
    """Runs a typed pipeline of preprocessing Operations on an image.

    Operations are dispatched by Config type. Configs are constructed via
    ``lmi_utils.preprocess_utils.steps`` (recommended) or directly as dataclasses.
    JSON manifests should be converted via
    ``lmi_utils.preprocess_utils._parser.parse_steps`` before being passed in.

    Device contract:
        Output tensors live on the same device as the input tensors.
    """

    _DEFAULT_OPS: Tuple[Type[Operation], ...] = (
        ResizeOperation,
        PadOperation,
        FlipOperation,
        TileOperation,
        CropOperation,
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
    ) -> Tuple[List[ImageLike], List[Meta]]:
        """Run the preprocessing pipeline.

        Args:
            images: HW/HWC image, list of HW/HWC images, or BHWC batch (numpy or torch).
            configs: List of typed Config objects (one per step).

        Returns:
            (processed_images, history): the processed image list and a per-step
            list of typed Meta objects suitable for Reconstructor.
        """
        self._validate_configs(configs)

        images = self.as_image_list(images)
        self.validate_image_list(images, stage="preprocessing")

        if not configs:
            return list(images), []

        processed_imgs, is_numpy = self.to_tensor_list(images)

        history: List[Meta] = []
        for cfg in configs:
            op = self._ops.get(type(cfg))
            if op is None:
                raise ValueError(f"No Operation registered for {type(cfg).__name__}")

            new_images, meta = op.forward(processed_imgs, cfg)
            self.validate_image_handler_output(new_images, type(cfg).__name__, expected_type="preprocess handler")
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
