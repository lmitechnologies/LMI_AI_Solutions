"""Tiler configuration callback. from anomalib v2.3.3"""

import logging
from collections.abc import Sequence
from typing import Any

import lightning.pytorch as pl
from anomalib.callbacks.tiler_configuration import TilerConfigurationCallback
from anomalib.data.utils.tiler import Tiler as AnomalibTiler
from anomalib.models.components import AnomalibModule

from lmi_utils.image_utils.tiler import ScaleMode, Tiler

logger = logging.getLogger(__name__)


class CallbackTiler(Tiler):
    def __init__(
        self,
        tile_size: int | Sequence,
        stride: int | Sequence | None,
        remove_border_count: int = 0,
        mode: ScaleMode = ScaleMode.PADDING,
        **kwargs,
    ):
        # Anomalib's Tiler allows stride=None -> stride=tile_size
        # This behavior is included in this subclass instead of main
        # repo Tiler, as stride is expected to always be present in
        # main Tiler use case
        tile_size = list(AnomalibTiler.validate_size_type(tile_size))
        stride = stride or tile_size
        super().__init__(tile_size, stride, scale_mode=mode)
        self.remove_border_count = remove_border_count  # unused

    @property
    def mode(self):
        return self.scale_mode


class TilerConfigCallback(TilerConfigurationCallback):
    """Callback for configuring image tiling operations"""

    def __init__(
        self,
        enable: bool = False,
        tile_size: int | Sequence = 224,
        stride: int | Sequence | None = None,
        remove_border_count: int = 0,
        mode: ScaleMode = ScaleMode.PADDING,
        tiler_class: Any = None,
        **tiler_kwargs,
    ) -> None:
        """Initialize tiling configuration."""
        self.enable = enable
        self.tile_size = tile_size
        self.stride = stride
        self.remove_border_count = remove_border_count
        self.mode = mode
        tiler_class = globals()[tiler_class] if isinstance(tiler_class, str) else tiler_class
        self.tiler_class = tiler_class or CallbackTiler
        self.tiler_kwargs = tiler_kwargs

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        """Set Tiler object within Anomalib Model"""
        del trainer, stage  # These variables are not used.

        if not self.enable:
            return

        if isinstance(pl_module, AnomalibModule) and hasattr(pl_module.model, "tiler"):
            pl_module.model.tiler = self.tiler_class(tile_size=self.tile_size, stride=self.stride, mode=self.mode, **self.tiler_kwargs)
        else:
            msg = "Model does not support tiling."
            raise ValueError(msg)
