"""Public namespace for building preprocessing pipelines.

Forward (used with self.preprocessor.preprocess in pipeline):
    steps.cropbox(boxes=...)
    steps.resize(width=..., height=..., preserve_aspect=...)
    ...

Inverse (used with self.revert_preprocess in pipeline):
    steps.revert_cropbox(boxes=..., orig_sizes=...)
    steps.revert_resize(src_sizes=..., dst_sizes=..., pads=...)
    ...
"""

from .ops import (
    CropBoxConfig,
    CropBoxMeta,
    FlipConfig,
    FlipMeta,
    PadConfig,
    PadMeta,
    ResizeConfig,
    ResizeMeta,
    RotateConfig,
    RotateMeta,
    TileConfig,
    TileMeta,
)

# forward
cropbox = CropBoxConfig
resize = ResizeConfig
pad = PadConfig
flip = FlipConfig
rotate = RotateConfig
tile = TileConfig

# inverse
revert_cropbox = CropBoxMeta
revert_resize = ResizeMeta
revert_pad = PadMeta
revert_flip = FlipMeta
revert_rotate = RotateMeta
revert_tile = TileMeta

__all__ = [
    "cropbox",
    "resize",
    "pad",
    "flip",
    "rotate",
    "tile",
    "revert_cropbox",
    "revert_resize",
    "revert_pad",
    "revert_flip",
    "revert_rotate",
    "revert_tile",
]
