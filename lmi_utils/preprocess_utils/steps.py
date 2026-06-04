"""Public namespace for building preprocessing pipelines.

Forward (used with self.preprocessor.preprocess in pipeline):
    steps.crop(boxes=...)
    steps.resize(width=..., height=..., preserve_aspect=...)
    ...

Inverse (used with self.revert_preprocess in pipeline):
    steps.revert_crop(boxes=..., orig_sizes=...)
    steps.revert_resize(src_sizes=..., dst_sizes=..., pads=...)
    ...
"""

from .ops import (
    CropConfig,
    CropMeta,
    FlipConfig,
    FlipMeta,
    PadConfig,
    PadMeta,
    ResizeConfig,
    ResizeMeta,
    TileConfig,
    TileMeta,
)

# forward
crop = CropConfig
resize = ResizeConfig
pad = PadConfig
flip = FlipConfig
tile = TileConfig

# inverse
revert_crop = CropMeta
revert_resize = ResizeMeta
revert_pad = PadMeta
revert_flip = FlipMeta
revert_tile = TileMeta

__all__ = [
    "crop",
    "resize",
    "pad",
    "flip",
    "tile",
    "revert_crop",
    "revert_resize",
    "revert_pad",
    "revert_flip",
    "revert_tile",
]
