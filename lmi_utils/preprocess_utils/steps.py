"""Typed builders for preprocessing step dicts.

Each function returns a step dict matching the v3 manifest shape
(``{"type": str, "configuration": dict, "id"?: str}``) so the result drops
directly into ``Preprocessor.preprocess(images, steps, ...)``.

Builders live as ``build_step`` classmethods on each Operation; this module
just re-exports them under their op names for a single import surface:

    from lmi_utils.preprocess_utils import steps
    steps.resize(width=640, height=640, preserve_aspect=True)
"""

from .ops import (
    CropOperation,
    CropToLabelOperation,
    FlipOperation,
    PadOperation,
    ResizeOperation,
    TileOperation,
)

resize = ResizeOperation.build_step
crop = CropOperation.build_step
crop_to_label = CropToLabelOperation.build_step
flip = FlipOperation.build_step
pad = PadOperation.build_step
tile = TileOperation.build_step

__all__ = ["resize", "crop", "crop_to_label", "flip", "pad", "tile"]
