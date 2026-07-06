from .cropbox import CropBoxConfig, CropBoxMeta, CropBoxOperation
from .flip import FlipConfig, FlipMeta, FlipOperation
from .pad import PadConfig, PadMeta, PadOperation
from .resize import ResizeConfig, ResizeMeta, ResizeOperation
from .tile import TileConfig, TileMeta, TileOperation

# Operations registered by default on both Preprocessor and Reconstructor.
DEFAULT_OPERATIONS = (
    ResizeOperation,
    PadOperation,
    FlipOperation,
    TileOperation,
    CropBoxOperation,
)

__all__ = [
    "DEFAULT_OPERATIONS",
    "CropBoxConfig",
    "CropBoxMeta",
    "CropBoxOperation",
    "FlipConfig",
    "FlipMeta",
    "FlipOperation",
    "PadConfig",
    "PadMeta",
    "PadOperation",
    "ResizeConfig",
    "ResizeMeta",
    "ResizeOperation",
    "TileConfig",
    "TileMeta",
    "TileOperation",
]
