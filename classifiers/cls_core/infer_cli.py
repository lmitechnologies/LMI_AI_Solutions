"""Command-line flags and tiling helpers shared by classifier ``infer.py`` scripts."""

import argparse
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from lmi_utils.preprocess_utils.ops import TileConfig

TILE_PLOT_DIR = "tiles"


def add_infer_args(
    parser: argparse.ArgumentParser,
    weights: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
) -> None:
    """Add the flags every classifier script takes. A path left as None becomes a required flag."""

    def path_arg(short, long, default, what):
        help_text = what if default is None else f"{what}, default={default}"
        parser.add_argument(short, long, default=default, required=default is None, help=help_text)

    path_arg("-w", "--weights", weights, "path to the model weights file")
    path_arg("-i", "--input", input, "path to the input images")
    path_arg("-o", "--output", output, "path to the output folder")
    parser.add_argument(
        "-s",
        "--image_size",
        nargs="+",
        type=int,
        metavar="N",
        help="[optional] model input size: one int for a square, or two ints: h w. By default read from the model",
    )
    parser.add_argument(
        "--tile",
        nargs="+",
        type=int,
        metavar="N",
        help="[optional] tile each image before inference: one int for a square tile, or two ints: h w. Requires --stride",
    )
    parser.add_argument(
        "--stride",
        nargs="+",
        type=int,
        metavar="N",
        help=f"[optional] step between tiles: one int, or two ints: h w. Overlap = tile - stride. "
        f"Saves a tile-grid debug image per image to {TILE_PLOT_DIR}/ in the output folder.",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0,
        help="[optional] confidence threshold in [0, 1]. Predictions below it are reported as no detection. Default 0 = no filtering",
    )


def check_infer_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate shared flags. Sets ``args.image_size`` to [h, w] or None and ``args.tile_step`` to a TileConfig or None."""
    try:
        size = _size("--image_size", args.image_size)
        args.image_size = None if size is None else _pair(size)
        args.tile_step = tile_step(args.tile, args.stride)
        if args.confidence < 0:
            raise ValueError(f"--confidence must be >= 0, got {args.confidence}")
    except ValueError as e:
        parser.error(str(e))


def tile_step(tile: Optional[list], stride: Optional[list]) -> Optional["TileConfig"]:
    """Build a TileConfig from flag values, or return None when not tiling."""
    if tile is None and stride is None:
        return None
    if tile is None or stride is None:
        raise ValueError("--tile and --stride must be given together")
    tile_size = _size("--tile", tile)
    stride_size = _size("--stride", stride)

    from lmi_utils.preprocess_utils import steps  # late import: keeps CLI startup fast

    # classifiers have no spatial outputs to merge, so merge_origin tracking is unnecessary
    return steps.tile(tile_size=tile_size, stride=stride_size)


def predict_tiled(model, image, step: "TileConfig", **predict_kwargs) -> Tuple[Dict, Dict, Any]:
    """Tile one image, run ``model.predict`` on the tiles, and aggregate predictions to image level.

    Returns:
        (results, time_info, tile_boxes) — results and timing from ``model.predict`` after tile
        aggregation, and the (T, 4) xyxy box of each tile in source-image coordinates.
    """
    from lmi_utils.preprocess_utils.preprocessor import Preprocessor

    tiles, history = Preprocessor().preprocess([image], [step])
    results, time_info = model.predict(tiles, operators=history, **predict_kwargs)
    return results, time_info, history[0].tiler(0).tile_boxes()


def save_tile_plot(output_dir: str, name: str, image, tile_boxes) -> None:
    """Save a debug image showing the tile grid overlaid on the source image."""
    import os

    import cv2
    import numpy as np

    from lmi_utils.label_utils.plot_utils import plot_tile_grid

    im = np.ascontiguousarray(image).copy()
    plot_tile_grid(tile_boxes, im)

    folder = os.path.join(output_dir, TILE_PLOT_DIR)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


# ── internal helpers ──────────────────────────────────────────────────────────


def _size(flag: str, values: Optional[list]) -> Optional[Any]:
    if values is None:
        return None
    if len(values) not in (1, 2) or min(values) <= 0:
        raise ValueError(f"{flag} takes one or two positive ints (h w), got {values}")
    return values[0] if len(values) == 1 else values


def _pair(v) -> list:
    return [v, v] if isinstance(v, int) else list(v)
