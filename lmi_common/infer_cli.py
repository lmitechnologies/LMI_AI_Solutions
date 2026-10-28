"""Command-line flags and tiling helpers shared by the detector and classifier ``infer.py`` scripts."""

import argparse
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from lmi_utils.preprocess_utils.ops import TileConfig

TILE_PLOT_DIR = "_tile_plots"  # tiled runs save their tile plots in this subfolder of the output folder


def add_infer_args(
    parser: argparse.ArgumentParser,
    confidence: float,
    weights: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
    tile_plot: str = "the tile grid",
) -> None:
    """Add the flags every infer script takes. A path left as None becomes a required flag.

    ``tile_plot`` describes what the saved tile plot shows, for the ``--stride`` help.
    """

    def path_arg(short, long, default, what):
        help_text = what if default is None else f"{what}, default={default}"
        parser.add_argument(short, long, default=default, required=default is None, help=help_text)

    path_arg("-w", "--weights", weights, "the path to the model weights file")
    path_arg("-i", "--input", input, "the path to the input images")
    path_arg("-o", "--output", output, "the path to the output folder")
    parser.add_argument(
        "-c", "--confidence", type=float, default=confidence, help=f"[optional] the confidence threshold, default={confidence}"
    )
    parser.add_argument(
        "-s",
        "--image_size",
        nargs="+",
        type=int,
        metavar="N",
        help="[optional] the model input size: one int for a square, or two ints: h w. By default it is read from the model",
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
        help="[optional] step between tiles: one int, or two ints: h w. A stride smaller than the tile makes tiles overlap. "
        f"Tiling also saves an image to {TILE_PLOT_DIR}/ in the output folder showing {tile_plot}",
    )


def check_infer_args(parser: argparse.ArgumentParser, args: argparse.Namespace, report_merge_origin: bool = False) -> None:
    """Validate the shared flags, exiting through ``parser.error`` on bad input.

    Sets ``args.image_size`` to [h, w] or None, and ``args.tile_step`` to a tile step or None.
    """
    try:
        size = _size("--image_size", args.image_size)
        args.image_size = None if size is None else _pair(size)
        if args.confidence < 0:
            raise ValueError(f"--confidence must be >= 0, got {args.confidence}")
        args.tile_step = tile_step(args.tile, args.stride, report_merge_origin=report_merge_origin)
    except ValueError as e:
        parser.error(str(e))


def tile_step(tile: Optional[List[int]], stride: Optional[List[int]], report_merge_origin: bool = False) -> Optional["TileConfig"]:
    """Build the tile step from the flag values, or return None when not tiling.

    Raises:
        ValueError: only one of the flags is given, or a flag is not one or two positive ints.
    """
    if tile is None and stride is None:
        return None
    if tile is None or stride is None:
        raise ValueError("--tile and --stride must be given together")
    tile_size, stride_size = _size("--tile", tile), _size("--stride", stride)

    from lmi_utils.preprocess_utils import steps  # imports torch: keep CLI startup fast

    return steps.tile(tile_size=tile_size, stride=stride_size, report_merge_origin=report_merge_origin)


def predict_tiled(model, image, step: "TileConfig", **predict_kwargs) -> Tuple[Dict, Dict, Any]:
    """Tile one image, run ``model.predict`` on the tiles, and merge the results back to image level.

    Returns:
        The results and timing from ``model.predict``, and the (T, 4) xyxy box of each tile in the image.
    """
    from lmi_utils.preprocess_utils.preprocessor import Preprocessor

    tiles, history = Preprocessor().preprocess([image], [step])
    results, time_info = model.predict(tiles, operators=history, **predict_kwargs)
    return results, time_info, history[0].tiler(0).tile_boxes()


def write_tile_plot(output_dir: str, name: str, plot) -> None:
    """Write one RGB tile plot to ``output_dir/TILE_PLOT_DIR/name``."""
    import os

    import cv2

    folder = os.path.join(output_dir, TILE_PLOT_DIR)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name), cv2.cvtColor(plot, cv2.COLOR_RGB2BGR))


def _size(name: str, value: Optional[List[int]]) -> Union[int, List[int], None]:
    if value is None:
        return None
    if len(value) not in (1, 2) or min(value) <= 0:
        raise ValueError(f"{name} takes one or two positive ints (h w), got {value}")
    return value[0] if len(value) == 1 else list(value)


def _pair(size: Union[int, List[int]]) -> List[int]:
    return [size, size] if isinstance(size, int) else size
