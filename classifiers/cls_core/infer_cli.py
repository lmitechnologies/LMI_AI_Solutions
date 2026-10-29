"""Command-line flags and tiling helpers shared by classifier ``infer.py`` scripts."""

from lmi_common.infer_cli import TILE_PLOT_DIR, add_infer_args, check_infer_args, predict_tiled, write_tile_plot

__all__ = ["TILE_PLOT_DIR", "add_infer_args", "check_infer_args", "predict_tiled", "save_tile_plot"]


def save_tile_plot(output_dir: str, name: str, image, tile_boxes) -> None:
    """Write a copy of one RGB image with the tile grid drawn on it to ``output_dir/TILE_PLOT_DIR/name``."""
    import numpy as np

    from lmi_utils.label_utils.plot_utils import plot_tile_grid

    im = np.ascontiguousarray(image).copy()
    plot_tile_grid(tile_boxes, im, line_thickness=2)
    write_tile_plot(output_dir, name, im)
