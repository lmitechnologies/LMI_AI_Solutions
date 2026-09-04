import copy
import logging
import os
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box

# LMI packages
from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    BoxAnnotation,
    FileAnnotations,
    Mask,
    MaskAnnotation,
    Polygon,
    PolygonAnnotation,
)
from lmi_utils.image_utils.tiler import Tiler, compute_new_edges

logger = logging.getLogger(__name__)


def _as_pair(v) -> List[int]:
    return [int(v), int(v)] if isinstance(v, int) else [int(v[0]), int(v[1])]


def _reject_unsupported(annotations: List[Annotation]) -> None:
    """Match the inference-side rejection in preprocess_utils/ops/tile.py — a tiled dataset you cannot predict on is useless."""
    for annot in annotations:
        if annot.type == AnnotationType.KEYPOINT:
            raise ValueError("tile: tiling does not support keypoints")
        if annot.type == AnnotationType.BOX and annot.value.angle:
            raise ValueError("tile: tiling does not support oriented boxes")


def _too_thin(width: float, height: float, min_label_size: float) -> bool:
    return min(width, height) < min_label_size


def _derive(annot: Annotation, cls, value, suffix: str = ""):
    return cls(
        id=f"{annot.id}{suffix}",
        label_id=annot.label_id,
        value=value,
        link=annot.link,
        confidence=annot.confidence,
        iou=annot.iou,
    )


def _clip_box(annot: Annotation, x1: int, y1: int, tile_w: int, tile_h: int, min_label_size: float) -> List[Annotation]:
    box = annot.value
    nx1, ny1 = max(0.0, box.x_min - x1), max(0.0, box.y_min - y1)
    nx2, ny2 = min(float(tile_w), box.x_max - x1), min(float(tile_h), box.y_max - y1)
    if nx2 <= nx1 or ny2 <= ny1 or _too_thin(nx2 - nx1, ny2 - ny1, min_label_size):
        return []
    return [_derive(annot, BoxAnnotation, Box(x_min=nx1, y_min=ny1, x_max=nx2, y_max=ny2))]


def _clip_polygon(annot: Annotation, x1: int, y1: int, tile_w: int, tile_h: int, min_label_size: float) -> List[Annotation]:
    points = np.asarray(annot.value.points, dtype=float) - [x1, y1]
    if len(points) < 3:
        return []
    poly = ShapelyPolygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)  # self-intersecting outlines make intersection() raise
    clipped = poly.intersection(shapely_box(0, 0, tile_w, tile_h))
    parts = [g for g in getattr(clipped, "geoms", [clipped]) if isinstance(g, ShapelyPolygon) and not g.is_empty]

    out = []
    for part in parts:
        px1, py1, px2, py2 = part.bounds
        if _too_thin(px2 - px1, py2 - py1, min_label_size):
            continue
        # a concave outline crossing a corner can come back in pieces; each piece is its own label
        suffix = f"_p{len(out)}" if len(parts) > 1 else ""
        out.append(_derive(annot, PolygonAnnotation, Polygon(points=[[x, y] for x, y in part.exterior.coords[:-1]]), suffix))
    return out


def _clip_mask(
    annot: Annotation, raster: np.ndarray, x1: int, y1: int, tile_w: int, tile_h: int, min_label_size: float
) -> List[Annotation]:
    sub = raster[y1 : y1 + tile_h, x1 : x1 + tile_w]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0 or _too_thin(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, min_label_size):
        return []
    return [_derive(annot, MaskAnnotation, Mask(mask=sub.copy()))]


def tile_annotated_image(
    image: np.ndarray,
    annotations: List[Annotation],
    tile_size,
    stride,
    min_label_size: float = 0.0,
) -> List[Tuple[int, int, np.ndarray, List[Annotation]]]:
    """Cut one annotated image into a grid of tiles, clipping the annotations into each.

    Uses the same grid as inference (``Tiler``, padding scale mode): the image is zero-padded on the
    bottom and right to the next stride multiple, and tile (row, col) starts at (row*stride_h, col*stride_w).

    Keypoints and oriented boxes raise. Annotations are deep-copied, so the input is not mutated.

    Args:
        min_label_size: drop a clipped label thinner than this many pixels on either axis. Slivers only —
            an interior fragment showing none of the object's edges must survive.

    Returns:
        one (row, col, tile_image, tile_annotations) per tile, row-major.
    """
    _reject_unsupported(annotations)
    tile_h, tile_w = _as_pair(tile_size)
    stride_h, stride_w = _as_pair(stride)
    Tiler.validate_tile_and_stride([tile_h, tile_w], [stride_h, stride_w])

    h, w = image.shape[:2]
    scale_h, scale_w = compute_new_edges([h, w], [tile_h, tile_w], [stride_h, stride_w])
    pad = [(0, scale_h - h), (0, scale_w - w)] + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(image, pad)

    # rasterize each mask once at the padded size, then slice per tile
    rasters = {a.id: np.pad(a.value.to_numpy(h=h, w=w), pad[:2]) for a in annotations if a.type == AnnotationType.MASK}

    n_h = (scale_h - tile_h) // stride_h + 1
    n_w = (scale_w - tile_w) // stride_w + 1

    tiles = []
    for row in range(n_h):
        for col in range(n_w):
            y1, x1 = row * stride_h, col * stride_w
            tile_annots: List[Annotation] = []
            for annot in annotations:
                a = copy.deepcopy(annot)
                if a.type == AnnotationType.BOX:
                    tile_annots.extend(_clip_box(a, x1, y1, tile_w, tile_h, min_label_size))
                elif a.type == AnnotationType.POLYGON:
                    tile_annots.extend(_clip_polygon(a, x1, y1, tile_w, tile_h, min_label_size))
                elif a.type == AnnotationType.MASK:
                    tile_annots.extend(_clip_mask(a, rasters[a.id], x1, y1, tile_w, tile_h, min_label_size))
                else:
                    raise ValueError(f"tile: unsupported annotation type {a.type}")
            tiles.append((row, col, padded[y1 : y1 + tile_h, x1 : x1 + tile_w].copy(), tile_annots))
    return tiles


def tile_dataset(dataset, images, tile_size, stride, min_label_size: float = 0.0):
    """Expand every file in the dataset into its tiles, in place.

    One source file becomes N ``FileAnnotations``, each carrying ``source_id`` so tiles of one image can
    be kept on the same side of a train/val split. Tiles with no labels are kept — ``delete_empty_files``
    is the caller's decision (``apply_ops --bg``).

    Returns:
        (tiled_images, dataset) — ``tiled_images`` keyed by the new tile paths, matching ``dataset.files``.
    """
    tiled_images = {}
    tiled_files = []
    for f in dataset.files:
        stem, ext = os.path.splitext(f.path)
        tiles = tile_annotated_image(images[f.path], f.annotations, tile_size, stride, min_label_size)
        for row, col, tile_im, tile_annots in tiles:
            # "id<id>_" prefix is the repo-wide unique-name convention; carrying it here stops
            # save_dataset and json_to_yolo from prepending the tile position a second time
            tile_id = f"{f.id}_r{row}_c{col}"
            path = f"id{tile_id}_{os.path.basename(stem)}{ext}"
            tiled_images[path] = tile_im
            tiled_files.append(
                FileAnnotations(
                    id=tile_id,
                    path=path,
                    height=tile_im.shape[0],
                    width=tile_im.shape[1],
                    annotations=tile_annots,
                    source_id=f.id,
                )
            )
        logger.debug(f"tiled {f.path} ({f.width}x{f.height}) into {len(tiles)} tiles")

    dataset.files = tiled_files
    return tiled_images, dataset
