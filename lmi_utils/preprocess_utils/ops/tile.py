import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from lmi_utils.image_utils.tiler import Tiler
from lmi_utils.postprocess_utils.nms import binarize_masks, class_aware_nms, filter_instances
from lmi_utils.postprocess_utils.tile_merge import DEFAULT_EDGE_TOLERANCE, instance_boxes, merge_tile_fragments

from .._coords import apply_coord_transform
from ..operation import Config, Meta, Operation

logger = logging.getLogger(__name__)


@dataclass
class TileConfig(Config):
    """Tile each image into fixed-size patches; remember per-image grid for stitching.

    Keypoints and oriented boxes are not supported and raise; only boxes, segments and masks tile.

    tile_size: int or [h, w] — patch size.
    stride: int or [h, w] — step between tile origins. Overlap is ``tile_size - stride``.
    scale_mode: how the image is fit to the tile grid before slicing ("padding" or "interpolation").
    overlap_mode: how overlapping regions are merged on untile ("average", "max", ...).

    The rest configure ``revert_coords``, which rebuilds per-tile predictions in image space:

    merge_fragments: union predictions of one object that adjacent tiles each saw only part of.
        Needs ``scale_mode="padding"`` and more than ``2 * edge_tolerance`` px of overlap on both
        axes. None (default) merges wherever the grid allows it and quietly skips where it does not.
        True demands it and raises when the grid cannot support it. False turns it off, leaving
        seam-split objects split.
    score_threshold: dropped after merging, so a fragment is judged on its group's score.
    nms_iou: class-aware NMS IoU threshold across tiles. None disables both NMS rules.
    containment: fraction of one prediction that must lie inside another to count as contained.
        Used to suppress a fragment nested in a whole detection, and to fold a fragment into the
        overlapping tile's prediction that covers it. None disables the containment rule.

    edge_tolerance: px from a tile edge that still counts as touching it, which is what marks a
        prediction as a fragment. Absolute, not a fraction of the tile: it tracks the detector's
        box-regression error at a crop boundary, which the detection head's feature stride fixes.
        Raising it catches more true fragments and flags more whole objects as cut ones.
    min_label_size: on ``apply_coords``, drop a clipped label thinner than this many pixels on
        either axis. Slivers only — an interior fragment showing none of the object's edges must
        survive, or the model never learns to fire on the middle of an object wider than a tile.
    """

    tile_size: Union[int, List[int], None] = None
    stride: Union[int, List[int], None] = None
    scale_mode: str = "padding"
    overlap_mode: str = "average"
    merge_fragments: Optional[bool] = None
    score_threshold: float = 0.0
    nms_iou: Optional[float] = 0.5
    containment: Optional[float] = 0.8
    edge_tolerance: float = DEFAULT_EDGE_TOLERANCE
    min_label_size: float = 0.0

    def __post_init__(self):
        if self.tile_size is None or self.stride is None:
            raise ValueError("TileConfig: 'tile_size' and 'stride' are required")
        if self.merge_fragments is True:
            if self.scale_mode != "padding":
                raise ValueError("TileConfig: merge_fragments needs scale_mode='padding'; interpolation rescales tile origins")
            _validate_merge_overlap(_as_pair(self.tile_size), _as_pair(self.stride), self.edge_tolerance)

    def coord_options(self) -> Dict[str, Any]:
        return {
            "merge_fragments": self.merge_fragments,
            "score_threshold": self.score_threshold,
            "nms_iou": self.nms_iou,
            "containment": self.containment,
            "edge_tolerance": self.edge_tolerance,
            "min_label_size": self.min_label_size,
        }


def _as_pair(v: Union[int, List[int]]) -> List[int]:
    return [int(v), int(v)] if isinstance(v, int) else [int(v[0]), int(v[1])]


def _validate_merge_overlap(tile_size: List[int], stride: List[int], edge_tolerance: float) -> None:
    """Fragment merging needs real overlap on both axes.

    At or below twice the edge tolerance the facing tile edges are effectively the same line, and
    two same-class objects that merely touch at a seam become indistinguishable from one cut
    object — they pair, merge, and containment NMS then drops both true detections.
    """
    minimum = 2 * edge_tolerance
    overlap = [tile_size[i] - stride[i] for i in (0, 1)]
    if min(overlap) <= minimum:
        raise ValueError(
            f"TileConfig: merge_fragments needs overlap > {minimum} px on both axes "
            f"(2 x edge_tolerance {edge_tolerance}); tile_size {tile_size} and stride {stride} give {overlap}"
        )


@dataclass
class TileMeta(Meta):
    """Batched tile metadata (one entry per *source* image; each source produces n_h*n_w tiles)."""

    tile_sizes: List[List[int]] = field(default_factory=list)
    strides: List[List[int]] = field(default_factory=list)
    im_sizes: List[List[int]] = field(default_factory=list)
    scale_sizes: List[List[int]] = field(default_factory=list)
    n_tiles: List[List[int]] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    num_channels: List[int] = field(default_factory=list)
    scale_modes: List[str] = field(default_factory=list)
    overlap_modes: List[str] = field(default_factory=list)
    coord_options: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.n_tiles)
        if not self.coord_options:  # optional; absent means plain concatenation with no filtering
            self.coord_options = [{} for _ in range(n)]
        for name in (
            "tile_sizes",
            "strides",
            "im_sizes",
            "scale_sizes",
            "batch_sizes",
            "num_channels",
            "scale_modes",
            "overlap_modes",
            "coord_options",
        ):
            if len(getattr(self, name)) != n:
                raise ValueError(f"TileMeta: field '{name}' length mismatch")

    def per_image_dict(self, i: int) -> Dict[str, Any]:
        """Reconstruct a single source image's dict suitable for Tiler.from_dict()."""
        return {
            "tile_size": list(self.tile_sizes[i]),
            "stride": list(self.strides[i]),
            "im_size": list(self.im_sizes[i]),
            "scale_size": list(self.scale_sizes[i]),
            "n_tiles": list(self.n_tiles[i]),
            "batch_size": self.batch_sizes[i],
            "num_channel": self.num_channels[i],
            "scale_mode": self.scale_modes[i],
            "overlap_mode": self.overlap_modes[i],
            **self.coord_options[i],
        }


class TileOperation(Operation[TileConfig, TileMeta]):
    config_cls = TileConfig
    meta_cls = TileMeta
    filters_instances = True  # padding discard, score threshold and NMS can remove every instance

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: TileConfig) -> Tuple[List[torch.Tensor], TileMeta]:
        scale_mode = config.scale_mode
        overlap_mode = config.overlap_mode

        out_images: List[torch.Tensor] = []
        tile_sizes: List[List[int]] = []
        strides: List[List[int]] = []
        im_sizes: List[List[int]] = []
        scale_sizes: List[List[int]] = []
        n_tiles_list: List[List[int]] = []
        batch_sizes: List[int] = []
        num_channels: List[int] = []
        scale_modes: List[str] = []
        overlap_modes: List[str] = []
        coord_options: List[Dict[str, Any]] = []

        for img in images:
            ndim = img.dim()
            add_channel = False
            if ndim == 2:
                add_channel = True
                img = img.unsqueeze(-1)

            tiler = Tiler(tile_size=config.tile_size, stride=config.stride)
            img_batch = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            tiles_batch = tiler.tile(img_batch, mode=scale_mode)  # [N, C, H, W]

            tiles_list_chw = list(torch.unbind(tiles_batch, dim=0))
            tiles_list_hwc = [t.permute(1, 2, 0) for t in tiles_list_chw]
            if add_channel:
                tiles_list_hwc = [t.squeeze(-1) for t in tiles_list_hwc]
            out_images.extend(tiles_list_hwc)

            d = tiler.to_dict()
            tile_sizes.append(list(d["tile_size"]))
            strides.append(list(d["stride"]))
            im_sizes.append(list(d["im_size"]))
            scale_sizes.append(list(d["scale_size"]))
            n_tiles_list.append(list(d["n_tiles"]))
            batch_sizes.append(int(d["batch_size"]))
            num_channels.append(int(d["num_channel"]))
            scale_modes.append(scale_mode)
            overlap_modes.append(overlap_mode)
            coord_options.append(config.coord_options())

        return out_images, TileMeta(
            tile_sizes=tile_sizes,
            strides=strides,
            im_sizes=im_sizes,
            scale_sizes=scale_sizes,
            n_tiles=n_tiles_list,
            batch_sizes=batch_sizes,
            num_channels=num_channels,
            scale_modes=scale_modes,
            overlap_modes=overlap_modes,
            coord_options=coord_options,
        )

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], meta: TileMeta) -> List[torch.Tensor]:
        restored_images = []
        cursor = 0

        for i in range(len(meta.n_tiles)):
            n_h, n_w = meta.n_tiles[i]
            count = n_h * n_w
            tiler = Tiler.from_dict(meta.per_image_dict(i))

            batch_slice_hwc = images[cursor : cursor + count]
            cursor += count
            if len(batch_slice_hwc) != count:
                raise RuntimeError(f"Expected {count} tiles, found {len(batch_slice_hwc)}")

            batch_hwc = torch.stack(batch_slice_hwc)  # [N, H, W, C] or [N, H, W]
            ndim = batch_hwc.dim()
            if ndim not in {4, 3}:
                raise ValueError(f"Tile batch must have 3 or 4 dimensions (N, H, W) or (N, H, W, C). Got {ndim} dimensions.")

            add_channel = False
            if ndim == 3:
                add_channel = True
                batch_hwc = batch_hwc.unsqueeze(-1)
            batch_chw = batch_hwc.permute(0, 3, 1, 2)  # [N, C, H, W]

            restored_batch = tiler.untile(batch_chw, scale_mode=meta.scale_modes[i], overlap_mode=meta.overlap_modes[i])
            restored_img = restored_batch.squeeze(0).permute(1, 2, 0)
            if add_channel:
                restored_img = restored_img.squeeze(-1)
            restored_images.append(restored_img)

        if cursor != len(images):
            raise RuntimeError(f"Tile reconstruction mismatch: processed {cursor} images, but received {len(images)}")
        return restored_images

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], meta: TileMeta) -> List[Dict[str, Any]]:
        """Project original-space coords into per-tile space.

        Expands the per-image results list into a per-tile list (one dict per emitted tile,
        row-major order matching ``forward``). Instances that don't overlap a given tile
        are dropped from that tile; partially overlapping ones are clipped:
            - xyxy boxes: axis-aligned clip to tile rect.
            - segments: Sutherland-Hodgman clip; instances with empty clipped polygon are dropped.
            - masks: per-tile spatial slice.

        Results carrying no geometry (only image-level scores/classes) are passed through to
        every tile unchanged, since there is nothing to clip.

        Mask input convention (original space, matching the box/segment coords):
            - interpolation: masks are in ``im_size``; resampled to ``scale_size`` before tiling.
            - padding: masks are in ``im_size``; zero-padded to ``scale_size`` before tiling.
        """
        for r in results:
            _reject_unsupported(r)
        if len(results) != len(meta.n_tiles):
            raise ValueError(f"tile: results count ({len(results)}) != meta count ({len(meta.n_tiles)})")
        output = []
        for i, r in enumerate(results):
            d = meta.per_image_dict(i)
            n_h, n_w = d["n_tiles"]
            for row in range(n_h):
                for col in range(n_w):
                    output.append(_project_to_tile(r, d, row, col))
        return output

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], meta: TileMeta) -> List[Dict[str, Any]]:
        """Rebuild per-tile predictions in image space. See ``_merge_tile_coords`` for the steps."""
        for r in results:
            _reject_unsupported(r)
        output = []
        cursor = 0
        for i in range(len(meta.n_tiles)):
            n_h, n_w = meta.n_tiles[i]
            count = n_h * n_w
            tile_results = results[cursor : cursor + count]
            cursor += count
            if len(tile_results) != count:
                raise RuntimeError(f"Expected {count} tile results, found {len(tile_results)}")
            output.append(_merge_tile_coords(tile_results, meta.per_image_dict(i)))

        if cursor != len(results):
            raise RuntimeError(f"Tile coord reconstruction mismatch: processed {cursor}, received {len(results)}")
        return output


def _merge_tile_coords(tile_results: List[Dict[str, Any]], tiler_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Offset per-tile predictions into image space and rebuild whole objects from them.

    Offset -> drop what landed in the padded region -> merge seam fragments -> score threshold ->
    class-aware NMS -> clip to the image. Thresholding runs after merging so a fragment is judged
    on its group's restored score, and NMS runs last so it cannot delete fragments their partners
    still need.
    """
    n_tiles_h, n_tiles_w = tiler_meta["n_tiles"]
    tile_h, tile_w = tiler_meta["tile_size"]
    stride_h, stride_w = tiler_meta["stride"]
    im_h, im_w = tiler_meta["im_size"]
    scale_h, scale_w = tiler_meta["scale_size"]
    is_interp = tiler_meta.get("scale_mode", "padding") == "interpolation" and (scale_h != im_h or scale_w != im_w)
    sx = im_w / scale_w if is_interp else 1.0
    sy = im_h / scale_h if is_interp else 1.0

    # paste straight at image size: a padded canvas sliced afterwards stays a non-contiguous view that
    # pins the bigger tensor and forces a fresh contiguous copy on every downstream flatten
    target_size = (im_h, im_w)

    counts = [_instance_count(r) for r in tile_results]
    mask_buf = _allocate_mask_buffer(tile_results, sum(counts), target_size)

    shifted = []
    tile_idx_parts = []
    at = 0
    for idx, r in enumerate(tile_results):
        row = idx // n_tiles_w
        col = idx % n_tiles_w
        out = None if mask_buf is None else mask_buf[at : at + counts[idx]]
        s = _shift_tile_coords(r, col * stride_w, row * stride_h, sx, sy, target_size, mask_out=out)
        at += counts[idx]
        shifted.append(s)
        tile_idx_parts.append(torch.full((_instance_count(s),), idx, dtype=torch.long))

    merged = _concat_tile_results(shifted, masks=mask_buf)
    tile_idx = torch.cat(tile_idx_parts) if tile_idx_parts else torch.zeros(0, dtype=torch.long)

    if not is_interp and (scale_h != im_h or scale_w != im_w):
        merged, tile_idx = _drop_in_padding(merged, tile_idx, im_h, im_w)

    requested = tiler_meta.get("merge_fragments", False)  # absent means a hand-built meta: off
    did_merge = False
    if requested is not False:
        tolerance = tiler_meta.get("edge_tolerance")
        tolerance = DEFAULT_EDGE_TOLERANCE if tolerance is None else float(tolerance)
        skip = None if requested is True else _auto_merge_skip_reason((tile_h, tile_w), (stride_h, stride_w), is_interp, tolerance)
        if skip is not None:
            logger.debug("tile: skipping fragment merging - %s", skip)
        else:
            did_merge = True
            rc = np.array([[i // n_tiles_w, i % n_tiles_w] for i in range(n_tiles_h * n_tiles_w)])
            origins = rc * np.array([stride_h, stride_w])
            merged = merge_tile_fragments(
                merged,
                tile_idx,
                origins,
                (tile_h, tile_w),
                (im_h, im_w),
                containment=tiler_meta.get("containment") or 1.0,
                edge_tolerance=tolerance,
            )

    merged = _apply_score_threshold(merged, float(tiler_meta.get("score_threshold") or 0.0))

    # containment suppression runs once: in the merge if it ran, here if it did not. Merging widens a box
    # to its group's union, so re-testing containment would delete the neighbours that union now encloses.
    iou_thr = tiler_meta.get("nms_iou")
    containment = None if did_merge else tiler_meta.get("containment")
    if iou_thr is not None or containment is not None:
        merged = class_aware_nms(merged, iou_thr, containment)
    return _clip_to_image(merged, im_h, im_w)


def _auto_merge_skip_reason(
    tile_size: Tuple[int, int],
    stride: Tuple[int, int],
    is_interp: bool,
    tolerance: float,
) -> Optional[str]:
    """Why a grid cannot support automatic merging; None when it can.

    Only grid geometry — an unsupported model type raises instead, see ``_reject_unsupported``.
    """
    if is_interp:
        return "scale_mode='interpolation' rescales tile origins"
    overlap = [tile_size[i] - stride[i] for i in (0, 1)]
    if min(overlap) <= 2 * tolerance:
        # Below this the facing tile edges are one line, so two objects touching at a seam are
        # indistinguishable from one cut object and would be fused.
        return f"overlap {overlap} is not more than 2 x edge_tolerance ({2 * tolerance})"
    return None


def _reject_unsupported(result: Dict[str, Any]) -> None:
    """Tiling has no rule for keypoints or oriented boxes; fail rather than return them wrong.

    The clip/refit code for both is still in ``_project_to_tile``, kept for future support.
    """
    points = result.get("points")
    if isinstance(points, torch.Tensor) and len(points):
        raise ValueError("tile: tiling does not support keypoints")
    boxes = result.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes) and boxes.ndim == 3:
        raise ValueError("tile: tiling does not support oriented boxes")


def _drop_in_padding(merged: Dict[str, Any], tile_idx: torch.Tensor, im_h: int, im_w: int) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Discard predictions that lie wholly outside the original image, in the padding."""
    boxes = instance_boxes(merged)
    if boxes is None or len(boxes) != len(tile_idx):
        return merged, tile_idx
    keep = ((boxes[:, 0] < im_w) & (boxes[:, 1] < im_h)).nonzero(as_tuple=True)[0]
    if len(keep) == len(tile_idx):
        return merged, tile_idx
    return filter_instances(merged, keep), tile_idx[keep]


def _apply_score_threshold(merged: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    scores = merged.get("scores")
    if threshold <= 0 or not isinstance(scores, torch.Tensor) or not len(scores):
        return merged
    keep = (scores.detach().cpu().float() >= threshold).nonzero(as_tuple=True)[0]
    if len(keep) == len(scores):
        return merged
    return filter_instances(merged, keep)


def _clip_to_image(merged: Dict[str, Any], im_h: int, im_w: int) -> Dict[str, Any]:
    """Clip geometry to the original image bounds. Masks are already cropped to it."""
    out = dict(merged)
    boxes = merged.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes):
        clipped = boxes.clone().float()
        clipped[..., 0::2] = clipped[..., 0::2].clamp(0, im_w)
        clipped[..., 1::2] = clipped[..., 1::2].clamp(0, im_h)
        out["boxes"] = clipped
    segments = merged.get("segments")
    if segments is not None and len(segments):
        out["segments"] = [_polygon_clip(s, im_w, im_h) if len(s) else s for s in segments]
    return out


def _shift_tile_coords(
    result: Dict[str, Any],
    offset_x: int,
    offset_y: int,
    sx: float,
    sy: float,
    target_size: Tuple[int, int],
    mask_out: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Offset one tile's predictions into image space.

    ``mask_out`` is this tile's zeroed slice of the batch mask buffer; passing it keeps the caller
    from holding every tile's canvas alive while ``torch.cat`` allocates the combined one.
    """
    scaled = sx != 1.0 or sy != 1.0

    def xy_fn(xy: torch.Tensor) -> torch.Tensor:
        off = torch.tensor([offset_x, offset_y], dtype=torch.float32, device=xy.device)
        shifted = xy.float() + off
        if scaled:
            scale = torch.tensor([sx, sy], dtype=torch.float32, device=xy.device)
            shifted = shifted * scale
        return shifted

    def mask_fn(masks: torch.Tensor) -> torch.Tensor:
        canvas_h, canvas_w = target_size
        if scaled:
            new_h = max(1, round(masks.shape[1] * sy))
            new_w = max(1, round(masks.shape[2] * sx))
            masks = (
                torch.nn.functional.interpolate(
                    masks.float().unsqueeze(1), size=(new_h, new_w), mode="bilinear", align_corners=False
                ).squeeze(1)
                > 0.5
            )
            paste_y = round(offset_y * sy)
            paste_x = round(offset_x * sx)
        else:
            masks = binarize_masks(masks)
            paste_y, paste_x = offset_y, offset_x
        # bool, not float: a full-image float32 canvas per tile is gigabytes on a dense scene.
        # od_base restores the caller's dtype once the whole batch is reverted.
        canvas = mask_out if mask_out is not None else torch.zeros(len(masks), canvas_h, canvas_w, dtype=torch.bool, device=masks.device)
        h_end = min(paste_y + masks.shape[1], canvas_h)
        w_end = min(paste_x + masks.shape[2], canvas_w)
        canvas[:, paste_y:h_end, paste_x:w_end] = masks[:, : h_end - paste_y, : w_end - paste_x]
        return canvas

    return apply_coord_transform(result, xy_fn=xy_fn, mask_fn=mask_fn)


def _allocate_mask_buffer(tile_results: List[Dict[str, Any]], total: int, target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    """One zeroed (total, H, W) bool buffer for the batch, or None when no tile carries masks."""
    sample = next((r["masks"] for r in tile_results if isinstance(r.get("masks"), torch.Tensor) and len(r["masks"])), None)
    if sample is None or not total:
        return None
    return torch.zeros(total, target_size[0], target_size[1], dtype=torch.bool, device=sample.device)


def _concat_tile_results(results: List[Dict[str, Any]], masks: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    if not results:
        return {}

    merged = {}
    all_keys = {k for r in results for k in r}
    if masks is not None:
        all_keys.discard("masks")  # already written in place, concatenating would double the peak

    for key in all_keys:
        vals = [r[key] for r in results if key in r and r[key] is not None]
        if not vals:
            continue
        if key == "segments":
            merged[key] = [seg for segs in vals for seg in segs]
        elif key == "classes":
            non_empty = [v for v in vals if len(v) > 0]
            merged[key] = np.concatenate(non_empty) if non_empty else vals[0]
        else:  # boxes, scores, points, masks
            non_empty = [v for v in vals if len(v) > 0]
            merged[key] = torch.cat(non_empty) if non_empty else vals[0]

    if masks is not None:
        merged["masks"] = masks
    return merged


def _project_to_tile(result: Dict[str, Any], m: Dict[str, Any], row: int, col: int) -> Dict[str, Any]:
    import torch.nn.functional as F

    tile_h, tile_w = m["tile_size"]
    stride_h, stride_w = m["stride"]
    im_h, im_w = m["im_size"]
    scale_h, scale_w = m["scale_size"]
    is_interp = m.get("scale_mode", "padding") == "interpolation" and (scale_h != im_h or scale_w != im_w)
    sx = scale_w / im_w if is_interp else 1.0
    sy = scale_h / im_h if is_interp else 1.0
    off_x = col * stride_w
    off_y = row * stride_h

    def to_tile_xy(xy: torch.Tensor) -> torch.Tensor:
        xy = xy.float()
        if is_interp:
            xy = xy * torch.tensor([sx, sy], dtype=torch.float32, device=xy.device)
        return xy - torch.tensor([off_x, off_y], dtype=torch.float32, device=xy.device)

    n = _instance_count(result)
    out: Dict[str, Any] = dict(result)
    if n == 0:
        return out

    device = _result_device(result)
    kept = torch.zeros(n, dtype=torch.bool, device=device)
    boxes_out = points_out = masks_out = None
    segments_out: Optional[List[torch.Tensor]] = None
    has_spatial = False  # whether any geometry field exists to clip against

    boxes = result.get("boxes")
    if boxes is not None and len(boxes):
        has_spatial = True
        if boxes.ndim == 3:  # OBB (N, 4, 2)
            # Unreachable: _reject_unsupported turns OBB away. Kept for future support.
            obb_t = to_tile_xy(boxes.reshape(-1, 2)).reshape(n, 4, 2)
            obb_kept = torch.zeros(n, dtype=torch.bool, device=device)
            refit = obb_t.clone()
            for i in range(n):
                clipped = _polygon_clip(obb_t[i], tile_w, tile_h)
                if clipped.shape[0] < 3:
                    continue
                fitted = _refit_obb_keep_orientation(clipped, obb_t[i])
                if fitted is None:
                    continue
                refit[i] = fitted.to(refit.dtype).to(refit.device)
                obb_kept[i] = True
            boxes_out = refit
            kept |= obb_kept
        else:  # xyxy (N, 4) — rotate-style: take aabb of forward-projected corners, then clip
            corners = torch.stack(
                [
                    torch.stack([boxes[:, 0], boxes[:, 1]], dim=-1),
                    torch.stack([boxes[:, 2], boxes[:, 1]], dim=-1),
                    torch.stack([boxes[:, 2], boxes[:, 3]], dim=-1),
                    torch.stack([boxes[:, 0], boxes[:, 3]], dim=-1),
                ],
                dim=1,
            )  # (N, 4, 2)
            corners_t = to_tile_xy(corners.reshape(-1, 2)).reshape(n, 4, 2)
            x1 = corners_t[..., 0].amin(dim=1).clamp(0, tile_w)
            x2 = corners_t[..., 0].amax(dim=1).clamp(0, tile_w)
            y1 = corners_t[..., 1].amin(dim=1).clamp(0, tile_h)
            y2 = corners_t[..., 1].amax(dim=1).clamp(0, tile_h)
            boxes_out = torch.stack([x1, y1, x2, y2], dim=-1)
            kept |= (x2 > x1) & (y2 > y1)

    points = result.get("points")
    if points is not None and len(points):
        # Unreachable: _reject_unsupported turns keypoints away. Kept for future support.
        has_spatial = True
        xy = points[..., :2]
        xy_t = to_tile_xy(xy.reshape(-1, 2)).reshape(xy.shape)
        in_tile = (xy_t[..., 0] >= 0) & (xy_t[..., 0] <= tile_w) & (xy_t[..., 1] >= 0) & (xy_t[..., 1] <= tile_h)
        if points.shape[-1] == 3:
            vis = points[..., 2]
            new_vis = torch.where(in_tile, vis, torch.zeros_like(vis))
            points_out = torch.cat([xy_t, new_vis.unsqueeze(-1)], dim=-1)
            kept |= (new_vis > 0).any(dim=-1)
        else:
            points_out = xy_t
            kept |= in_tile.any(dim=-1)

    segments = result.get("segments")
    if segments is not None and len(segments):
        has_spatial = True
        segments_out = []
        for i, s in enumerate(segments):
            if len(s) == 0:
                segments_out.append(s)
                continue
            clipped = _polygon_clip(to_tile_xy(s), tile_w, tile_h)
            segments_out.append(clipped)
            if clipped.shape[0] >= 3:
                kept[i] = True

    masks = result.get("masks")
    if masks is not None and len(masks):
        has_spatial = True
        if is_interp:
            full = F.interpolate(masks.float().unsqueeze(1), size=(scale_h, scale_w), mode="nearest").squeeze(1)
        else:
            mh, mw = masks.shape[1], masks.shape[2]
            if (mh, mw) != (scale_h, scale_w):
                full = F.pad(masks, [0, max(0, scale_w - mw), 0, max(0, scale_h - mh)])
            else:
                full = masks
        y0, x0 = off_y, off_x
        y1, x1 = min(y0 + tile_h, scale_h), min(x0 + tile_w, scale_w)
        sliced = full[:, max(0, y0) : y1, max(0, x0) : x1]
        pad_b = tile_h - sliced.shape[1]
        pad_r = tile_w - sliced.shape[2]
        if pad_b or pad_r:
            sliced = F.pad(sliced, [0, max(0, pad_r), 0, max(0, pad_b)])
        masks_out = sliced.to(masks.dtype)
        kept |= (masks_out.reshape(n, -1) != 0).any(dim=-1)

    if not has_spatial:
        # Image-level label (only scores/classes, no geometry): nothing to clip, so
        # propagate every instance to this tile rather than dropping them all.
        kept[:] = True

    kept_idx = kept.nonzero(as_tuple=True)[0]
    if boxes_out is not None:
        out["boxes"] = boxes_out[kept_idx]
    if points_out is not None:
        out["points"] = points_out[kept_idx]
    if segments_out is not None:
        out["segments"] = [segments_out[i] for i in kept_idx.tolist()]
    if masks_out is not None:
        out["masks"] = masks_out[kept_idx]

    scores = result.get("scores")
    if scores is not None and len(scores) == n:
        out["scores"] = scores[kept_idx]
    classes = result.get("classes")
    if classes is not None and len(classes) == n:
        if isinstance(classes, np.ndarray):
            out["classes"] = classes[kept_idx.cpu().numpy()]
        else:
            out["classes"] = classes[kept_idx]

    return _drop_slivers(out, float(m.get("min_label_size") or 0.0))


def _drop_slivers(result: Dict[str, Any], min_size: float) -> Dict[str, Any]:
    """Drop clipped labels thinner than ``min_size`` px on either axis."""
    if min_size <= 0:
        return result
    boxes = instance_boxes(result)
    if boxes is None or not len(boxes):
        return result
    extent = boxes[:, 2:] - boxes[:, :2]
    keep = (extent.amin(dim=1) >= min_size).nonzero(as_tuple=True)[0]
    if len(keep) == len(boxes):
        return result
    return filter_instances(result, keep)


def _result_device(result: Dict[str, Any]) -> torch.device:
    """Device of the result's coord tensors (defaults to CPU when none are present)."""
    for key in ("boxes", "points", "masks", "scores"):
        v = result.get(key)
        if isinstance(v, torch.Tensor) and len(v):
            return v.device
    segments = result.get("segments")
    if segments is not None:
        for s in segments:
            if isinstance(s, torch.Tensor) and len(s):
                return s.device
    return torch.device("cpu")


def _instance_count(result: Dict[str, Any]) -> int:
    """Per-instance N = max len across populated coord fields (empty placeholders ignored)."""
    n = 0
    for key in ("boxes", "points", "masks", "scores", "classes", "segments"):
        v = result.get(key)
        if v is not None and len(v) > n:
            n = len(v)
    return n


def _polygon_clip(poly: torch.Tensor, w: float, h: float) -> torch.Tensor:
    """Sutherland-Hodgman clip of a polygon (M, 2) against rect [0, w] x [0, h].

    Returns (M', 2) clipped vertices; M' may be 0 if polygon is fully outside.
    """
    pts = poly.detach().cpu().numpy().tolist()
    # Each edge: keep points where a*x + b*y >= c.
    edges = [
        (1.0, 0.0, 0.0),  # x >= 0
        (-1.0, 0.0, -float(w)),  # x <= w
        (0.0, 1.0, 0.0),  # y >= 0
        (0.0, -1.0, -float(h)),  # y <= h
    ]
    for a, b, c in edges:
        if not pts:
            break
        out: List[List[float]] = []
        for i in range(len(pts)):
            curr = pts[i]
            prev = pts[i - 1]
            curr_in = a * curr[0] + b * curr[1] >= c
            prev_in = a * prev[0] + b * prev[1] >= c
            if curr_in:
                if not prev_in:
                    denom = a * (curr[0] - prev[0]) + b * (curr[1] - prev[1])
                    t = (c - (a * prev[0] + b * prev[1])) / denom if denom != 0 else 0.0
                    out.append([prev[0] + t * (curr[0] - prev[0]), prev[1] + t * (curr[1] - prev[1])])
                out.append(curr)
            elif prev_in:
                denom = a * (curr[0] - prev[0]) + b * (curr[1] - prev[1])
                t = (c - (a * prev[0] + b * prev[1])) / denom if denom != 0 else 0.0
                out.append([prev[0] + t * (curr[0] - prev[0]), prev[1] + t * (curr[1] - prev[1])])
        pts = out
    if not pts:
        return torch.zeros((0, 2), dtype=poly.dtype, device=poly.device)
    return torch.tensor(pts, dtype=poly.dtype, device=poly.device)


def _refit_obb_keep_orientation(clipped: torch.Tensor, original: torch.Tensor) -> Optional[torch.Tensor]:
    """Refit a clipped polygon as a 4-corner rect aligned with the original OBB's orientation.

    Unreachable while tiling rejects OBB; kept for future support.

    Project clipped points into the original OBB's local frame, take the AABB there,
    then rotate back. Result keeps the original rotation (so a near-vertical OBB stays
    near-vertical even after clipping a sliver). Output corners are in TL→TR→BR→BL
    order in the original OBB's local frame.

    Returns None for degenerate inputs (< 3 vertices, zero-length edge, or zero-area AABB).
    """
    if clipped.shape[0] < 3:
        return None
    pts = clipped.detach().cpu().numpy().astype(np.float32)
    orig = original.detach().cpu().numpy().astype(np.float32)

    dx = orig[1, 0] - orig[0, 0]
    dy = orig[1, 1] - orig[0, 1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    theta = float(np.arctan2(dy, dx))
    c, s = float(np.cos(theta)), float(np.sin(theta))

    R_to_local = np.array([[c, -s], [s, c]], dtype=np.float32)
    local = pts @ R_to_local
    x_min, y_min = float(local[:, 0].min()), float(local[:, 1].min())
    x_max, y_max = float(local[:, 0].max()), float(local[:, 1].max())
    if (x_max - x_min) <= 1e-6 or (y_max - y_min) <= 1e-6:
        return None

    local_corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
    R_to_world = np.array([[c, s], [-s, c]], dtype=np.float32)
    world = local_corners @ R_to_world
    return torch.from_numpy(world)
