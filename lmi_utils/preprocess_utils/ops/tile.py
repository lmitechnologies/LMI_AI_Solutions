import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from lmi_utils.image_utils.tiler import Tiler
from lmi_utils.postprocess_utils.mask_crops import MaskCrops
from lmi_utils.postprocess_utils.nms import class_aware_nms, filter_instances, result_device
from lmi_utils.postprocess_utils.tile_merge import DEFAULT_EDGE_TOLERANCE, instance_boxes, merge_tile_fragments

from .._coords import apply_coord_transform
from ..operation import Config, Meta, Operation

logger = logging.getLogger(__name__)

_MASK_RESIZE_BUDGET = 64 * 1024 * 1024  # bytes one mask resize may hold; a single mask bigger than this cannot be split


@dataclass
class TileConfig(Config):
    """Tile each image into fixed-size patches; remember per-image grid for stitching.

    Keypoints and oriented boxes are not supported and raise; only boxes, segments and masks tile.

    tile_size: int or [h, w] — patch size.
    stride: int or [h, w] — step between tile origins, never more than ``tile_size``. Overlap is ``tile_size - stride``.
    scale_mode: how the image is fit to the tile grid before slicing ("padding" or "interpolation").
    overlap_mode: how overlapping regions are merged on untile ("average", "max", ...).

    The rest configure ``revert_coords``, which rebuilds per-tile predictions in image space:

    merge_fragments: union predictions of one object that adjacent tiles each saw only part of. On by
        default; False leaves seam-split objects split. Works without tile overlap, though overlap scores
        better: two pieces that only meet at a seam can be compared along it but not across it.
    score_threshold: predictions below this are dropped before merging, so a low-scoring piece cannot
        become its group's representative and take the whole group down with it.
    nms_iou: class-aware NMS IoU threshold across tiles. None disables both NMS rules.
    containment: fraction of one prediction that must lie inside another to count as contained.
        Used to suppress a fragment nested in a whole detection, and to fold a fragment into the
        overlapping tile's prediction that covers it. None disables the containment rule.

    edge_tolerance: px from a tile edge that still counts as touching it. Absolute, not a fraction of
        the tile: it tracks the detector's box-regression error at a crop boundary, which the detection
        head's feature stride fixes. It decides which leftover fragments may be dropped; joining reaches
        further, to ``tile_merge.JOIN_MARGIN``, and with no tile overlap it also sets how wide a band
        either side of a seam two pieces are compared in. Results change little between 0.5 and 4.
    min_label_size: on ``apply_coords``, drop a clipped label thinner than this many pixels on
        either axis. Slivers only — an interior fragment showing none of the object's edges must
        survive, or the model never learns to fire on the middle of an object wider than a tile.
    report_merge_origin: add a ``merge_origin`` code per prediction, saying whether a tile saw it whole, it
        absorbed fragments, or it is a union of fragments (``tile_merge.ORIGIN_*``). For inspection; off by
        default so the result keys do not change.
    """

    tile_size: Union[int, List[int], None] = None
    stride: Union[int, List[int], None] = None
    scale_mode: str = "padding"
    overlap_mode: str = "average"
    merge_fragments: bool = True
    score_threshold: float = 0.0
    nms_iou: Optional[float] = 0.5
    containment: Optional[float] = 0.8
    edge_tolerance: float = DEFAULT_EDGE_TOLERANCE
    min_label_size: float = 0.0
    report_merge_origin: bool = False

    def __post_init__(self):
        if self.tile_size is None or self.stride is None:
            raise ValueError("TileConfig: 'tile_size' and 'stride' are required")
        try:  # the same rule ``Tiler`` enforces, raised here so a bad config fails before the first image
            Tiler.validate_tile_and_stride(_as_pair(self.tile_size), _as_pair(self.stride))
        except ValueError as e:
            raise ValueError(f"TileConfig: {e}; got tile_size {self.tile_size}, stride {self.stride}") from e

    def coord_options(self) -> Dict[str, Any]:
        return {
            "merge_fragments": self.merge_fragments,
            "score_threshold": self.score_threshold,
            "nms_iou": self.nms_iou,
            "containment": self.containment,
            "edge_tolerance": self.edge_tolerance,
            "min_label_size": self.min_label_size,
            "report_merge_origin": self.report_merge_origin,
        }


def _as_pair(v: Union[int, List[int]]) -> List[int]:
    return [int(v), int(v)] if isinstance(v, int) else [int(v[0]), int(v[1])]


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

    def tiler(self, i: int) -> Tiler:
        """Rebuild a single source image's Tiler."""
        return Tiler.from_dict(self.per_image_dict(i))


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
            tiler = meta.tiler(i)

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

    Offset -> drop what landed in the padded region -> score threshold -> merge seam fragments ->
    class-aware NMS -> clip to the image. Thresholding runs before merging so a weak view cannot
    represent its group, and NMS runs last so it cannot delete fragments their partners still need.

    Under interpolation every step runs in the scaled image's coordinates, where the tile grid is exact and
    ``edge_tolerance`` still measures the detector's error; the survivors are rescaled to image space at the end.
    """
    n_tiles_h, n_tiles_w = tiler_meta["n_tiles"]
    tile_h, tile_w = tiler_meta["tile_size"]
    stride_h, stride_w = tiler_meta["stride"]
    im_h, im_w = tiler_meta["im_size"]
    scale_h, scale_w = tiler_meta["scale_size"]
    is_interp = tiler_meta.get("scale_mode", "padding") == "interpolation" and (scale_h != im_h or scale_w != im_w)
    work_h, work_w = (scale_h, scale_w) if is_interp else (im_h, im_w)

    # masks stay as box crops until the survivors are pasted: a full-image mask per detection runs out of memory on large images
    target_size = (work_h, work_w)
    mask_dtype = _output_mask_dtype(tile_results)

    shifted = []
    tile_idx_parts = []
    for idx, r in enumerate(tile_results):
        row = idx // n_tiles_w
        col = idx % n_tiles_w
        s = _shift_tile_coords(r, col * stride_w, row * stride_h, target_size, mask_dtype)
        shifted.append(s)
        tile_idx_parts.append(torch.full((_instance_count(s),), idx, dtype=torch.long))

    merged = _concat_tile_results(shifted)
    tile_idx = torch.cat(tile_idx_parts) if tile_idx_parts else torch.zeros(0, dtype=torch.long)

    if not is_interp and (scale_h != im_h or scale_w != im_w):
        merged, tile_idx = _drop_in_padding(merged, tile_idx, im_h, im_w)

    merged, tile_idx = _apply_score_threshold(merged, float(tiler_meta.get("score_threshold") or 0.0), tile_idx)

    report_origin = bool(tiler_meta.get("report_merge_origin"))
    did_merge = bool(tiler_meta.get("merge_fragments", False))  # absent means a hand-built meta: off
    if did_merge:
        tolerance = tiler_meta.get("edge_tolerance")
        tolerance = DEFAULT_EDGE_TOLERANCE if tolerance is None else float(tolerance)
        rc = np.array([[i // n_tiles_w, i % n_tiles_w] for i in range(n_tiles_h * n_tiles_w)])
        origins = rc * np.array([stride_h, stride_w])
        merged = merge_tile_fragments(
            merged,
            tile_idx,
            origins,
            (tile_h, tile_w),
            (work_h, work_w),
            containment=_containment_or_strict(tiler_meta.get("containment")),
            edge_tolerance=tolerance,
            report_origin=report_origin,
        )

    # merging replaces containment suppression: it widens a box to its group's union, so a containment test
    # would delete the neighbours that union now encloses.
    iou_thr = tiler_meta.get("nms_iou")
    containment = None if did_merge else tiler_meta.get("containment")
    if iou_thr is not None or containment is not None:
        merged = class_aware_nms(merged, iou_thr, containment)
    masks = merged.get("masks")
    if isinstance(masks, MaskCrops):
        merged["masks"] = masks.paste()
    elif isinstance(masks, torch.Tensor) and not len(masks):  # every tile was empty, so no crops were ever built
        merged["masks"] = masks.new_zeros((0, im_h, im_w))
    if report_origin and "merge_origin" not in merged:
        merged["merge_origin"] = torch.zeros(_instance_count(merged), dtype=torch.uint8)  # no merging ran: every row stands alone
    if is_interp:
        merged = _rescale_to_image(merged, im_w / work_w, im_h / work_h, im_h, im_w, mask_dtype)
    return _clip_to_image(merged, im_h, im_w)


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
    # a mask keeps only its in-image pixels, so an empty box means the prediction sat entirely in the padding
    inside = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    keep = (inside & (boxes[:, 0] < im_w) & (boxes[:, 1] < im_h)).nonzero(as_tuple=True)[0]
    if len(keep) == len(tile_idx):
        return merged, tile_idx
    return filter_instances(merged, keep), tile_idx[keep]


def _apply_score_threshold(merged: Dict[str, Any], threshold: float, tile_idx: torch.Tensor) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Drop predictions below ``threshold``, keeping ``tile_idx`` in step. Runs before merging."""
    scores = merged.get("scores")
    if threshold <= 0 or not isinstance(scores, torch.Tensor) or len(scores) != len(tile_idx):
        return merged, tile_idx
    keep = (scores.detach().cpu().float() >= threshold).nonzero(as_tuple=True)[0]
    if len(keep) == len(scores):
        return merged, tile_idx
    return filter_instances(merged, keep), tile_idx[keep]


def _containment_or_strict(value: Optional[float]) -> float:
    """A missing containment demands a whole object; 0.0 means no containment requirement at all."""
    return 1.0 if value is None else float(value)


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
    target_size: Tuple[int, int],
    mask_dtype: torch.dtype = torch.bool,
) -> Dict[str, Any]:
    """Offset one tile's predictions into the tiled image's space. Masks become ``MaskCrops`` that paste as ``mask_dtype``."""

    def xy_fn(xy: torch.Tensor) -> torch.Tensor:
        off = torch.tensor([offset_x, offset_y], dtype=torch.float32, device=xy.device)
        return xy.float() + off

    def mask_fn(masks: torch.Tensor) -> MaskCrops:
        return MaskCrops.from_masks(masks, target_size, (offset_x, offset_y), mask_dtype)

    return apply_coord_transform(result, xy_fn=xy_fn, mask_fn=mask_fn)


def _rescale_to_image(merged: Dict[str, Any], sx: float, sy: float, im_h: int, im_w: int, mask_dtype: torch.dtype) -> Dict[str, Any]:
    """Map results from the interpolated image's space back to the original image's."""

    def xy_fn(xy: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor([sx, sy], dtype=torch.float32, device=xy.device)
        return xy.float() * scale

    def mask_fn(masks: torch.Tensor) -> torch.Tensor:
        # resampling holds a float copy of both sides at once, so go in chunks: a crowded image carries hundreds
        # of full-image masks and converting them all at once costs more than the rest of the pipeline
        per_mask = 2 * (masks.shape[1] * masks.shape[2] + im_h * im_w)  # fp16: 2.4e-4 error against a 0.5 cut
        chunk = max(1, _MASK_RESIZE_BUDGET // per_mask)
        out = masks.new_empty((len(masks), im_h, im_w), dtype=mask_dtype)
        for start in range(0, len(masks), chunk):
            block = masks[start : start + chunk].to(torch.float16).unsqueeze(1)
            resized = torch.nn.functional.interpolate(block, size=(im_h, im_w), mode="bilinear", align_corners=False)
            out[start : start + chunk] = (resized.squeeze(1) > 0.5).to(mask_dtype)
        return out

    return apply_coord_transform(merged, xy_fn=xy_fn, mask_fn=mask_fn)


def _output_mask_dtype(tile_results: List[Dict[str, Any]]) -> torch.dtype:
    """The tiles' mask dtype; bool for float masks, which the Reconstructor casts back."""
    sample = next((r["masks"] for r in tile_results if isinstance(r.get("masks"), torch.Tensor) and len(r["masks"])), None)
    if sample is None or sample.is_floating_point():
        return torch.bool
    return sample.dtype


def _concat_tile_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    merged = {}
    all_keys = {k for r in results for k in r}

    for key in all_keys:
        vals = [r[key] for r in results if key in r and r[key] is not None]
        if not vals:
            continue
        if key == "segments":
            merged[key] = [seg for segs in vals for seg in segs]
        elif key == "classes":
            non_empty = [v for v in vals if len(v) > 0]
            merged[key] = np.concatenate(non_empty) if non_empty else vals[0]
        elif key == "masks" and any(isinstance(v, MaskCrops) for v in vals):
            # a tile without detections keeps its empty mask tensor
            merged[key] = MaskCrops.cat([v for v in vals if isinstance(v, MaskCrops)])
        else:  # boxes, scores, points, masks
            non_empty = [v for v in vals if len(v) > 0]
            merged[key] = torch.cat(non_empty) if non_empty else vals[0]

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

    device = result_device(result)
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
