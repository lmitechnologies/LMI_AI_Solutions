from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from lmi_utils.image_utils.tiler import Tiler

from .._coords import apply_coord_transform
from ..operation import Operation


class TileOperation(Operation):
    """Tile each image into fixed-size patches; remember per-image grid for stitching.

    Each input image expands to ``n_tiles_h * n_tiles_w`` output tiles in row-major order;
    ``revert_images`` / ``revert_coords`` consume the same flat tile list and re-group by
    metadata entry.

    Configuration:
        tile_size (int | [h, w], required): patch size.
        stride (int | [h, w], required): step between tile origins.
        scale_mode (str, optional): how the image is fit to the tile grid before slicing
            (``"padding"`` or ``"interpolation"``). Default ``"padding"``.
        overlap_mode (str, optional): how overlapping regions are merged on untile. Default ``"average"``.

    Metadata schema (per input image)::

        {
            "tile_size": [h, w], "stride": [h, w],
            "im_size": [H, W],           # original image size
            "scale_size": [H', W'],      # size after scale_mode fit, before tiling
            "n_tiles": [n_h, n_w],       # grid shape
            "batch_size": int, "num_channel": int,
            "scale_mode": str, "overlap_mode": str,
        }
    """

    name = "tile"

    @classmethod
    def build_step(
        cls,
        *,
        tile_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        scale_mode: str = "padding",
        overlap_mode: str = "average",
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return cls._finalize_step(
            {"tile_size": tile_size, "stride": stride, "scale_mode": scale_mode, "overlap_mode": overlap_mode},
            id=id,
        )

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Any]]:
        required_keys = {"tile_size", "stride"}
        if not required_keys.issubset(config.keys()):
            raise ValueError(f"Tiler configuration must contain keys: {required_keys}")

        scale_mode = config.get("scale_mode", "padding")
        overlap_mode = config.get("overlap_mode", "average")

        output_images = []
        tiler_metadata = []

        for img in images:
            ndim = img.dim()
            add_channel = False
            if ndim == 2:
                add_channel = True
                img = img.unsqueeze(-1)

            tiler = Tiler(tile_size=config["tile_size"], stride=config["stride"])
            img_batch = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            tiles_batch = tiler.tile(img_batch, mode=scale_mode)  # [N, C, H, W]

            tiles_list_chw = list(torch.unbind(tiles_batch, dim=0))
            tiles_list_hwc = [t.permute(1, 2, 0) for t in tiles_list_chw]
            if add_channel:
                tiles_list_hwc = [t.squeeze(-1) for t in tiles_list_hwc]

            output_images.extend(tiles_list_hwc)

            meta = tiler.to_dict()
            meta["overlap_mode"] = overlap_mode
            meta["scale_mode"] = scale_mode
            tiler_metadata.append(meta)

        return output_images, tiler_metadata

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> List[torch.Tensor]:
        restored_images = []
        cursor = 0

        for tiler_meta in metadata:
            tiler = Tiler.from_dict(tiler_meta)
            count = tiler_meta["n_tiles"][0] * tiler_meta["n_tiles"][1]

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

            scale_mode = tiler_meta.get("scale_mode", "padding")
            overlap_mode = tiler_meta.get("overlap_mode", "average")
            restored_batch = tiler.untile(batch_chw, scale_mode=scale_mode, overlap_mode=overlap_mode)

            restored_img = restored_batch.squeeze(0).permute(1, 2, 0)
            if add_channel:
                restored_img = restored_img.squeeze(-1)

            restored_images.append(restored_img)

        if cursor != len(images):
            raise RuntimeError(f"Tile reconstruction mismatch: processed {cursor} images, but received {len(images)}")

        return restored_images

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Project original-space coords into per-tile space.

        Expands the per-image results list into a per-tile list (one dict per emitted tile,
        row-major order matching ``forward``). Instances that don't overlap a given tile
        are dropped from that tile; partially overlapping ones are clipped:
            - xyxy boxes: axis-aligned clip to tile rect.
            - OBB boxes: Sutherland-Hodgman clip then refit a rect aligned with the original
              OBB's orientation (AABB in the OBB's local frame, rotated back). Preserves the
              source object's rotation rather than the clipped fragment's tightest fit.
            - segments: Sutherland-Hodgman clip; instances with empty clipped polygon are dropped.
            - masks: per-tile spatial slice.
        Keypoint visibility flags are set to 0 for any keypoint falling outside the tile,
        and instances with all keypoints invisible are dropped.

        Mask input shape convention mirrors ``revert_coords``:
            - interpolation: masks are in ``im_size``; resampled to ``scale_size`` before tiling.
            - padding: masks are in ``scale_size``; zero-padded to ``scale_size`` if smaller.
        """
        if len(results) != len(metadata):
            raise ValueError(f"tile: results count ({len(results)}) != metadata count ({len(metadata)})")
        output = []
        for r, m in zip(results, metadata):
            n_h, n_w = m["n_tiles"]
            for row in range(n_h):
                for col in range(n_w):
                    output.append(self._project_to_tile(r, m, row, col))
        return output

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # NOTE: per-tile instances are concatenated as-is; objects spanning tile seams remain
        # as separate labels (no NMS / IoU merge / mask union). Callers needing one label per
        # object must dedupe downstream.
        output = []
        cursor = 0

        for tiler_meta in metadata:
            n_tiles_h, n_tiles_w = tiler_meta["n_tiles"]
            count = n_tiles_h * n_tiles_w
            tile_results = results[cursor : cursor + count]
            cursor += count

            if len(tile_results) != count:
                raise RuntimeError(f"Expected {count} tile results, found {len(tile_results)}")

            output.append(self._merge_tile_coords(tile_results, tiler_meta))

        if cursor != len(results):
            raise RuntimeError(f"Tile coord reconstruction mismatch: processed {cursor}, received {len(results)}")

        return output

    @classmethod
    def _merge_tile_coords(cls, tile_results: List[Dict[str, Any]], tiler_meta: Dict[str, Any]) -> Dict[str, Any]:
        _, n_tiles_w = tiler_meta["n_tiles"]
        stride_h, stride_w = tiler_meta["stride"]
        im_h, im_w = tiler_meta["im_size"]
        scale_h, scale_w = tiler_meta["scale_size"]
        is_interp = tiler_meta.get("scale_mode", "padding") == "interpolation" and (scale_h != im_h or scale_w != im_w)
        sx = im_w / scale_w if is_interp else 1.0
        sy = im_h / scale_h if is_interp else 1.0

        target_size = (im_h, im_w) if is_interp else (scale_h, scale_w)

        shifted = []
        for idx, r in enumerate(tile_results):
            row = idx // n_tiles_w
            col = idx % n_tiles_w
            shifted.append(cls._shift_tile_coords(r, col * stride_w, row * stride_h, sx, sy, target_size))

        return cls._concat_tile_results(shifted)

    @staticmethod
    def _shift_tile_coords(
        result: Dict[str, Any], offset_x: int, offset_y: int, sx: float, sy: float, target_size: Tuple[int, int]
    ) -> Dict[str, Any]:
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
                masks = torch.nn.functional.interpolate(masks.float().unsqueeze(1), size=(new_h, new_w), mode="nearest").squeeze(1)
                paste_y = round(offset_y * sy)
                paste_x = round(offset_x * sx)
            else:
                masks = masks.float()
                paste_y, paste_x = offset_y, offset_x
            canvas = torch.zeros(len(masks), canvas_h, canvas_w, dtype=masks.dtype, device=masks.device)
            h_end = min(paste_y + masks.shape[1], canvas_h)
            w_end = min(paste_x + masks.shape[2], canvas_w)
            canvas[:, paste_y:h_end, paste_x:w_end] = masks[:, : h_end - paste_y, : w_end - paste_x]
            return canvas

        return apply_coord_transform(result, xy_fn=xy_fn, mask_fn=mask_fn)

    @staticmethod
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
            else:  # boxes, scores, points, masks
                non_empty = [v for v in vals if len(v) > 0]
                merged[key] = torch.cat(non_empty) if non_empty else vals[0]

        return merged

    @classmethod
    def _project_to_tile(cls, result: Dict[str, Any], m: Dict[str, Any], row: int, col: int) -> Dict[str, Any]:
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

        kept = torch.zeros(n, dtype=torch.bool)
        boxes_out = points_out = masks_out = None
        segments_out: Optional[List[torch.Tensor]] = None

        boxes = result.get("boxes")
        if boxes is not None and len(boxes):
            if boxes.ndim == 3:  # OBB (N, 4, 2)
                obb_t = to_tile_xy(boxes.reshape(-1, 2)).reshape(n, 4, 2)
                obb_kept = torch.zeros(n, dtype=torch.bool)
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

        return out


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
