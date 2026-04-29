from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from lmi_utils.image_utils.tiler import Tiler


@torch.inference_mode()
def tile(images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """
    Wraps Tiler.
    Args:
        images (list[torch.Tensor]): List of input images (H, W, C).
        config (dict): Configuration for Tiler.
    """
    required_keys = {"tile_size", "stride"}
    if not required_keys.issubset(config.keys()):
        raise ValueError(f"Tiler configuration must contain keys: {required_keys}")

    scale_mode = config.get("scale_mode", "padding")
    overlap_mode = config.get("overlap_mode", "average")

    output_images = []
    tiler_metadata = []

    for img in images:
        ndim = img.dim()

        # Handle single channel
        add_channel = False
        if ndim == 2:
            add_channel = True
            img = img.unsqueeze(-1)  # Add channel dimension for grayscale images

        tiler = Tiler(tile_size=config["tile_size"], stride=config["stride"])
        img_batch = img.permute(2, 0, 1).unsqueeze(0)  # Convert to [1, C, H, W]
        tiles_batch = tiler.tile(img_batch, mode=scale_mode)  # Returns [N, C, H, W]

        # Convert back to List of (H, W, C)
        tiles_list_chw = list(torch.unbind(tiles_batch, dim=0))
        tiles_list_hwc = [t.permute(1, 2, 0) for t in tiles_list_chw]
        if add_channel:
            tiles_list_hwc = [t.squeeze(-1) for t in tiles_list_hwc]  # Remove added channel

        output_images.extend(tiles_list_hwc)

        metadata = tiler.to_dict()
        metadata["overlap_mode"] = overlap_mode
        metadata["scale_mode"] = scale_mode
        tiler_metadata.append(metadata)

    return output_images, {"metadata": tiler_metadata}


@torch.inference_mode()
def revert_tile(images: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> List[torch.Tensor]:
    """
    undoes the 'tile' operation.
    Args:
        images (list[torch.Tensor]): List of input tiles (H, W, C).
        metadata (list): Per-image tiler metadata list returned by tile.
    """
    tiler_meta_list = metadata
    restored_images = []
    cursor = 0

    for tiler_meta in tiler_meta_list:
        tiler = Tiler.from_dict(tiler_meta)
        count = tiler_meta["n_tiles"][0] * tiler_meta["n_tiles"][1]

        # Slice the batch
        batch_slice_hwc = images[cursor : cursor + count]
        cursor += count

        # Integrity Check
        if len(batch_slice_hwc) != count:
            raise RuntimeError(f"Expected {count} tiles, found {len(batch_slice_hwc)}")

        # Prepare for Untile
        batch_hwc = torch.stack(batch_slice_hwc)  # Stack -> [N, H, W, C] or [N, H, W,]

        ndim = batch_hwc.dim()
        if ndim not in {4, 3}:
            raise ValueError(f"Tile batch must have 3 or 4 dimensions (N, H, W) or (N, H, W, C). Got {ndim} dimensions.")

        # Handle single channel case
        add_channel = False
        if ndim == 3:
            add_channel = True
            batch_hwc = batch_hwc.unsqueeze(-1)
        batch_chw = batch_hwc.permute(0, 3, 1, 2)  # Permute -> [N, C, H, W]

        # Untile the batch -> [1, C, H, W]
        scale_mode = tiler_meta.get("scale_mode", "padding")
        overlap_mode = tiler_meta.get("overlap_mode", "average")
        restored_batch = tiler.untile(batch_chw, scale_mode=scale_mode, overlap_mode=overlap_mode)

        # Convert back to (H, W, C)
        restored_img = restored_batch.squeeze(0).permute(1, 2, 0)
        if add_channel:
            restored_img = restored_img.squeeze(-1)  # Remove added channel

        restored_images.append(restored_img)

    if cursor != len(images):
        raise RuntimeError(f"Tile reconstruction mismatch: processed {cursor} images, but received {len(images)}")

    return restored_images


@torch.inference_mode()
def revert_tile_coords(results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reverses the tile operation on detection coordinates.

    Groups the flat per-tile result list back to per-original-image results, shifts each
    tile's coordinates by its position in the (possibly scaled) image, then concatenates
    detections across all tiles.

    Args:
        results: Flat list of per-tile result dicts (boxes, scores, classes, segments, points).
        metadata: Per-original-image tiler metadata returned by tile().

    Returns:
        List of per-original-image result dicts with coordinates in original image space.
    """
    output = []
    cursor = 0

    for tiler_meta in metadata:
        n_tiles_h, n_tiles_w = tiler_meta["n_tiles"]
        count = n_tiles_h * n_tiles_w
        tile_results = results[cursor : cursor + count]
        cursor += count

        if len(tile_results) != count:
            raise RuntimeError(f"Expected {count} tile results, found {len(tile_results)}")

        output.append(_merge_tile_coords(tile_results, tiler_meta))

    if cursor != len(results):
        raise RuntimeError(f"Tile coord reconstruction mismatch: processed {cursor}, received {len(results)}")

    return output


def _merge_tile_coords(tile_results: List[Dict[str, Any]], tiler_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Shift and concatenate results from all tiles of one original image."""
    n_tiles_h, n_tiles_w = tiler_meta["n_tiles"]
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
        shifted.append(_shift_tile_coords(r, col * stride_w, row * stride_h, sx, sy, target_size))

    return _concat_tile_results(shifted)


def _shift_tile_coords(
    result: Dict[str, Any], offset_x: int, offset_y: int, sx: float, sy: float, target_size: Tuple[int, int]
) -> Dict[str, Any]:
    """Add tile offset and apply optional interpolation scale to all coordinate fields."""
    out = dict(result)

    boxes = result.get("boxes")
    if boxes is not None and len(boxes):
        if boxes.ndim == 3:  # OBB (N, 4, 2)
            off = torch.tensor([offset_x, offset_y], dtype=torch.float32, device=boxes.device)
            scale = torch.tensor([sx, sy], dtype=torch.float32, device=boxes.device)
        else:  # regular (N, 4) [x1, y1, x2, y2]
            off = torch.tensor([offset_x, offset_y, offset_x, offset_y], dtype=torch.float32, device=boxes.device)
            scale = torch.tensor([sx, sy, sx, sy], dtype=torch.float32, device=boxes.device)
        out["boxes"] = boxes.float() + off
        if sx != 1.0 or sy != 1.0:
            out["boxes"] = out["boxes"] * scale

    segs = result.get("segments")
    if segs is not None and len(segs):
        off = torch.tensor([offset_x, offset_y], dtype=torch.float32)
        scale = torch.tensor([sx, sy], dtype=torch.float32) if sx != 1.0 or sy != 1.0 else None
        new_segs = []
        for seg in segs:
            if len(seg):
                shifted = seg.float() + off.to(seg.device)
                if scale is not None:
                    shifted = shifted * scale.to(seg.device)
                new_segs.append(shifted)
            else:
                new_segs.append(seg)
        out["segments"] = new_segs

    pts = result.get("points")
    if pts is not None and len(pts):
        visibility = None
        pts_xy = pts
        if pts.shape[-1] == 3:
            visibility = pts[:, :, -1]
            pts_xy = pts[:, :, :2]
        off = torch.tensor([offset_x, offset_y], dtype=torch.float32, device=pts_xy.device)
        shifted_xy = pts_xy.float() + off
        if sx != 1.0 or sy != 1.0:
            scale = torch.tensor([sx, sy], dtype=torch.float32, device=pts_xy.device)
            shifted_xy = shifted_xy * scale
        out["points"] = torch.cat((shifted_xy, visibility.unsqueeze(-1)), dim=-1) if visibility is not None else shifted_xy

    masks = result.get("masks")
    if masks is not None and len(masks):
        canvas_h, canvas_w = target_size
        if sx != 1.0 or sy != 1.0:
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
        out["masks"] = canvas

    return out


def _concat_tile_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate per-tile result dicts into a single result dict."""
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
