from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from lmi_utils.image_utils.tiler import Tiler

from .._coords import apply_coord_transform
from ..operation import Config, Meta, Operation


@dataclass
class TileConfig(Config):
    """Tile each image into fixed-size patches; remember per-image grid for stitching.

    tile_size: int or [h, w] — patch size.
    stride: int or [h, w] — step between tile origins.
    scale_mode: how the image is fit to the tile grid before slicing ("padding" or "interpolation").
    overlap_mode: how overlapping regions are merged on untile ("average", "max", ...).
    """

    tile_size: Union[int, List[int], None] = None
    stride: Union[int, List[int], None] = None
    scale_mode: str = "padding"
    overlap_mode: str = "average"

    def __post_init__(self):
        if self.tile_size is None or self.stride is None:
            raise ValueError("TileConfig: 'tile_size' and 'stride' are required")


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

    def __post_init__(self):
        n = len(self.n_tiles)
        for name in (
            "tile_sizes",
            "strides",
            "im_sizes",
            "scale_sizes",
            "batch_sizes",
            "num_channels",
            "scale_modes",
            "overlap_modes",
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
        }


class TileOperation(Operation[TileConfig, TileMeta]):
    config_cls = TileConfig
    meta_cls = TileMeta

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
    def revert_coords(self, results: List[Dict[str, Any]], meta: TileMeta) -> List[Dict[str, Any]]:
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
        shifted.append(_shift_tile_coords(r, col * stride_w, row * stride_h, sx, sy, target_size))

    return _concat_tile_results(shifted)


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
