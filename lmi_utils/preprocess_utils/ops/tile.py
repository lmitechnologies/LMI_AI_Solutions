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
        overlap_mode (str, optional): how overlapping regions are merged on untile
            (e.g. ``"average"``). Default ``"average"``.

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
    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
