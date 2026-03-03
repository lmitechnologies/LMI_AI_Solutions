from typing import Any, Dict, List, Tuple

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
        if ndim not in {2, 3}:
            raise ValueError(f"Input image must have 2 or 3 dimensions (H, W) or (H, W, C). Got {ndim} dimensions.")

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

    return output_images, {"tiler_metadata": tiler_metadata}


@torch.inference_mode()
def revert_tile(images: List[torch.Tensor], meta: Dict[str, Any]) -> List[torch.Tensor]:
    """
    undoes the 'tile' operation.
    Args:
        images (list[torch.Tensor]): List of input tiles (H, W, C).
        meta (dict): Metadata containing 'tiler_metadata'.
    """
    if "tiler_metadata" not in meta:
        raise KeyError(f"Metadata missing required key 'tiler_metadata'. Got keys: {list(meta.keys())}")

    tiler_meta_list = meta["tiler_metadata"]
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
