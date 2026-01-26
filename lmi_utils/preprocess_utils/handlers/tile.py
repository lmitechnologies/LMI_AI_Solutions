from typing import Any, Dict, List, Tuple

import torch
from image_utils.tiler import Tiler


def tile_handler(images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """
    Wraps Tiler.
    Args:
        images (list[torch.Tensor]): List of input images (H, W, C).
        config (dict): Configuration for Tiler.
    """
    if not images:
        raise ValueError("Cannot tile empty image list")

    required_keys = {"tile_size", "stride"}
    if not required_keys.issubset(config.keys()):
        raise ValueError(f"Tiler configuration must contain keys: {required_keys}")

    scale_mode = config.get("scale_mode", "padding")
    overlap_mode = config.get("overlap_mode", "average")

    output_images = []
    tiler_metadata = []

    for img in images:
        if img.dim() != 3:
            raise ValueError(f"Expected 3D tensor (H, W, C), got {img.dim()}D tensor with shape {img.shape}")

        tiler = Tiler(tile_size=config["tile_size"], stride=config["stride"])
        img_batch = img.permute(2, 0, 1).unsqueeze(0)  # Convert to [1, C, H, W]
        tiles_batch = tiler.tile(img_batch, mode=scale_mode)  # Returns [N, C, H, W]

        # Convert back to List of (H, W, C)
        tiles_list_chw = list(torch.unbind(tiles_batch, dim=0))
        tiles_list_hwc = [t.permute(1, 2, 0) for t in tiles_list_chw]

        output_images.extend(tiles_list_hwc)

        metadata = tiler.save_metadata()
        metadata["overlap_mode"] = overlap_mode
        metadata["scale_mode"] = scale_mode
        tiler_metadata.append(metadata)

    return output_images, {"tiler_metadata": tiler_metadata}


def undo_tile_handler(images: List[torch.Tensor], meta: Dict[str, Any]) -> List[torch.Tensor]:
    """
    undoes the 'tile' operation.
    Args:
        images (list[torch.Tensor]): List of input tiles (H, W, C).
        meta (dict): Metadata containing 'tiler_metadata'.
    """
    if not images:
        raise ValueError("No input images provided for untile operation.")

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
        batch_hwc = torch.stack(batch_slice_hwc)  # Stack -> [N, H, W, C]
        batch_chw = batch_hwc.permute(0, 3, 1, 2)  # Permute -> [N, C, H, W]

        # Untile the batch -> [1, C, H, W]
        scale_mode = tiler_meta.get("scale_mode", "padding")
        overlap_mode = tiler_meta.get("overlap_mode", "average")
        restored_batch = tiler.untile(batch_chw, scale_mode=scale_mode, overlap_mode=overlap_mode)

        # Convert back to (H, W, C)
        restored_img = restored_batch.squeeze(0).permute(1, 2, 0)
        restored_images.append(restored_img)

    if cursor != len(images):
        raise RuntimeError(f"Tile reconstruction mismatch: processed {cursor} images, but received {len(images)}")

    return restored_images
