from typing import Any, Dict, List, Tuple

import torch

from lmi_utils.gadget_utils.pipeline_utils import revert_mask_to_origin, revert_masks_to_origin, revert_to_origin
from lmi_utils.image_utils.img_resize import resize_and_pad


@torch.inference_mode()
def resize(images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """
    Wraps the user's custom 'resize_and_pad' function.

    Args:
        images (list[torch.Tensor]): List of input images (H, W, C).
        config (dict): Configuration for resize_and_pad.
    """
    # Map config keys to function arguments
    resize_configs = {
        "width": config.get("width"),
        "height": config.get("height"),
        "preserve_aspect": config.get("preserve_aspect", False),
        "mode": config.get("mode", "bilinear"),
    }

    output_images = []
    image_ops_list = []
    for img in images:
        processed, ops = resize_and_pad(
            img,
            return_operators=True,
            **resize_configs,
        )

        output_images.append(processed)
        image_ops_list.append(ops)

    return output_images, {"metadata": image_ops_list}


@torch.inference_mode()
def revert_resize(images: List[torch.Tensor], metadata: List[Any]) -> List[torch.Tensor]:
    """
    Reverses the composite 'resize_and_pad' operation.
    Args:
        images (list[torch.Tensor]): List of input images (H, W, C).
        metadata (list): Per-image ops list returned by resize.
    """
    if len(images) != len(metadata):
        raise ValueError(f"Image count ({len(images)}) doesn't match ops count ({len(metadata)})")

    output_images = [revert_mask_to_origin(image, ops) for image, ops in zip(images, metadata)]
    return output_images


def _revert_coords_single(result: Dict[str, Any], ops: list) -> Dict[str, Any]:
    """Apply coordinate reversion to a single-image result dict using resize_and_pad ops."""
    if not ops:
        return result

    reverted = dict(result)

    boxes = result.get("boxes")
    if boxes is not None and len(boxes):
        if boxes.ndim == 3:  # OBB (N, 4, 2)
            reverted["boxes"] = torch.stack([revert_to_origin(box, ops) for box in boxes])
        else:
            reverted["boxes"] = revert_to_origin(boxes, ops)

    masks = result.get("masks")
    if masks is not None and len(masks):
        reverted["masks"] = revert_masks_to_origin(masks, ops)

    segments = result.get("segments")
    if segments is not None and len(segments):
        reverted["segments"] = [revert_to_origin(seg, ops) if len(seg) else seg for seg in segments]

    points = result.get("points")
    if points is not None and len(points):
        pts_xy = points
        visibility = None
        if points.shape[-1] == 3:
            visibility = points[:, :, -1]
            pts_xy = points[:, :, :2]
        reverted_pts = [revert_to_origin(p, ops) for p in pts_xy]
        if visibility is not None:
            reverted_pts = [torch.cat((p, v.unsqueeze(-1)), dim=-1) for p, v in zip(reverted_pts, visibility)]
        reverted["points"] = torch.stack(reverted_pts)

    return reverted


def revert_resize_coords(results: List[Dict[str, Any]], metadata: List[Any]) -> List[Dict[str, Any]]:
    """
    Reverses the resize operation on detection coordinates.

    Args:
        results: List of per-image result dicts with keys boxes, masks, segments, points.
        metadata: Per-image ops lists as returned by resize_and_pad (revert_to_origin format).

    Returns:
        List of per-image result dicts with coordinates in original image space.
    """
    if len(results) != len(metadata):
        raise ValueError(f"Result count ({len(results)}) doesn't match ops count ({len(metadata)})")
    return [_revert_coords_single(r, ops) for r, ops in zip(results, metadata)]
