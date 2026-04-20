from typing import Any, Dict, List, Tuple

import torch

from lmi_utils.gadget_utils.pipeline_utils import revert_mask_to_origin
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
