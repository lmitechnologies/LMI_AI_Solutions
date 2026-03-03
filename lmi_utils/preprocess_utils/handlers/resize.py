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

    return output_images, {"ops": image_ops_list}


@torch.inference_mode()
def revert_resize(images: List[torch.Tensor], meta: Dict[str, Any]) -> List[torch.Tensor]:
    """
    Reverses the composite 'resize_and_pad' operation.
    Args:
        images (list[torch.Tensor]): List of input images (H, W, C).
        meta (dict): Configuration containing 'ops' for each image.
    """
    if "ops" not in meta:
        raise KeyError("Metadata missing required key 'ops'")

    image_ops_list = meta["ops"]
    if len(images) != len(image_ops_list):
        raise ValueError(f"Image count ({len(images)}) doesn't match ops count ({len(image_ops_list)})")

    output_images = [revert_mask_to_origin(image, ops) for image, ops in zip(images, image_ops_list)]
    return output_images
