from typing import Any, Dict, List, Tuple

import torch

from lmi_utils.gadget_utils.pipeline_utils import revert_mask_to_origin, revert_masks_to_origin, revert_to_origin
from lmi_utils.image_utils.img_resize import resize_and_pad

from .._coords import apply_coord_transform
from ..operation import Operation


class ResizeOperation(Operation):
    """Resize-and-pad each image; remember per-image ops for inversion."""

    name = "resize"

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Any]]:
        resize_kwargs = {
            "width": config.get("width"),
            "height": config.get("height"),
            "preserve_aspect": config.get("preserve_aspect", False),
            "mode": config.get("mode", "bilinear"),
        }

        output_images = []
        image_ops_list = []
        for img in images:
            processed, ops = resize_and_pad(img, return_operators=True, **resize_kwargs)
            output_images.append(processed)
            image_ops_list.append(ops)

        return output_images, image_ops_list

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], metadata: List[Any]) -> List[torch.Tensor]:
        if len(images) != len(metadata):
            raise ValueError(f"Image count ({len(images)}) doesn't match ops count ({len(metadata)})")
        return [revert_mask_to_origin(image, ops) for image, ops in zip(images, metadata)]

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Any]) -> List[Dict[str, Any]]:
        if len(results) != len(metadata):
            raise ValueError(f"Result count ({len(results)}) doesn't match ops count ({len(metadata)})")
        return [self._revert_coords_single(r, ops) for r, ops in zip(results, metadata)]

    @staticmethod
    def _revert_coords_single(result: Dict[str, Any], ops: list) -> Dict[str, Any]:
        if not ops:
            return result

        def box_fn(boxes: torch.Tensor) -> torch.Tensor:
            # OBB: per-box stack (each (4, 2) is a valid (N, 2) input).
            # xyxy: pass (N, 4) natively so flip's corner-swap is correct.
            if boxes.ndim == 3:
                return torch.stack([revert_to_origin(box, ops) for box in boxes])
            return revert_to_origin(boxes, ops)

        return apply_coord_transform(
            result,
            xy_fn=lambda xy: revert_to_origin(xy, ops),
            box_fn=box_fn,
            mask_fn=lambda masks: revert_masks_to_origin(masks, ops),
        )
