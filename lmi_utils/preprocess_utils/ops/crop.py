from typing import Any, Dict, List, Optional, Tuple

import torch

from .._coords import apply_coord_transform
from ..operation import Operation


class CropOperation(Operation):
    """Crop each image to a per-image box; paste back into the original canvas on revert.

    Configuration:
        boxes: list of [x1, y1, x2, y2], one per input image.

    Metadata (per image):
        box: clamped [x1, y1, x2, y2] actually used for the crop.
        orig_size: [W, H] of the input image, used to restore the canvas on revert.
    """

    name = "crop"

    @classmethod
    def build_step(cls, *, boxes: List[List[int]], id: Optional[str] = None) -> Dict[str, Any]:
        return cls._finalize_step({"boxes": boxes}, id=id)

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Any]]:
        boxes = config.get("boxes")
        if not isinstance(boxes, list) or len(boxes) != len(images):
            raise ValueError(f"crop: 'boxes' must be a list of length {len(images)}, got {boxes!r}")

        out_images: List[torch.Tensor] = []
        meta: List[Any] = []
        for img, box in zip(images, boxes):
            if len(box) != 4:
                raise ValueError(f"crop: each box must have 4 elements [x1,y1,x2,y2], got {box!r}")
            H, W = img.shape[0], img.shape[1]
            x1 = max(0, min(W, int(round(float(box[0])))))
            y1 = max(0, min(H, int(round(float(box[1])))))
            x2 = max(x1, min(W, int(round(float(box[2])))))
            y2 = max(y1, min(H, int(round(float(box[3])))))
            out_images.append(img[y1:y2, x1:x2])
            meta.append({"box": [x1, y1, x2, y2], "orig_size": [W, H]})

        return out_images, meta

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> List[torch.Tensor]:
        if len(images) != len(metadata):
            raise ValueError(f"Image count ({len(images)}) doesn't match crop metadata count ({len(metadata)})")
        restored = []
        for img, m in zip(images, metadata):
            x1, y1, _x2, _y2 = m["box"]
            W, H = m["orig_size"]
            ch, cw = img.shape[0], img.shape[1]
            if img.dim() == 2:
                canvas = torch.zeros((H, W), dtype=img.dtype, device=img.device)
                canvas[y1 : y1 + ch, x1 : x1 + cw] = img
            else:
                C = img.shape[2]
                canvas = torch.zeros((H, W, C), dtype=img.dtype, device=img.device)
                canvas[y1 : y1 + ch, x1 : x1 + cw, :] = img
            restored.append(canvas)
        return restored

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(results) != len(metadata):
            raise ValueError(f"Result count ({len(results)}) doesn't match crop metadata count ({len(metadata)})")
        return [self._apply_single(r, m, forward=False) for r, m in zip(results, metadata)]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(results) != len(metadata):
            raise ValueError(f"Result count ({len(results)}) doesn't match crop metadata count ({len(metadata)})")
        return [self._apply_single(r, m, forward=True) for r, m in zip(results, metadata)]

    @staticmethod
    def _apply_single(result: Dict[str, Any], m: Dict[str, Any], *, forward: bool) -> Dict[str, Any]:
        x1, y1, x2, y2 = m["box"]
        W, H = m["orig_size"]

        def xy_fn(xy: torch.Tensor) -> torch.Tensor:
            off = torch.tensor([x1, y1], dtype=torch.float32, device=xy.device)
            xy = xy.float()
            return xy - off if forward else xy + off

        def mask_fn(masks: torch.Tensor) -> torch.Tensor:
            if forward:
                # crop full-image masks down to the box region
                return masks[:, y1:y2, x1:x2]
            n, mh, mw = masks.shape[0], masks.shape[1], masks.shape[2]
            canvas = torch.zeros((n, H, W), dtype=masks.dtype, device=masks.device)
            canvas[:, y1 : y1 + mh, x1 : x1 + mw] = masks
            return canvas

        return apply_coord_transform(result, xy_fn=xy_fn, mask_fn=mask_fn)
