from typing import Any, Dict, List, Optional, Tuple

import torch

from .._coords import apply_coord_transform
from ..operation import Operation


class FlipOperation(Operation):
    """Flip each image horizontally and/or vertically.

    Configuration:
        lr (bool, optional): flip left-right. Default False.
        ud (bool, optional): flip up-down. Default False.

    Metadata schema (per image)::

        {"lr": bool, "ud": bool, "size": [w, h]}
    """

    name = "flip"

    @classmethod
    def build_step(cls, *, lr: bool = False, ud: bool = False, id: Optional[str] = None) -> Dict[str, Any]:
        return cls._finalize_step({"lr": lr, "ud": ud}, id=id)

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        lr = bool(config.get("lr", False))
        ud = bool(config.get("ud", False))

        out_images: List[torch.Tensor] = []
        meta_list: List[Dict[str, Any]] = []
        for img in images:
            h, w = img.shape[:2]
            meta = {"lr": lr, "ud": ud, "size": [w, h]}
            out_images.append(self._flip(img, meta))
            meta_list.append(meta)
        return out_images, meta_list

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> List[torch.Tensor]:
        # flip is its own inverse
        return [self._flip(img, m) for img, m in zip(images, metadata)]

    @staticmethod
    def _flip(img: torch.Tensor, m: Dict[str, Any]) -> torch.Tensor:
        out = img
        if m.get("lr"):
            out = torch.flip(out, dims=[1])
        if m.get("ud"):
            out = torch.flip(out, dims=[0])
        return out

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [_apply_flip(r, m) for r, m in zip(results, metadata)]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # flip is involutive; same code path
        return [_apply_flip(r, m) for r, m in zip(results, metadata)]


def _apply_flip(result: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, Any]:
    lr = m.get("lr", False)
    ud = m.get("ud", False)
    w, h = m["size"]

    def xy_fn(xy: torch.Tensor) -> torch.Tensor:
        xy = xy.float().clone()
        if lr:
            xy[..., 0] = w - xy[..., 0]
        if ud:
            xy[..., 1] = h - xy[..., 1]
        return xy

    def box_fn(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.ndim == 3:  # OBB
            n = boxes.shape[0]
            return xy_fn(boxes.reshape(-1, 2)).reshape(n, 4, 2)
        # xyxy: flip then swap corner pairs so x1<x2, y1<y2 is preserved
        out = boxes.float().clone()
        if lr:
            x1 = w - out[:, 0]
            x2 = w - out[:, 2]
            out[:, 0] = torch.minimum(x1, x2)
            out[:, 2] = torch.maximum(x1, x2)
        if ud:
            y1 = h - out[:, 1]
            y2 = h - out[:, 3]
            out[:, 1] = torch.minimum(y1, y2)
            out[:, 3] = torch.maximum(y1, y2)
        return out

    def mask_fn(masks: torch.Tensor) -> torch.Tensor:
        out = masks
        if lr:
            out = torch.flip(out, dims=[-1])
        if ud:
            out = torch.flip(out, dims=[-2])
        return out

    return apply_coord_transform(result, xy_fn=xy_fn, box_fn=box_fn, mask_fn=mask_fn)
