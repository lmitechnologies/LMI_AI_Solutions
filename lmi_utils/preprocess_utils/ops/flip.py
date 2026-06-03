from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

from .._coords import apply_coord_transform
from ..operation import Config, Meta, Operation


@dataclass
class FlipConfig(Config):
    """Flip each image horizontally and/or vertically."""

    lr: bool = False
    ud: bool = False


@dataclass
class FlipMeta(Meta):
    """Batched flip metadata.

    lr/ud: per-image flags (same for all entries in a single forward call,
    repeated to preserve the per-image-list invariant).
    sizes: per-image [W, H] of the flipped image.
    """

    lr: List[bool] = field(default_factory=list)
    ud: List[bool] = field(default_factory=list)
    sizes: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.sizes)
        if not (len(self.lr) == n and len(self.ud) == n):
            raise ValueError("FlipMeta: field lengths must match")


class FlipOperation(Operation[FlipConfig, FlipMeta]):
    config_cls = FlipConfig
    meta_cls = FlipMeta

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: FlipConfig) -> Tuple[List[torch.Tensor], FlipMeta]:
        lr = bool(config.lr)
        ud = bool(config.ud)

        out_images: List[torch.Tensor] = []
        lrs: List[bool] = []
        uds: List[bool] = []
        sizes: List[List[int]] = []
        for img in images:
            h, w = img.shape[:2]
            out_images.append(_flip(img, lr, ud))
            lrs.append(lr)
            uds.append(ud)
            sizes.append([w, h])
        return out_images, FlipMeta(lr=lrs, ud=uds, sizes=sizes)

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], meta: FlipMeta) -> List[torch.Tensor]:
        return [_flip(img, lr, ud) for img, lr, ud in zip(images, meta.lr, meta.ud)]

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], meta: FlipMeta) -> List[Dict[str, Any]]:
        return [_apply_flip(r, lr, ud, s) for r, lr, ud, s in zip(results, meta.lr, meta.ud, meta.sizes)]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], meta: FlipMeta) -> List[Dict[str, Any]]:
        return [_apply_flip(r, lr, ud, s) for r, lr, ud, s in zip(results, meta.lr, meta.ud, meta.sizes)]


def _flip(img: torch.Tensor, lr: bool, ud: bool) -> torch.Tensor:
    out = img
    if lr:
        out = torch.flip(out, dims=[1])
    if ud:
        out = torch.flip(out, dims=[0])
    return out


def _apply_flip(result: Dict[str, Any], lr: bool, ud: bool, size: List[int]) -> Dict[str, Any]:
    w, h = size

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
