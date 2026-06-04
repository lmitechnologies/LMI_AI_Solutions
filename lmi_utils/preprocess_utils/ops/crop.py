from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

from .._coords import apply_coord_transform
from ..operation import Config, Meta, Operation


@dataclass
class CropConfig(Config):
    """Crop each image to a per-image box.

    boxes: list of [x1, y1, x2, y2], one per input image.
    """

    boxes: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.boxes, list) or not self.boxes:
            raise ValueError(f"CropConfig: 'boxes' must be a non-empty list, got {self.boxes!r}")
        for i, box in enumerate(self.boxes):
            if len(box) != 4:
                raise ValueError(f"CropConfig: boxes[{i}] must have 4 elements [x1,y1,x2,y2], got {box!r}")


@dataclass
class CropMeta(Meta):
    """Batched crop metadata.

    boxes: clamped [x1, y1, x2, y2] actually used per image.
    orig_sizes: [W, H] per image, used to restore the canvas on revert.
    """

    boxes: List[List[int]] = field(default_factory=list)
    orig_sizes: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        if len(self.boxes) != len(self.orig_sizes):
            raise ValueError(f"CropMeta: boxes ({len(self.boxes)}) and orig_sizes ({len(self.orig_sizes)}) lengths must match")


class CropOperation(Operation[CropConfig, CropMeta]):
    config_cls = CropConfig
    meta_cls = CropMeta

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: CropConfig) -> Tuple[List[torch.Tensor], CropMeta]:
        if len(config.boxes) != len(images):
            raise ValueError(f"crop: boxes length ({len(config.boxes)}) != image count ({len(images)})")

        out_images: List[torch.Tensor] = []
        out_boxes: List[List[int]] = []
        out_sizes: List[List[int]] = []
        for img, box in zip(images, config.boxes):
            H, W = img.shape[0], img.shape[1]
            x1 = max(0, min(W, int(round(float(box[0])))))
            y1 = max(0, min(H, int(round(float(box[1])))))
            x2 = max(x1, min(W, int(round(float(box[2])))))
            y2 = max(y1, min(H, int(round(float(box[3])))))
            out_images.append(img[y1:y2, x1:x2])
            out_boxes.append([x1, y1, x2, y2])
            out_sizes.append([W, H])

        return out_images, CropMeta(boxes=out_boxes, orig_sizes=out_sizes)

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], meta: CropMeta) -> List[torch.Tensor]:
        if len(images) != len(meta.boxes):
            raise ValueError(f"crop: image count ({len(images)}) != meta count ({len(meta.boxes)})")
        restored = []
        for img, box, size in zip(images, meta.boxes, meta.orig_sizes):
            x1, y1, _, _ = box
            W, H = size
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
    def revert_coords(self, results: List[Dict[str, Any]], meta: CropMeta) -> List[Dict[str, Any]]:
        if len(results) != len(meta.boxes):
            raise ValueError(f"crop: results count ({len(results)}) != meta count ({len(meta.boxes)})")
        return [self._apply_single(r, b, s, forward=False) for r, b, s in zip(results, meta.boxes, meta.orig_sizes)]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], meta: CropMeta) -> List[Dict[str, Any]]:
        if len(results) != len(meta.boxes):
            raise ValueError(f"crop: results count ({len(results)}) != meta count ({len(meta.boxes)})")
        return [self._apply_single(r, b, s, forward=True) for r, b, s in zip(results, meta.boxes, meta.orig_sizes)]

    @staticmethod
    def _apply_single(result: Dict[str, Any], box: List[int], size: List[int], *, forward: bool) -> Dict[str, Any]:
        x1, y1, x2, y2 = box
        W, H = size

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
