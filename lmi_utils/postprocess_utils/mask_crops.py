"""Binary instance masks stored as the crop inside each mask's box.

N full-image masks cost N x H x W; the crops cost the sum of their box areas, often hundreds of times less.
"""

from typing import List, Optional, Sequence, Tuple

import torch


def binarize_masks(masks: torch.Tensor) -> torch.Tensor:
    """Instance masks as bool. Floats threshold at 0.5, matching the resampling path."""
    if masks.dtype == torch.bool:
        return masks
    return masks > 0.5 if masks.is_floating_point() else masks != 0


def boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """(N, 4) xyxy box around each mask's pixels, on the masks' device. Max edges are exclusive; an empty mask gets zeros."""
    m = binarize_masks(masks)
    rows, cols = m.any(dim=2), m.any(dim=1)
    h, w = rows.shape[1], cols.shape[1]
    # argmax returns the first maximum, so it finds the first set row/column from each end
    y0, y1 = rows.float().argmax(dim=1), h - rows.flip(1).float().argmax(dim=1)
    x0, x1 = cols.float().argmax(dim=1), w - cols.flip(1).float().argmax(dim=1)
    boxes = torch.stack([x0, y0, x1, y1], dim=1).float()
    boxes[~rows.any(dim=1)] = 0
    return boxes


class MaskCrops:
    """N binary masks in an (H, W) image, each kept as a bool CPU crop of its box.

    ``crops[k]`` has shape (y1 - y0, x1 - x0) for ``boxes[k]`` = (x0, y0, x1, y1), a (N, 4) long CPU tensor. A box may be
    looser than its mask's pixels; an empty mask has a zero box. ``paste`` rebuilds full-image masks of ``dtype`` on
    ``device``.
    """

    def __init__(self, crops: List[torch.Tensor], boxes: torch.Tensor, size: Tuple[int, int], device: torch.device, dtype: torch.dtype):
        self.crops = crops
        self.boxes = boxes
        self.size = (int(size[0]), int(size[1]))
        self.device = device
        self.dtype = dtype
        self._areas: Optional[torch.Tensor] = None

    @classmethod
    def from_masks(
        cls, masks: torch.Tensor, size: Tuple[int, int], offset: Tuple[int, int] = (0, 0), dtype: Optional[torch.dtype] = None
    ) -> "MaskCrops":
        """Crops of (N, h, w) masks placed with their top-left at ``offset`` (x, y) in an image of ``size`` (H, W).
        Pixels outside the image are dropped. ``dtype`` defaults to the masks' own."""
        m = binarize_masks(masks)
        ox, oy = int(offset[0]), int(offset[1])
        shift = torch.tensor([ox, oy, ox, oy])
        boxes = boxes_from_masks(m).long().cpu() + shift
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, size[1])
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, size[0])
        empty = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
        boxes[empty] = 0
        # slice on the masks' device and copy all crops back in one transfer
        pieces = [m[k, y0 - oy : y1 - oy, x0 - ox : x1 - ox].reshape(-1) for k, (x0, y0, x1, y1) in enumerate(boxes.tolist())]
        flat = torch.cat(pieces).cpu() if pieces else torch.zeros(0, dtype=torch.bool)
        crops = [p.view(y1 - y0, x1 - x0) for p, (x0, y0, x1, y1) in zip(torch.split(flat, [len(p) for p in pieces]), boxes.tolist())]
        return cls(crops, boxes, size, masks.device, masks.dtype if dtype is None else dtype)

    @classmethod
    def cat(cls, parts: Sequence["MaskCrops"]) -> "MaskCrops":
        """Join crops of the same image, in order."""
        first = parts[0]
        return cls([c for p in parts for c in p.crops], torch.cat([p.boxes for p in parts]), first.size, first.device, first.dtype)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, keep: torch.Tensor) -> "MaskCrops":
        """The crops at the (K,) indices ``keep``, as new lists: editing the result leaves this one unchanged."""
        idx = keep.tolist()
        out = MaskCrops([self.crops[i] for i in idx], self.boxes[keep], self.size, self.device, self.dtype)
        if self._areas is not None:
            out._areas = self._areas[keep]
        return out

    def areas(self) -> torch.Tensor:
        """(N,) float pixel count of each mask."""
        if self._areas is None:
            self._areas = torch.tensor([float(c.sum()) for c in self.crops], dtype=torch.float32)
        return self._areas

    def intersection(self, a: int, b: int) -> float:
        """Pixels shared by masks a and b."""
        ax0, ay0, ax1, ay1 = self.boxes[a].tolist()
        bx0, by0, bx1, by1 = self.boxes[b].tolist()
        x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        pa = self.crops[a][y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0]
        pb = self.crops[b][y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0]
        return float((pa & pb).sum())

    def union(self, members: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crop and (4,) box of masks ``members`` combined."""
        boxes = self.boxes[list(members)]
        boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        if not len(boxes):
            return torch.zeros(0, 0, dtype=torch.bool), torch.zeros(4, dtype=torch.long)
        box = torch.cat([boxes[:, :2].amin(dim=0), boxes[:, 2:].amax(dim=0)])
        x0, y0, x1, y1 = box.tolist()
        crop = torch.zeros(y1 - y0, x1 - x0, dtype=torch.bool)
        for m in members:
            mx0, my0, mx1, my1 = self.boxes[m].tolist()
            if mx1 > mx0 and my1 > my0:
                crop[my0 - y0 : my1 - y0, mx0 - x0 : mx1 - x0] |= self.crops[m]
        return crop, box

    def set(self, k: int, crop: torch.Tensor, box: torch.Tensor) -> None:
        """Replace mask k."""
        self.crops[k] = crop
        self.boxes[k] = box
        self._areas = None

    def paste(self) -> torch.Tensor:
        """(N, H, W) full-image masks of ``dtype`` on ``device``."""
        out = torch.zeros(len(self), *self.size, dtype=self.dtype, device=self.device)
        if not len(self):
            return out
        flat = torch.cat([c.reshape(-1) for c in self.crops]).to(self.device)  # one transfer for all crops
        for k, (piece, (x0, y0, x1, y1)) in enumerate(zip(torch.split(flat, [c.numel() for c in self.crops]), self.boxes.tolist())):
            if len(piece):
                out[k, y0:y1, x0:x1] = piece.view(y1 - y0, x1 - x0)
        return out
