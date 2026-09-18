"""Binary instance masks stored as the crop inside each mask's box.

N full-image masks cost N x H x W; the crops cost the sum of their box areas, often hundreds of times less.

The crops live end to end in one tensor on the masks' own device, not as a list of small tensors, so the
operations below are single kernels instead of a pass per mask.
"""

from typing import Optional, Sequence, Tuple

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


def _offsets(sizes: torch.Tensor) -> torch.Tensor:
    """(N + 1,) start of each run in a flat buffer of runs of ``sizes``."""
    return torch.cat([torch.zeros(1, dtype=torch.long, device=sizes.device), sizes.cumsum(0)])


def _runs(sizes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """For runs of ``sizes`` laid end to end: each element's run, and its position within that run."""
    dev = sizes.device
    item = torch.repeat_interleave(torch.arange(len(sizes), device=dev), sizes)
    return item, torch.arange(int(sizes.sum()), device=dev) - (sizes.cumsum(0) - sizes)[item]


def _cells(width: torch.Tensor, height: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For a set of differently sized rectangles, one entry per cell: its rectangle, its row and its column.

    Every rectangle's cells lie end to end, so a set of them can be gathered or scattered in one indexing call
    with nothing allocated for the gaps between their sizes.
    """
    item, pos = _runs(width * height)
    w = width[item]
    return item, torch.div(pos, w, rounding_mode="floor"), pos % w


def _image_index(boxes: torch.Tensor, grid: Tuple[int, int], offset: Tuple[int, int] = (0, 0)) -> torch.Tensor:
    """Flat index of every crop cell into (N, H, W) images of ``grid``, for crops placed at ``boxes``.

    ``offset`` (x, y) is where those images sit, so the boxes come back to their own coordinates.
    """
    h, w = grid
    ox, oy = offset
    wh = boxes[:, 2:] - boxes[:, :2]
    item, row, col = _cells(wh[:, 0], wh[:, 1])
    return item * (h * w) + (row + boxes[item, 1] - oy) * w + col + boxes[item, 0] - ox


def _rows(width: torch.Tensor, height: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The same rectangles as one row per rectangle row, over a column grid as wide as the widest.

    ``valid`` is false past each rectangle's own width. Costs the padding ``_cells`` avoids, but a reduction
    along a row is then a contiguous one, which beats scattering every cell.
    """
    item, row = _runs(height)
    col = torch.arange(int(width.max()) if len(width) else 0, device=width.device)[None, :]
    return item, row, col, col < width[item][:, None]


class MaskCrops:
    """N binary masks in an (H, W) image, each kept as the bool crop of its box, all in one buffer.

    ``boxes[k]`` = (x0, y0, x1, y1) is mask k's placement, a (N, 4) long tensor; its pixels are
    ``data[offset[k]:offset[k + 1]]`` read as (y1 - y0, x1 - x0). The boxes alone fix that layout, so ``offset``
    is derived from them rather than stored alongside. A box may be looser than its mask's pixels; an empty mask
    has a zero box. ``paste`` rebuilds full-image masks of ``dtype``.
    """

    def __init__(self, data: torch.Tensor, boxes: torch.Tensor, size: Tuple[int, int], dtype: torch.dtype):
        self.data = data
        self.boxes = boxes
        self.offset = _offsets(self.wh.prod(dim=1))
        self.size = (int(size[0]), int(size[1]))
        self.dtype = dtype
        self._areas: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def wh(self) -> torch.Tensor:
        """(N, 2) width and height of each crop."""
        return self.boxes[:, 2:] - self.boxes[:, :2]

    @classmethod
    def from_masks(
        cls, masks: torch.Tensor, size: Tuple[int, int], offset: Tuple[int, int] = (0, 0), dtype: Optional[torch.dtype] = None
    ) -> "MaskCrops":
        """Crops of (N, h, w) masks placed with their top-left at ``offset`` (x, y) in an image of ``size`` (H, W).
        Pixels outside the image are dropped. ``dtype`` defaults to the masks' own."""
        m = binarize_masks(masks)
        dev = m.device
        ox, oy = int(offset[0]), int(offset[1])
        boxes = boxes_from_masks(m).long() + torch.tensor([ox, oy, ox, oy], device=dev)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, size[1])
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, size[0])
        boxes[(boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])] = 0

        data = m.reshape(-1)[_image_index(boxes, (m.shape[1], m.shape[2]), (ox, oy))]
        return cls(data, boxes, size, masks.dtype if dtype is None else dtype)

    @classmethod
    def cat(cls, parts: Sequence["MaskCrops"]) -> "MaskCrops":
        """Join crops of the same image, in order."""
        first = parts[0]
        return cls(torch.cat([p.data for p in parts]), torch.cat([p.boxes for p in parts]), first.size, first.dtype)

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, keep: torch.Tensor) -> "MaskCrops":
        """The crops at the (K,) indices ``keep``, copied: editing the result leaves this one unchanged."""
        at = keep.to(self.device)
        sizes = self.offset.diff()[at]
        item, pos = _runs(sizes)
        out = MaskCrops(self.data[self.offset[at][item] + pos], self.boxes[at], self.size, self.dtype)
        if self._areas is not None:
            out._areas = self._areas[at]
        return out

    def areas(self) -> torch.Tensor:
        """(N,) float pixel count of each mask, on the masks' own device."""
        if self._areas is None:
            self._areas = torch.segment_reduce(self.data.float(), "sum", offsets=self.offset, axis=0, initial=0)
        return self._areas

    def intersections(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(K,) pixels shared by masks a[k] and b[k], every pair in one pass.

        Each pair overlaps in a rectangle of its own, so all those rectangles' rows are laid end to end and
        gathered together; there is no work per pair.
        """
        a, b = a.to(self.device), b.to(self.device)
        ba, bb = self.boxes[a], self.boxes[b]
        lo, hi = torch.maximum(ba[:, :2], bb[:, :2]), torch.minimum(ba[:, 2:], bb[:, 2:])
        ca, ra, w, h = self._clip(a, lo, hi)
        if not len(a) or not int(h.sum()):
            return torch.zeros(len(a), device=self.device)
        cb, rb, _, _ = self._clip(b, lo, hi)
        item, row, col, valid = _rows(w, h)
        pa = self._window(a, item, row, col, ca, ra)
        pb = self._window(b, item, row, col, cb, rb)
        both = (pa & pb & valid).sum(dim=1)
        return torch.zeros(len(a), device=self.device).scatter_add_(0, item, both.float())

    def _clip(self, idx: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """The region between the (K, 2) corners ``lo`` and ``hi`` in crop idx[k]'s own coordinates: its first
        column and row, then its width and height, which are zero where the region misses the crop."""
        box = self.boxes[idx]
        start = (lo - box[:, :2]).clamp(min=0)
        wh = (torch.minimum(box[:, 2:], hi) - torch.maximum(box[:, :2], lo)).clamp(min=0)
        return start[:, 0], start[:, 1], wh[:, 0], wh[:, 1]

    def _window(
        self, idx: torch.Tensor, item: torch.Tensor, row: torch.Tensor, col: torch.Tensor, c0: torch.Tensor, r0: torch.Tensor
    ) -> torch.Tensor:
        """Rows of the window of crop idx[k] that starts at column c0[k], row r0[k], on the ``_rows`` layout
        ``item``/``row``/``col``.

        Columns past a window's own width read whatever is next in the buffer; the layout's ``valid`` masks them off.
        """
        stride = (self.boxes[idx, 2] - self.boxes[idx, 0])[item]
        at = self.offset[idx][item] + (row + r0[item]) * stride + c0[item]
        return self.data[(at[:, None] + col).clamp(0, self.data.numel() - 1)]

    def boxes_in_regions(self, idx: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
        """(K, 4) box around mask idx[k]'s pixels inside the region lo[k]-hi[k], or ``empty[k]`` where it has none.

        ``lo`` and ``hi`` are (K, 4) xyxy, read as ``lo[:, :2]`` for the region's low corner and ``hi[:, 2:]`` for
        its high one, so they can be the same tensors the box test uses. One pass over every mask at once.
        """
        idx, lo, hi = idx.to(self.device), lo.to(self.device), hi.to(self.device)
        empty = empty.to(self.device)
        x0, y0 = self.boxes[idx, 0], self.boxes[idx, 1]
        c0, r0, w, h = self._clip(idx, lo[:, :2].long(), hi[:, 2:].ceil().long())

        k, far = len(idx), max(self.size) + 1
        if not k or not int((w * h).sum()):  # no region has area; the column reductions below need a non-empty axis
            return empty
        item, row, col, valid = _rows(w, h)
        px = self._window(idx, item, row, col, c0, r0) & valid

        set_row = px.any(dim=1)
        first_r = torch.full((k,), far, device=self.device).scatter_reduce_(0, item, torch.where(set_row, row, far), reduce="amin")
        last_r = torch.full((k,), -1, device=self.device).scatter_reduce_(0, item, torch.where(set_row, row, -1), reduce="amax")
        # a column is set for a mask if any of its rows has it, so fold the rows down to one per mask first
        blank = torch.zeros(k, px.shape[1], dtype=torch.uint8, device=self.device)  # cuda scatter_reduce has no bool kernel
        set_col = blank.scatter_reduce_(0, item[:, None].expand_as(px), px.to(torch.uint8), reduce="amax").bool()
        span = torch.arange(px.shape[1], device=self.device)[None, :]
        first_c, last_c = torch.where(set_col, span, far).amin(dim=1), torch.where(set_col, span, -1).amax(dim=1)
        found = torch.stack([x0 + c0 + first_c, y0 + r0 + first_r, x0 + c0 + last_c + 1, y0 + r0 + last_r + 1], dim=1)
        return torch.where((last_r < 0)[:, None], empty, found.float())

    def intersection(self, a: int, b: int) -> float:
        """Pixels shared by masks a and b."""
        return float(self.intersections(torch.tensor([a]), torch.tensor([b]))[0])

    def union(self, members: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crop and (4,) box of masks ``members`` combined."""
        idx = torch.as_tensor(list(members), dtype=torch.long, device=self.device)
        data, boxes = self.unions(idx, torch.zeros(len(idx), dtype=torch.long, device=self.device), 1)
        box = boxes[0]
        return data.view(int(box[3] - box[1]), int(box[2] - box[0])), box

    def unions(self, members: torch.Tensor, group: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packed data and boxes of ``n`` masks, mask g being every member with ``group[k] == g`` combined.

        One scatter for every group at once: each member's rows are written into its group's crop, and where two
        members land on the same pixel the larger value wins, which for bool is the same as OR.
        """
        members, group = members.to(self.device), group.to(self.device)
        mb = self.boxes[members]
        live = (mb[:, 2] > mb[:, 0]) & (mb[:, 3] > mb[:, 1])
        members, group, mb = members[live], group[live], mb[live]
        far = max(self.size) + 1
        pair = group[:, None].expand(-1, 2)
        lo = torch.full((n, 2), far, dtype=torch.long, device=self.device).scatter_reduce_(0, pair, mb[:, :2], reduce="amin")
        hi = torch.zeros(n, 2, dtype=torch.long, device=self.device).scatter_reduce_(0, pair, mb[:, 2:], reduce="amax")
        boxes = torch.cat([lo, hi], dim=1)
        boxes[lo[:, 0] >= hi[:, 0]] = 0  # a group whose members are all empty
        sizes = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
        offset = _offsets(sizes)
        out = torch.zeros(int(sizes.sum()), dtype=torch.uint8, device=self.device)
        if not len(members) or not out.numel():
            return out.bool(), boxes

        wh = mb[:, 2:] - mb[:, :2]
        item, row, col = _cells(wh[:, 0], wh[:, 1])
        px = self.data[self.offset[members][item] + row * wh[item, 0] + col]
        g = group[item]
        at = offset[g] + (row + mb[item, 1] - boxes[g, 1]) * (boxes[:, 2] - boxes[:, 0])[g] + col + mb[item, 0] - boxes[g, 0]
        out.scatter_reduce_(0, at, px.to(torch.uint8), reduce="amax")  # cuda scatter_reduce has no bool kernel
        return out.bool(), boxes

    def set_runs(self, rows: torch.Tensor, data: torch.Tensor, boxes: torch.Tensor) -> None:
        """Replace masks ``rows`` with the crops packed end to end in ``data``, in one rebuild of the buffer."""
        at = rows.to(self.device)
        boxes = boxes.to(self.device)
        incoming = _offsets((boxes[:, 2:] - boxes[:, :2]).prod(dim=1))
        self.boxes[at] = boxes
        # the new crops go on the end of the old buffer, then one gather lays every crop out in order again
        source = self.offset[:-1].clone()
        source[at] = self.data.numel() + incoming[:-1]
        sizes = self.wh.prod(dim=1)
        item, pos = _runs(sizes)
        self.data = torch.cat([self.data, data.to(self.device)])[source[item] + pos]
        self.offset = _offsets(sizes)
        self._areas = None

    def set(self, k: int, crop: torch.Tensor, box: torch.Tensor) -> None:
        """Replace mask k. Prefer ``set_runs``: a crop can change size, so each call rebuilds the buffer."""
        self.set_runs(torch.tensor([int(k)]), crop.reshape(-1), box.reshape(1, 4))

    def paste(self) -> torch.Tensor:
        """(N, H, W) full-image masks of ``dtype``."""
        out = torch.zeros(len(self), *self.size, dtype=self.dtype, device=self.device)
        if not len(self) or not self.data.numel():
            return out
        flat = out.reshape(-1)
        flat[_image_index(self.boxes, self.size)] = self.data.to(self.dtype)
        return flat.view(len(self), *self.size)
