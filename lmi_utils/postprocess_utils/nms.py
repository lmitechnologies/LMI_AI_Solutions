"""Class-aware NMS over a merged per-image result dict.

Result-dict convention (per-instance fields, all length N or absent):
    boxes:    (N, 4) xyxy  or  (N, 4, 2) OBB
    masks:    (N, H, W) binary instance masks
    segments: list of (M, 2) polygons
    scores:   (N,)
    classes:  (N,) tensor or np.ndarray

Generic and geometry-agnostic — no knowledge of how the instances were produced.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_EPS = 1e-9


def class_aware_nms(
    merged: Dict[str, Any], iou_thr: Optional[float], containment_thr: Optional[float] = None, in_place: bool = False
) -> Dict[str, Any]:
    """Greedy class-aware NMS over a merged result dict. No-op without usable scores/geometry.

    Suppresses a lower-scoring instance of the same class when it either overlaps the kept one
    past ``iou_thr`` or lies inside it past ``containment_thr`` (the containment rule catches a
    fragment nested in a whole detection, which IoU alone misses). Suppression only — instances
    are dropped, never merged. Returns ``merged`` unchanged when fewer than two scores are
    present, no geometry field can yield an overlap, or both thresholds are None.
    ``in_place``: see ``filter_instances``.
    """
    if iou_thr is None and containment_thr is None:
        return merged
    scores = merged.get("scores")
    if not isinstance(scores, torch.Tensor) or len(scores) < 2:
        return merged
    overlap = pairwise_overlap(merged)
    if overlap is None or overlap[0].shape[0] != len(scores):
        return merged
    keep = _greedy_nms(overlap, scores, merged.get("classes"), iou_thr, containment_thr)
    if len(keep) == len(scores):
        return merged
    return filter_instances(merged, keep, in_place)


def pairwise_overlap(merged: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Pairwise intersection area (N, N) and per-instance area (N,), from the first usable
    geometry field: masks > segments > boxes.

    Masks and segments come first so containment is measured on the shape rather than its box —
    an object sitting in the hole of a ring is inside the ring's box but not inside the ring.
    Returns None when no field can yield an area (e.g. points only). OBB and segment geometry
    use shapely, imported lazily so the dependency is only needed for polygon-typed results.
    """
    masks = merged.get("masks")
    if isinstance(masks, torch.Tensor) and len(masks):
        return _mask_overlap(masks)
    segments = merged.get("segments")
    if segments is not None and len(segments):
        polys = [s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s) for s in segments]
        return _polygon_overlap(polys)
    boxes = merged.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes):
        if boxes.ndim == 2:  # xyxy
            return _box_overlap(boxes.float())
        if boxes.ndim == 3:  # OBB (N, 4, 2)
            return _polygon_overlap([b.detach().cpu().numpy() for b in boxes])
    return None


def iou_matrix(inter: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """IoU (N, N) from an intersection matrix and an area vector."""
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp(min=_EPS)


def containment_matrix(inter: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """``C[i, j]`` = fraction of instance j that lies inside instance i."""
    return inter / area[None, :].clamp(min=_EPS)


def _box_overlap(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intersection matrix and areas for xyxy boxes (N, 4)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    inter_w = (torch.minimum(x2[:, None], x2[None, :]) - torch.maximum(x1[:, None], x1[None, :])).clamp(min=0)
    inter_h = (torch.minimum(y2[:, None], y2[None, :]) - torch.maximum(y1[:, None], y1[None, :])).clamp(min=0)
    return inter_w * inter_h, area


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


def _mask_overlap(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intersection matrix and areas for binary instance masks (N, H, W), on the CPU.

    Only pairs whose mask boxes overlap are measured, and only inside the first mask's box: most pairs in a
    full-size image never touch, and comparing every pixel of every pair is what made tiled NMS slow.
    """
    m = binarize_masks(masks)
    n = len(m)
    b = boxes_from_masks(m).long().cpu()
    touch = (torch.minimum(b[:, None, 2], b[None, :, 2]) > torch.maximum(b[:, None, 0], b[None, :, 0])) & (
        torch.minimum(b[:, None, 3], b[None, :, 3]) > torch.maximum(b[:, None, 1], b[None, :, 1])
    )
    inter = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        js = touch[i, i:].nonzero(as_tuple=True)[0] + i  # an empty mask touches nothing, itself included
        if not len(js):
            continue
        x0, y0, x1, y1 = b[i].tolist()
        v = (m[js, y0:y1, x0:x1] & m[i, y0:y1, x0:x1]).flatten(1).sum(dim=1).float().cpu()
        inter[i, js] = v
        inter[js, i] = v
    return inter, inter.diagonal().clone()


def _polygon_overlap(polys: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intersection matrix and areas for polygons (each (M, 2)); degenerate polys get zero area."""
    from shapely.geometry import Polygon

    def _poly(coords: np.ndarray):
        if coords is None or len(coords) < 3:
            return None
        p = Polygon(coords)
        if not p.is_valid:
            p = p.buffer(0)
        return p if (not p.is_empty and p.area > 0) else None

    geoms = [_poly(c) for c in polys]
    bounds = [g.bounds if g is not None else None for g in geoms]  # (minx, miny, maxx, maxy)
    n = len(geoms)
    inter = torch.zeros(n, n)
    area = torch.tensor([g.area if g is not None else 0.0 for g in geoms], dtype=torch.float32)
    for i in range(n):
        gi = geoms[i]
        if gi is None:
            continue
        inter[i, i] = area[i]
        bi = bounds[i]
        for j in range(i + 1, n):
            gj = geoms[j]
            if gj is None:
                continue
            bj = bounds[j]
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue  # bounding boxes disjoint -> intersection is zero
            val = gi.intersection(gj).area
            if val > 0:
                inter[i, j] = inter[j, i] = val
    return inter, area


def class_codes(classes: Any, n: int) -> Optional[torch.Tensor]:
    """Map arbitrary class labels to dense integer codes (N,). None when unusable."""
    if classes is None or len(classes) != n:
        return None
    values = classes.tolist() if isinstance(classes, (np.ndarray, torch.Tensor)) else list(classes)
    lookup: Dict[Any, int] = {}
    return torch.tensor([lookup.setdefault(v, len(lookup)) for v in values], dtype=torch.long)


def _greedy_nms(
    overlap: Tuple[torch.Tensor, torch.Tensor],
    scores: torch.Tensor,
    classes: Any,
    iou_thr: Optional[float],
    containment_thr: Optional[float],
) -> torch.Tensor:
    """Greedy class-aware NMS. Returns kept instance indices (ascending) as a LongTensor."""
    inter, area = (t.detach().cpu() for t in overlap)
    n = len(scores)

    suppresses = torch.zeros(n, n, dtype=torch.bool)  # suppresses[i, j]: keeping i drops j
    if iou_thr is not None:
        suppresses |= iou_matrix(inter, area) > iou_thr
    if containment_thr is not None:
        suppresses |= containment_matrix(inter, area) >= containment_thr
    codes = class_codes(classes, n)
    if codes is not None:
        suppresses &= codes[:, None] == codes[None, :]
    suppresses.fill_diagonal_(False)

    alive = torch.ones(n, dtype=torch.bool)
    keep: List[int] = []
    for i in torch.argsort(scores.float().cpu(), descending=True).tolist():
        if not alive[i]:
            continue
        keep.append(i)
        alive &= ~suppresses[i]
    keep.sort()
    return torch.tensor(keep, dtype=torch.long)


def filter_instances(merged: Dict[str, Any], keep: torch.Tensor, in_place: bool = False) -> Dict[str, Any]:
    """Index every per-instance field of a merged result by ``keep`` (a CPU LongTensor).

    ``in_place`` moves the kept mask rows to the front of the existing masks tensor and returns a view of them,
    so no second full-size copy is allocated. It overwrites the caller's masks and needs ascending ``keep``.
    """
    out = dict(merged)
    for key in ("boxes", "scores", "points", "masks"):
        v = merged.get(key)
        if isinstance(v, torch.Tensor) and len(v):
            # CPU index into a CUDA tensor is allowed
            out[key] = _compact_rows(v, keep) if in_place and key == "masks" else v[keep]
    classes = merged.get("classes")
    if classes is not None and len(classes):
        out["classes"] = classes[keep.numpy()] if isinstance(classes, np.ndarray) else classes[keep]
    segments = merged.get("segments")
    if segments is not None and len(segments):
        out["segments"] = [segments[i] for i in keep.tolist()]
    return out


def _compact_rows(t: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Move rows ``keep`` to the front of ``t`` and return that view. Ascending ``keep`` only moves a row to an
    earlier slot, so no row is overwritten before it is read."""
    rows = keep.tolist()
    if any(b <= a for a, b in zip(rows, rows[1:])):
        raise ValueError("filter_instances: in_place needs strictly ascending indices")
    for new, old in enumerate(rows):
        if new != old:
            t[new] = t[old]
    return t[: len(rows)]
