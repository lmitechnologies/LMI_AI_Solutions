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


_OVERLAP_BLOCK_BYTES = 256 << 20  # float32 operand budget per matmul block


def class_aware_nms(merged: Dict[str, Any], iou_thr: Optional[float], containment_thr: Optional[float] = None) -> Dict[str, Any]:
    """Greedy class-aware NMS over a merged result dict. No-op without usable scores/geometry.

    Suppresses a lower-scoring instance of the same class when it either overlaps the kept one
    past ``iou_thr`` or lies inside it past ``containment_thr`` (the containment rule catches a
    fragment nested in a whole detection, which IoU alone misses). Suppression only — instances
    are dropped, never merged. Returns ``merged`` unchanged when fewer than two scores are
    present, no geometry field can yield an overlap, or both thresholds are None.
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
    return filter_instances(merged, keep)


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


def _mask_overlap(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intersection matrix and areas for binary instance masks (N, H, W).

    Blocked over both axes: the dense (N, H*W) float operand the matmul wants is gigabytes once a
    tiled batch and a full-size image meet. Same dot products, so the result is unchanged.
    """
    flat = binarize_masks(masks).flatten(1)  # flatten, not reshape: reshape(0, -1) is ambiguous and raises
    n, hw = flat.shape
    inter = torch.zeros(n, n, dtype=torch.float32, device=flat.device)
    block = max(1, min(n, int(_OVERLAP_BLOCK_BYTES // max(hw * 4, 1))))
    for i in range(0, n, block):
        rows = flat[i : i + block].float()
        for j in range(i, n, block):
            v = rows @ flat[j : j + block].float().t()
            inter[i : i + block, j : j + block] = v
            if j > i:
                inter[j : j + block, i : i + block] = v.t()
    # areas off the diagonal: torch promotes a bool .sum() to int64 across the whole array first
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


def filter_instances(merged: Dict[str, Any], keep: torch.Tensor) -> Dict[str, Any]:
    """Index every per-instance field of a merged result by ``keep`` (a CPU LongTensor)."""
    out = dict(merged)
    for key in ("boxes", "scores", "points", "masks"):
        v = merged.get(key)
        if isinstance(v, torch.Tensor) and len(v):
            out[key] = v[keep]  # CPU index into a CUDA tensor is allowed
    classes = merged.get("classes")
    if classes is not None and len(classes):
        out["classes"] = classes[keep.numpy()] if isinstance(classes, np.ndarray) else classes[keep]
    segments = merged.get("segments")
    if segments is not None and len(segments):
        out["segments"] = [segments[i] for i in keep.tolist()]
    return out
