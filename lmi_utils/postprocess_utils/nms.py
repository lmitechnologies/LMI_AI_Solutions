"""Class-aware NMS over a merged per-image result dict.

Result-dict convention (per-instance fields, all length N or absent):
    boxes:    (N, 4) xyxy  or  (N, 4, 2) OBB
    masks:    (N, H, W) binary instance masks, or ``MaskCrops``
    segments: list of (M, 2) polygons
    scores:   (N,)
    classes:  (N,) tensor or np.ndarray

Overlap is kept as the list of intersecting pairs, never an N x N matrix: one tiled image can carry tens of
thousands of detections, and a dense matrix of those costs tens of gigabytes while nearly every entry is zero.

Generic and geometry-agnostic — no knowledge of how the instances were produced.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .mask_crops import MaskCrops

_EPS = 1e-9
_PAIR_BUDGET = 4_000_000  # candidate pairs held at once while scanning for intersecting boxes

# (K, 2) index pairs i < j whose shapes intersect, (K,) intersection area, (N,) per-instance area
Overlap = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def class_aware_nms(
    merged: Dict[str, Any],
    iou_thr: Optional[float],
    containment_thr: Optional[float] = None,
) -> Dict[str, Any]:
    """Greedy class-aware NMS over a merged result dict. No-op without usable scores/geometry.

    Suppresses a lower-scoring instance of the same class when it either overlaps the kept one
    past ``iou_thr`` or lies inside it past ``containment_thr`` (the containment rule catches a
    fragment nested in a whole detection, which IoU alone misses). Instances that do not intersect
    never suppress each other, whatever the thresholds. Suppression only — instances are dropped,
    never merged. Returns ``merged`` unchanged when fewer than two scores are present, no geometry
    field can yield an overlap, or both thresholds are None.
    """
    if iou_thr is None and containment_thr is None:
        return merged
    scores = merged.get("scores")
    if not isinstance(scores, torch.Tensor) or len(scores) < 2:
        return merged
    overlap = pairwise_overlap(merged)
    if overlap is None or len(overlap[2]) != len(scores):
        return merged
    keep = _greedy_nms(overlap, scores, merged.get("classes"), iou_thr, containment_thr)
    if len(keep) == len(scores):
        return merged
    return filter_instances(merged, keep)


def pairwise_overlap(merged: Dict[str, Any]) -> Optional[Overlap]:
    """Intersecting pairs, their intersection area, and every instance's area, from the first usable
    geometry field: masks > segments > boxes.

    Masks and segments come first so containment is measured on the shape rather than its box —
    an object sitting in the hole of a ring is inside the ring's box but not inside the ring.
    Returns None when no field can yield an area (e.g. points only). OBB and segment geometry
    use shapely, imported lazily so the dependency is only needed for polygon-typed results.
    """
    masks = merged.get("masks")
    if isinstance(masks, (torch.Tensor, MaskCrops)) and len(masks):
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


def intersecting_pairs(boxes: torch.Tensor) -> torch.Tensor:
    """(K, 2) index pairs i < j whose xyxy boxes overlap with positive area.

    Sorted by left edge, a box can only meet the run of boxes that start before it ends, and one binary
    search finds where that run stops. Comparing every pair instead costs N squared, which is what ran a
    crowded tiled image out of memory and time. Runs on the device ``boxes`` is on.
    """
    n = len(boxes)
    dev = boxes.device
    empty = torch.zeros(0, 2, dtype=torch.long, device=dev)
    if n < 2:
        return empty
    order = torch.argsort(boxes[:, 0])
    b = boxes[order]
    left = b[:, 0].contiguous()
    stop = torch.searchsorted(left, b[:, 2].contiguous())  # first box starting at or after this one ends
    counts = (stop - torch.arange(1, n + 1, device=dev)).clamp(min=0)
    offsets = torch.cumsum(counts, 0) - counts

    parts = []
    for lo, hi in _row_blocks(counts, _PAIR_BUDGET):
        rows = torch.repeat_interleave(torch.arange(lo, hi, device=dev), counts[lo:hi])
        cols = rows + 1 + torch.arange(len(rows), device=dev) - (offsets[rows] - offsets[lo])
        hit = (torch.minimum(b[rows, 2], b[cols, 2]) > torch.maximum(b[rows, 0], b[cols, 0])) & (
            torch.minimum(b[rows, 3], b[cols, 3]) > torch.maximum(b[rows, 1], b[cols, 1])
        )
        a, c = order[rows[hit]], order[cols[hit]]
        parts.append(torch.stack([torch.minimum(a, c), torch.maximum(a, c)], dim=1))
    return torch.cat(parts) if parts else empty


def _row_blocks(counts: torch.Tensor, budget: int):
    """Row ranges whose candidate counts sum to at most ``budget``, so the pair list stays bounded."""
    total = torch.cumsum(counts, 0).cpu()  # block edges are Python ints, so read the running sums back once
    n = len(counts)
    lo = 0
    while lo < n:
        base = int(total[lo - 1]) if lo else 0
        hi = int(torch.searchsorted(total, torch.tensor(base + budget), right=True))
        hi = min(max(hi, lo + 1), n)
        yield lo, hi
        lo = hi


def _box_overlap(boxes: torch.Tensor) -> Overlap:
    """Intersecting pairs, their intersection area, and areas, for xyxy boxes (N, 4)."""
    area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    pairs = intersecting_pairs(boxes)
    a, b = boxes[pairs[:, 0]], boxes[pairs[:, 1]]
    w = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
    h = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
    return pairs, w * h, area


def _mask_overlap(masks: Union[torch.Tensor, MaskCrops]) -> Overlap:
    """Intersecting pairs, their intersection area, and areas, for binary instance masks, (N, H, W) or
    ``MaskCrops``, on the CPU.

    Only pairs whose mask boxes overlap are measured, and only where the boxes overlap: most pairs in a
    full-size image never touch, and comparing every pixel of every pair is what made tiled NMS slow.
    """
    crops = masks if isinstance(masks, MaskCrops) else MaskCrops.from_masks(masks, tuple(masks.shape[1:]))
    pairs = intersecting_pairs(crops.boxes.float())
    inter = torch.tensor([crops.intersection(i, j) for i, j in pairs.tolist()], dtype=torch.float32)
    return pairs, inter, crops.areas().clone()


def _polygon_overlap(polys: List[np.ndarray]) -> Overlap:
    """Intersecting pairs, their intersection area, and areas, for polygons (each (M, 2)); degenerate polys
    get zero area and intersect nothing."""
    from shapely.geometry import Polygon

    def _poly(coords: np.ndarray):
        if coords is None or len(coords) < 3:
            return None
        p = Polygon(coords)
        if not p.is_valid:
            p = p.buffer(0)
        return p if (not p.is_empty and p.area > 0) else None

    geoms = [_poly(c) for c in polys]
    area = torch.tensor([g.area if g is not None else 0.0 for g in geoms], dtype=torch.float32)
    bounds = torch.tensor([g.bounds if g is not None else (0.0, 0.0, 0.0, 0.0) for g in geoms], dtype=torch.float32)

    pairs, values = [], []
    for i, j in intersecting_pairs(bounds).tolist():
        val = geoms[i].intersection(geoms[j]).area
        if val > 0:
            pairs.append([i, j])
            values.append(val)
    return (
        torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
        torch.tensor(values, dtype=torch.float32),
        area,
    )


def class_codes(classes: Any, n: int) -> Optional[torch.Tensor]:
    """Map arbitrary class labels to dense integer codes (N,). None when unusable."""
    if classes is None or len(classes) != n:
        return None
    values = classes.tolist() if isinstance(classes, (np.ndarray, torch.Tensor)) else list(classes)
    lookup: Dict[Any, int] = {}
    return torch.tensor([lookup.setdefault(v, len(lookup)) for v in values], dtype=torch.long)


def _suppression_edges(
    overlap: Overlap,
    classes: Any,
    n: int,
    iou_thr: Optional[float],
    containment_thr: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(E,) source and (E,) target: keeping ``source[e]`` drops ``target[e]``.

    Both directions of each intersecting pair are tested, since containment is asymmetric.
    """
    pairs, inter, area = overlap
    a, b = pairs[:, 0], pairs[:, 1]
    area_a, area_b = area[a], area[b]
    drops_b = torch.zeros(len(pairs), dtype=torch.bool)  # a suppresses b
    drops_a = torch.zeros(len(pairs), dtype=torch.bool)
    if iou_thr is not None:
        both = inter / (area_a + area_b - inter).clamp(min=_EPS) > iou_thr
        drops_b |= both
        drops_a |= both
    if containment_thr is not None:
        drops_b |= inter / area_b.clamp(min=_EPS) >= containment_thr
        drops_a |= inter / area_a.clamp(min=_EPS) >= containment_thr
    codes = class_codes(classes, n)
    if codes is not None:
        same = codes[a] == codes[b]
        drops_b &= same
        drops_a &= same
    return torch.cat([a[drops_b], b[drops_a]]), torch.cat([b[drops_b], a[drops_a]])


def _greedy_nms(
    overlap: Overlap,
    scores: torch.Tensor,
    classes: Any,
    iou_thr: Optional[float],
    containment_thr: Optional[float],
) -> torch.Tensor:
    """Greedy class-aware NMS. Returns kept indices, ascending."""
    n = len(scores)
    overlap = tuple(t.detach().cpu() for t in overlap)  # the greedy loop below is sequential, so it stays on the CPU
    source, target = _suppression_edges(overlap, classes, n, iou_thr, containment_thr)
    order = torch.argsort(source)
    source, target = source[order], target[order]
    # each instance's targets are one slice of the sorted edge list
    bounds = torch.searchsorted(source, torch.arange(n + 1)).numpy()
    target = target.numpy()  # the loop below runs per instance, and numpy indexing is far cheaper than torch

    linked = np.zeros(n, dtype=bool)
    linked[source.numpy()] = True
    linked[target] = True
    ranked = torch.argsort(scores.float().cpu(), descending=True).numpy()

    alive = np.ones(n, dtype=bool)
    keep: List[int] = np.flatnonzero(~linked).tolist()  # no edge, so neither suppressed nor suppressing
    for i in ranked[linked[ranked]].tolist():
        if not alive[i]:
            continue
        keep.append(i)
        alive[target[bounds[i] : bounds[i + 1]]] = False
        alive[i] = False  # a kept row must not be suppressed by a later one: containment is asymmetric
    keep.sort()
    return torch.tensor(keep, dtype=torch.long)


def filter_instances(merged: Dict[str, Any], keep: torch.Tensor) -> Dict[str, Any]:
    """Index every per-instance field of a merged result by ``keep`` (a CPU LongTensor)."""
    out = dict(merged)
    for key in ("boxes", "scores", "points", "masks", "merge_origin"):
        v = merged.get(key)
        if isinstance(v, (torch.Tensor, MaskCrops)) and len(v):
            out[key] = v[keep]  # CPU index into a CUDA tensor is allowed
    classes = merged.get("classes")
    if classes is not None and len(classes):
        out["classes"] = classes[keep.numpy()] if isinstance(classes, np.ndarray) else classes[keep]
    segments = merged.get("segments")
    if segments is not None and len(segments):
        out["segments"] = [segments[i] for i in keep.tolist()]
    return out
