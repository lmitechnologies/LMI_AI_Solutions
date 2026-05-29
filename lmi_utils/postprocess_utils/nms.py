"""Class-aware NMS over a merged per-image result dict.

Result-dict convention (per-instance fields, all length N or absent):
    boxes:    (N, 4) xyxy  or  (N, 4, 2) OBB
    masks:    (N, H, W) binary instance masks
    segments: list of (M, 2) polygons
    scores:   (N,)
    classes:  (N,) tensor or np.ndarray

Generic and geometry-agnostic — no knowledge of how the instances were produced.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def class_aware_nms(merged: Dict[str, Any], iou_thr: float) -> Dict[str, Any]:
    """Greedy class-aware NMS over a merged result dict. No-op without usable scores/geometry.

    Keeps the highest-scoring instance among any group whose pairwise IoU exceeds
    ``iou_thr`` and that share a class. Suppression only — instances are dropped, never
    merged. Returns ``merged`` unchanged when fewer than two scores are present or no
    geometry field can yield an IoU.
    """
    scores = merged.get("scores")
    if not isinstance(scores, torch.Tensor) or len(scores) < 2:
        return merged
    iou = _instance_iou_matrix(merged)
    if iou is None or iou.shape[0] != len(scores):
        return merged
    keep = _greedy_nms(iou, scores, merged.get("classes"), iou_thr)
    if len(keep) == len(scores):
        return merged
    return _filter_instances(merged, keep)


def _instance_iou_matrix(merged: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Pairwise IoU (N, N) from the first usable geometry field: boxes > masks > segments.

    Returns None when no field can yield an IoU (e.g. points only). OBB and segment IoU
    use shapely (imported lazily so the dependency is only required when actually deduping
    polygon-typed results).
    """
    boxes = merged.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes):
        if boxes.ndim == 2:  # xyxy
            return _box_iou_matrix(boxes.float())
        if boxes.ndim == 3:  # OBB (N, 4, 2)
            return _polygon_iou_matrix([b.detach().cpu().numpy() for b in boxes])
    masks = merged.get("masks")
    if isinstance(masks, torch.Tensor) and len(masks):
        return _mask_iou_matrix(masks)
    segments = merged.get("segments")
    if segments is not None and len(segments):
        polys = [s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s) for s in segments]
        return _polygon_iou_matrix(polys)
    return None


def _box_iou_matrix(boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU (N, N) for xyxy boxes (N, 4)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    inter_w = (torch.minimum(x2[:, None], x2[None, :]) - torch.maximum(x1[:, None], x1[None, :])).clamp(min=0)
    inter_h = (torch.minimum(y2[:, None], y2[None, :]) - torch.maximum(y1[:, None], y1[None, :])).clamp(min=0)
    inter = inter_w * inter_h
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp(min=1e-9)


def _mask_iou_matrix(masks: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU (N, N) for binary instance masks (N, H, W)."""
    flat = (masks.reshape(masks.shape[0], -1) != 0).float()
    inter = flat @ flat.t()
    area = flat.sum(dim=1)
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp(min=1e-9)


def _polygon_iou_matrix(polys: List[np.ndarray]) -> torch.Tensor:
    """Pairwise IoU (N, N) for polygons (each (M, 2)); degenerate polys get zero IoU."""
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
    iou = torch.zeros(n, n)
    for i in range(n):
        gi = geoms[i]
        if gi is None:
            continue
        iou[i, i] = 1.0
        bi = bounds[i]
        for j in range(i + 1, n):
            gj = geoms[j]
            if gj is None:
                continue
            bj = bounds[j]
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue  # bounding boxes disjoint -> intersection (and IoU) is zero
            inter = gi.intersection(gj).area
            if inter <= 0:
                continue
            val = inter / (gi.area + gj.area - inter)
            iou[i, j] = iou[j, i] = val
    return iou


def _greedy_nms(iou: torch.Tensor, scores: torch.Tensor, classes: Any, iou_thr: float) -> torch.Tensor:
    """Greedy class-aware NMS. Returns kept instance indices (ascending) as a LongTensor."""
    n = len(scores)
    order = torch.argsort(scores.float(), descending=True).tolist()
    cls = None
    if classes is not None and len(classes) == n:
        cls = classes.tolist() if isinstance(classes, np.ndarray) else list(classes)
    iou_cpu = iou.detach().cpu()
    suppressed = [False] * n
    keep: List[int] = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if cls is not None and cls[i] != cls[j]:
                continue
            if float(iou_cpu[i, j]) > iou_thr:
                suppressed[j] = True
    keep.sort()
    return torch.tensor(keep, dtype=torch.long)


def _filter_instances(merged: Dict[str, Any], keep: torch.Tensor) -> Dict[str, Any]:
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
