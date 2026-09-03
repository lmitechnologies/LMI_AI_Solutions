"""Reassemble detections that a tiled inference pass split across tile seams.

A prediction touching an *interior* tile edge is a fragment: the object continued into the
neighbouring tile and that tile holds the rest of it. This module flags those fragments, pairs
them across seams, and unions each connected group into one instance. Union-find over the pairs
lets a group span any number of tiles, so there is no limit on object size.

Runs on a merged per-image result dict already in image coordinates (see ``nms``), plus the tile
each instance came from. Boxes, masks and segments are supported; keypoints and OBB are not.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .nms import class_codes, containment_matrix, filter_instances, pairwise_overlap

# Fixed, not exposed: these reflect detector and geometry behaviour, not the dataset.
DEFAULT_EDGE_TOLERANCE = 2.0  # px from a tile edge that still counts as touching it
PAIR_OVERLAP_RATIO = 0.5  # of the smaller perpendicular extent; guards against diagonal chaining
SCORE_FLOOR = 0.01  # keeps junk out of merge and NMS
SIMPLIFY_TOLERANCE = 1.0  # px, when a merged polygon is traced back from a union


def merge_tile_fragments(
    merged: Dict[str, Any],
    tile_idx: torch.Tensor,
    tile_rc: np.ndarray,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    im_size: Tuple[int, int],
    containment: float,
    edge_tolerance: float = DEFAULT_EDGE_TOLERANCE,
) -> Dict[str, Any]:
    """Union fragments of one object that were detected separately in adjacent tiles.

    Args:
        merged: per-image result dict, instances already offset into image coordinates.
        tile_idx: (N,) tile index each instance came from.
        tile_rc: (T, 2) grid row/col per tile index.
        tile_origins: (T, 2) tile top-left (y, x) in image coordinates.
        tile_size: (tile_h, tile_w).
        im_size: (im_h, im_w) of the original, unpadded image.
        containment: fraction of a fragment that must sit inside an untruncated prediction for
            that prediction to explain it (and so clear its truncation flags).
        edge_tolerance: px from a tile edge that still counts as touching it. Reflects the
            detector's box-regression error at a crop boundary, which is set by the detection
            head's feature stride, so it does not scale with tile size.

    Returns the dict with each fragment group replaced by a single unioned instance. Scores are
    the group maximum, since fragment scores are consistently low and averaging would penalise an
    object for spanning more tiles.
    """
    n = len(tile_idx)
    if n < 2:
        return merged

    merged, tile_idx = _drop_below_floor(merged, tile_idx)
    if len(tile_idx) < 2:
        return merged

    boxes = instance_boxes(merged)
    if boxes is None:
        return merged

    codes = class_codes(merged.get("classes"), len(tile_idx))
    flags = _truncation_flags(boxes, tile_idx, tile_origins, tile_size, im_size, edge_tolerance)
    flags = _clear_explained(merged, flags, tile_idx, tile_origins, tile_size, codes, containment)

    pairs = _pair_fragments(boxes, flags, tile_idx, tile_rc, codes)
    groups = _connected_groups(pairs, len(tile_idx))
    if all(len(g) == 1 for g in groups):
        return merged
    return _union_groups(merged, groups)


def _drop_below_floor(merged: Dict[str, Any], tile_idx: torch.Tensor) -> Tuple[Dict[str, Any], torch.Tensor]:
    scores = merged.get("scores")
    if not isinstance(scores, torch.Tensor) or len(scores) != len(tile_idx):
        return merged, tile_idx
    keep = (scores.detach().cpu().float() >= SCORE_FLOOR).nonzero(as_tuple=True)[0]
    if len(keep) == len(tile_idx):
        return merged, tile_idx
    return filter_instances(merged, keep), tile_idx[keep]


def instance_boxes(merged: Dict[str, Any]) -> Optional[torch.Tensor]:
    """(N, 4) xyxy image-space extents, from boxes, else masks, else segments."""
    boxes = merged.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes):
        b = boxes.detach().cpu().float()
        if b.ndim == 3:  # OBB (N, 4, 2) -> enclosing aabb
            return torch.cat([b.amin(dim=1), b.amax(dim=1)], dim=1)
        return b

    masks = merged.get("masks")
    if isinstance(masks, torch.Tensor) and len(masks):
        from torchvision.ops import masks_to_boxes

        return masks_to_boxes((masks != 0).cpu()).float()

    segments = merged.get("segments")
    if segments is not None and len(segments):
        out = torch.zeros(len(segments), 4)
        for i, s in enumerate(segments):
            pts = s.detach().cpu().float() if isinstance(s, torch.Tensor) else torch.as_tensor(np.asarray(s), dtype=torch.float32)
            if len(pts):
                out[i] = torch.cat([pts.amin(dim=0), pts.amax(dim=0)])
        return out
    return None


def _truncation_flags(
    boxes: torch.Tensor,
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    im_size: Tuple[int, int],
    tolerance: float,
) -> torch.Tensor:
    """(N, 4) bool [left, top, right, bottom]: instance sits on that *interior* tile edge.

    Image borders and edges out in the padded region do not count — nothing continues past them.
    """
    tile_h, tile_w = tile_size
    im_h, im_w = im_size
    origins = torch.as_tensor(tile_origins, dtype=torch.float32)[tile_idx]
    y0, x0 = origins[:, 0], origins[:, 1]
    y1, x1 = y0 + tile_h, x0 + tile_w

    return torch.stack(
        [
            (x0 > 0) & (boxes[:, 0] <= x0 + tolerance),
            (y0 > 0) & (boxes[:, 1] <= y0 + tolerance),
            (x1 < im_w) & (boxes[:, 2] >= x1 - tolerance),
            (y1 < im_h) & (boxes[:, 3] >= y1 - tolerance),
        ],
        dim=1,
    )


def _tile_overlap_matrix(tile_origins: np.ndarray, tile_size: Tuple[int, int]) -> torch.Tensor:
    """(T, T) bool: which tile rectangles intersect."""
    tile_h, tile_w = tile_size
    o = torch.as_tensor(tile_origins, dtype=torch.float32)
    y0, x0 = o[:, 0], o[:, 1]
    y1, x1 = y0 + tile_h, x0 + tile_w
    inter_h = torch.minimum(y1[:, None], y1[None, :]) - torch.maximum(y0[:, None], y0[None, :])
    inter_w = torch.minimum(x1[:, None], x1[None, :]) - torch.maximum(x0[:, None], x0[None, :])
    return (inter_h > 0) & (inter_w > 0)


def _clear_explained(
    merged: Dict[str, Any],
    flags: torch.Tensor,
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    codes: Optional[torch.Tensor],
    containment: float,
) -> torch.Tensor:
    """Drop the truncation flags of any fragment an untruncated prediction already explains.

    With overlap the whole object is usually also detected outright by a neighbouring tile.
    Left flagged, its fragments still merge behind it and the merged box beats the correct
    detection at containment NMS.
    """
    if not flags.any():
        return flags
    overlap = pairwise_overlap(merged)
    if overlap is None or overlap[0].shape[0] != len(tile_idx):
        return flags

    contained = containment_matrix(*(t.detach().cpu() for t in overlap)) >= containment  # [j, i]: i inside j
    explains = contained & (~flags.any(dim=1))[:, None] & (tile_idx[:, None] != tile_idx[None, :])
    explains &= _tile_overlap_matrix(tile_origins, tile_size)[tile_idx][:, tile_idx]
    if codes is not None:
        explains &= codes[:, None] == codes[None, :]

    flags = flags.clone()
    flags[explains.any(dim=0)] = False
    return flags


def _pair_fragments(
    boxes: torch.Tensor,
    flags: torch.Tensor,
    tile_idx: torch.Tensor,
    tile_rc: np.ndarray,
    codes: Optional[torch.Tensor],
) -> torch.Tensor:
    """(N, N) symmetric bool: fragments that are two halves of one object cut by a seam."""
    rc = torch.as_tensor(tile_rc, dtype=torch.long)[tile_idx]
    row, col = rc[:, 0], rc[:, 1]
    left, top, right, bottom = (flags[:, k] for k in range(4))

    # j is i's right neighbour: i must be cut on its right edge and j on its left.
    horizontal = (row[:, None] == row[None, :]) & (col[None, :] == col[:, None] + 1)
    horizontal &= right[:, None] & left[None, :]
    horizontal &= _extent_agrees(boxes[:, 1], boxes[:, 3])

    vertical = (col[:, None] == col[None, :]) & (row[None, :] == row[:, None] + 1)
    vertical &= bottom[:, None] & top[None, :]
    vertical &= _extent_agrees(boxes[:, 0], boxes[:, 2])

    pairs = horizontal | vertical
    if codes is not None:
        pairs &= codes[:, None] == codes[None, :]
    return pairs | pairs.t()


def _extent_agrees(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """(N, N) bool: extents perpendicular to the seam line up well enough to be one object.

    Without this two objects stacked along a seam pair diagonally and union-find chains them.
    """
    overlap = torch.minimum(hi[:, None], hi[None, :]) - torch.maximum(lo[:, None], lo[None, :])
    extent = hi - lo
    return overlap > PAIR_OVERLAP_RATIO * torch.minimum(extent[:, None], extent[None, :])


def _connected_groups(pairs: torch.Tensor, n: int) -> List[List[int]]:
    """Union-find over the pair matrix; returns member index lists, one per group."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs.triu(diagonal=1).nonzero().tolist():
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    return list(buckets.values())


def _union_groups(merged: Dict[str, Any], groups: List[List[int]]) -> Dict[str, Any]:
    """Collapse each group to its highest-scoring member, then widen that member's geometry."""
    scores = merged.get("scores")
    reps: List[int] = []
    for g in groups:
        if len(g) == 1 or not isinstance(scores, torch.Tensor) or len(scores) <= max(g):
            reps.append(g[0])
        else:
            reps.append(max(g, key=lambda i: float(scores[i])))
    order = sorted(range(len(groups)), key=lambda k: reps[k])
    groups = [groups[k] for k in order]
    reps = [reps[k] for k in order]

    out = filter_instances(merged, torch.tensor(reps, dtype=torch.long))

    boxes = merged.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes):
        stacked = out["boxes"].clone()
        for k, g in enumerate(groups):
            members = boxes[torch.tensor(g, dtype=torch.long, device=boxes.device)]
            stacked[k, :2] = members[:, :2].amin(dim=0)
            stacked[k, 2:] = members[:, 2:].amax(dim=0)
        out["boxes"] = stacked

    masks = merged.get("masks")
    if isinstance(masks, torch.Tensor) and len(masks):
        unioned = out["masks"].clone()
        for k, g in enumerate(groups):
            if len(g) > 1:
                idx = torch.tensor(g, dtype=torch.long, device=masks.device)
                unioned[k] = (masks[idx] != 0).any(dim=0).to(masks.dtype)
        out["masks"] = unioned

    segments = merged.get("segments")
    if segments is not None and len(segments):
        out["segments"] = [_union_polygons([segments[i] for i in g]) if len(g) > 1 else segments[g[0]] for g in groups]

    return out


def _union_polygons(polys: List[Any]) -> torch.Tensor:
    """Outer contour of a polygon union. A hole cannot survive: a segment is a single ring."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    geoms = []
    for p in polys:
        coords = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
        if len(coords) < 3:
            continue
        g = Polygon(coords)
        if not g.is_valid:
            g = g.buffer(0)
        if not g.is_empty and g.area > 0:
            geoms.append(g)
    if not geoms:
        return polys[0]

    union = unary_union(geoms)
    if isinstance(union, MultiPolygon):  # seams that did not actually touch
        union = max(union.geoms, key=lambda g: g.area)
    ring = np.asarray(union.simplify(SIMPLIFY_TOLERANCE).exterior.coords[:-1], dtype=np.float32)

    ref = polys[0]
    if isinstance(ref, torch.Tensor):
        return torch.as_tensor(ring, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(ring)
