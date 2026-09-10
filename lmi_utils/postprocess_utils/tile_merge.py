"""Rebuild objects that tiled inference split across tile seams.

A prediction touching an interior tile edge is a fragment: its object continues into a neighbouring tile.
Each fragment is linked to the predictions of the same object in other tiles, and every linked group
becomes one instance. A group can span any number of tiles.

Input is one image's result dict, already in image coordinates, plus the tile each instance came from.
Supports boxes, masks and segments; not keypoints or OBB.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .nms import binarize_masks, class_codes, filter_instances, pairwise_overlap

# Fixed: they depend on the detector and tile geometry, not the dataset.
DEFAULT_EDGE_TOLERANCE = 2.0  # px from a tile edge that still counts as touching it
SAME_OBJECT_CONTAINMENT = 0.95  # two uncut predictions this far inside each other are one object
AGREEMENT_IOU = 0.5  # min IoU of two linked boxes inside the region both tiles see
SCORE_FLOOR = 0.01  # predictions scoring lower are ignored
SIMPLIFY_TOLERANCE = 1.0  # px, polygon simplification after a union


def merge_tile_fragments(
    merged: Dict[str, Any],
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    im_size: Tuple[int, int],
    containment: float,
    edge_tolerance: float = DEFAULT_EDGE_TOLERANCE,
) -> Dict[str, Any]:
    """Merge the pieces of each object that overlapping tiles detected separately.

    Args:
        merged: one image's result dict, in image coordinates.
        tile_idx: (N,) tile index of each instance.
        tile_origins: (T, 2) top-left (y, x) of each tile in the image.
        tile_size: (tile_h, tile_w).
        im_size: (im_h, im_w) of the original image, without padding.
        containment: min fraction of a fragment that must lie inside an uncut prediction for the two to link.
            A fragment-only output lying this far inside another output is dropped.
        edge_tolerance: px from a tile edge that still counts as touching it. Matches the detector's box error
            at a crop edge, so it does not scale with tile size.

    Returns:
        The dict with each linked group replaced by its uncut members, one per distinct object. A group with
        no uncut member becomes the union of its fragments, scored by its best fragment.
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
    cut = _cut_flags(boxes, tile_idx, tile_origins, tile_size, im_size, edge_tolerance)
    links = _links(merged, boxes, cut, tile_idx, tile_origins, tile_size, codes, containment)
    whole = ~cut
    groups = _split_distinct_whole(merged, boxes, _connected_groups(links | links.t(), len(tile_idx)), whole)
    out = merged
    if any(len(g) > 1 for g in groups):
        out, groups = _union_groups(merged, groups, whole)
    fragment_only = torch.tensor([not bool(whole[torch.tensor(g)].any()) for g in groups])
    return _absorb_fragments(out, fragment_only, containment)


def _drop_below_floor(merged: Dict[str, Any], tile_idx: torch.Tensor) -> Tuple[Dict[str, Any], torch.Tensor]:
    scores = merged.get("scores")
    if not isinstance(scores, torch.Tensor) or len(scores) != len(tile_idx):
        return merged, tile_idx
    keep = (scores.detach().cpu().float() >= SCORE_FLOOR).nonzero(as_tuple=True)[0]
    if len(keep) == len(tile_idx):
        return merged, tile_idx
    return filter_instances(merged, keep), tile_idx[keep]


def instance_boxes(merged: Dict[str, Any]) -> Optional[torch.Tensor]:
    """(N, 4) xyxy box of each instance, taken from boxes, else masks, else segments."""
    boxes = merged.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes):
        b = boxes.detach().cpu().float()
        if b.ndim == 3:  # OBB (N, 4, 2): use its enclosing box
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


def _cut_flags(
    boxes: torch.Tensor,
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    im_size: Tuple[int, int],
    tolerance: float,
) -> torch.Tensor:
    """(N,) bool: instance touches an interior tile edge, so it is a fragment.

    Tile edges on the image border or in the padding do not count: nothing continues past them.
    """
    tile_h, tile_w = tile_size
    im_h, im_w = im_size
    origins = torch.as_tensor(tile_origins, dtype=torch.float32)[tile_idx]
    y0, x0 = origins[:, 0], origins[:, 1]
    y1, x1 = y0 + tile_h, x0 + tile_w
    return (
        ((x0 > 0) & (boxes[:, 0] <= x0 + tolerance))
        | ((y0 > 0) & (boxes[:, 1] <= y0 + tolerance))
        | ((x1 < im_w) & (boxes[:, 2] >= x1 - tolerance))
        | ((y1 < im_h) & (boxes[:, 3] >= y1 - tolerance))
    )


def _links(
    merged: Dict[str, Any],
    boxes: torch.Tensor,
    cut: torch.Tensor,
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    codes: Optional[torch.Tensor],
    containment: float,
) -> torch.Tensor:
    """(N, N) bool [j, i]: fragment i and prediction j show the same object. All must hold:

    1. i is cut at a tile edge.
    2. j comes from another tile and has the same class.
    3. Their boxes agree inside the region both tiles see.
    4. j is cut too, or holds at least ``containment`` of i. Without this, a piece that stops just short of
       the seam looks whole and would claim its longer other half.
    """
    n = len(cut)
    links = torch.zeros(n, n, dtype=torch.bool)
    frag = cut.nonzero(as_tuple=True)[0]
    if not len(frag):
        return links
    cand = _touching(boxes, frag) & (tile_idx[:, None] != tile_idx[frag][None, :])
    if codes is not None:
        cand &= codes[:, None] == codes[frag][None, :]
    j, col = cand.nonzero(as_tuple=True)
    i = frag[col]
    keep = _agree_in_shared_region(boxes, j, i, tile_idx, tile_origins, tile_size)
    j, i = j[keep], i[keep]
    uncut = ~cut[j]
    if uncut.any():
        keep = torch.ones(len(j), dtype=torch.bool)
        keep[uncut] = _containment(merged, boxes, j[uncut], i[uncut]) >= containment
        j, i = j[keep], i[keep]
    links[j, i] = True
    return links


def _touching(boxes: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """(N, len(cols)) bool: box k and box cols[c] overlap with positive area."""
    b = boxes[cols]
    return (torch.minimum(boxes[:, None, 2], b[None, :, 2]) > torch.maximum(boxes[:, None, 0], b[None, :, 0])) & (
        torch.minimum(boxes[:, None, 3], b[None, :, 3]) > torch.maximum(boxes[:, None, 1], b[None, :, 1])
    )


def _agree_in_shared_region(
    boxes: torch.Tensor,
    j: torch.Tensor,
    i: torch.Tensor,
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
) -> torch.Tensor:
    """(K,) bool: boxes j[k] and i[k], cut down to the region both their tiles see, reach ``AGREEMENT_IOU``.

    Both tiles see that region, so two views of one object nearly match there. A larger neighbour that
    merely contains the fragment does not.
    """
    if not len(j):
        return torch.zeros(0, dtype=torch.bool)
    tile_h, tile_w = tile_size
    origins = torch.as_tensor(tile_origins, dtype=torch.float32)[tile_idx].flip(1)  # (x, y)
    lo = torch.maximum(origins[j], origins[i]).repeat(1, 2)
    hi = (torch.minimum(origins[j], origins[i]) + torch.tensor([tile_w, tile_h], dtype=torch.float32)).repeat(1, 2)
    a = torch.minimum(torch.maximum(boxes[j], lo), hi)
    b = torch.minimum(torch.maximum(boxes[i], lo), hi)
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp(min=0).prod(dim=1)
    area_a = (a[:, 2:] - a[:, :2]).prod(dim=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(dim=1)
    return inter / (area_a + area_b - inter).clamp(min=1e-9) >= AGREEMENT_IOU


def _containment(merged: Dict[str, Any], boxes: torch.Tensor, j: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """(K,) fraction of instance i[k] that lies inside instance j[k]. Uses masks, else segments, else boxes.

    Measures only the given pairs; comparing every pair of masks is slow.
    """
    out = torch.zeros(len(i))
    masks = merged.get("masks")
    if isinstance(masks, torch.Tensor) and len(masks) == len(boxes):
        for r in i.unique().tolist():
            ks = (i == r).nonzero(as_tuple=True)[0]
            x0, y0 = boxes[r, :2].floor().clamp(min=0).long().tolist()
            x1, y1 = boxes[r, 2:].ceil().long().tolist()
            piece = binarize_masks(masks[r, y0:y1, x0:x1])  # a mask stays inside its box, so the crop loses nothing
            area = int(piece.sum())
            if area:
                inter = (binarize_masks(masks[j[ks], y0:y1, x0:x1]) & piece).flatten(1).sum(dim=1)
                out[ks] = inter.float().cpu() / area
        return out

    segments = merged.get("segments")
    if segments is not None and len(segments) == len(boxes):
        for k, (a, b) in enumerate(zip(j.tolist(), i.tolist())):
            overlap = pairwise_overlap({"segments": [segments[a], segments[b]]})
            if overlap is not None and overlap[1][1] > 0:
                out[k] = float(overlap[0][0, 1] / overlap[1][1])
        return out

    bj, bi = boxes[j], boxes[i]
    inter = (torch.minimum(bj[:, 2:], bi[:, 2:]) - torch.maximum(bj[:, :2], bi[:, :2])).clamp(min=0).prod(dim=1)
    return inter / (bi[:, 2:] - bi[:, :2]).prod(dim=1).clamp(min=1e-9)


def _split_distinct_whole(merged: Dict[str, Any], boxes: torch.Tensor, groups: List[List[int]], whole: torch.Tensor) -> List[List[int]]:
    """Split a group so each distinct uncut object gets its own group.

    A group keeps only one uncut member, so a wrong link would otherwise delete a whole object. Two uncut
    members are one object when either holds ``SAME_OBJECT_CONTAINMENT`` of the other. Fragments stay with the
    first object.
    """
    out: List[List[int]] = []
    for g in groups:
        members = [m for m in g if whole[m]]
        if len(members) < 2:
            out.append(g)
            continue
        a = [p for p in members for q in members if p != q]
        b = [q for p in members for q in members if p != q]
        frac = _containment(merged, boxes, torch.tensor(a), torch.tensor(b)).tolist()
        same = {(p, q) for p, q, f in zip(a, b, frac) if f >= SAME_OBJECT_CONTAINMENT}
        clusters: List[List[int]] = []
        for m in members:
            home = next((c for c in clusters if (c[0], m) in same or (m, c[0]) in same), None)
            if home is None:
                clusters.append([m])
            else:
                home.append(m)
        if len(clusters) < 2:
            out.append(g)
            continue
        out.append(clusters[0] + [m for m in g if not whole[m]])
        out.extend(clusters[1:])
    return out


def _connected_groups(pairs: torch.Tensor, n: int) -> List[List[int]]:
    """Union-find over the link matrix; returns the member indices of each group."""
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


def _union_groups(merged: Dict[str, Any], groups: List[List[int]], whole: torch.Tensor) -> Tuple[Dict[str, Any], List[List[int]]]:
    """Turn each group into one output row, keeping input order.

    A group with uncut members keeps its best-scoring uncut member. A group of fragments only becomes the union
    of their geometry. Also returns the groups in output order.
    """
    scores = merged.get("scores")
    reps: List[int] = []
    widen: List[bool] = []
    for g in groups:
        pool = [i for i in g if whole[i]] or g
        if len(pool) == 1 or not isinstance(scores, torch.Tensor) or len(scores) <= max(g):
            reps.append(pool[0])
        else:
            reps.append(max(pool, key=lambda i: float(scores[i])))
        widen.append(len(g) > 1 and pool is g)
    order = sorted(range(len(groups)), key=lambda k: reps[k])
    groups = [groups[k] for k in order]
    reps = [reps[k] for k in order]
    widen = [widen[k] for k in order]

    out = filter_instances(merged, torch.tensor(reps, dtype=torch.long))

    boxes = merged.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes):
        stacked = out["boxes"].clone()
        for k, g in enumerate(groups):
            if not widen[k]:
                continue
            members = boxes[torch.tensor(g, dtype=torch.long, device=boxes.device)]
            stacked[k, :2] = members[:, :2].amin(dim=0)
            stacked[k, 2:] = members[:, 2:].amax(dim=0)
        out["boxes"] = stacked

    masks = merged.get("masks")
    if isinstance(masks, torch.Tensor) and len(masks):
        unioned = out["masks"].clone()
        for k, g in enumerate(groups):
            if widen[k]:
                idx = torch.tensor(g, dtype=torch.long, device=masks.device)
                unioned[k] = (masks[idx] != 0).any(dim=0).to(masks.dtype)
        out["masks"] = unioned

    segments = merged.get("segments")
    if segments is not None and len(segments):
        out["segments"] = [_union_polygons([segments[i] for i in g]) if widen[k] else segments[reps[k]] for k, g in enumerate(groups)]

    return out, groups


def _absorb_fragments(out: Dict[str, Any], fragment_only: torch.Tensor, containment: float) -> Dict[str, Any]:
    """Drop leftover fragments. Output r is dropped when all hold:

    1. r was built only from fragments.
    2. Another output k has the same class and holds at least ``containment`` of r.
    3. k was seen whole, or outranks r (higher score, or same score and earlier row).

    The agreement check refuses some true links, which would leave pieces of a found object as extra detections.
    """
    n = len(fragment_only)
    boxes = instance_boxes(out) if n >= 2 and fragment_only.any() else None
    if boxes is None or len(boxes) != n:
        return out

    frag = fragment_only.nonzero(as_tuple=True)[0]
    scores = out.get("scores")
    s = scores.detach().cpu().float() if isinstance(scores, torch.Tensor) and len(scores) == n else torch.zeros(n)
    idx = torch.arange(n)
    outranks = (s[:, None] > s[frag][None, :]) | ((s[:, None] == s[frag][None, :]) & (idx[:, None] < frag[None, :]))
    cand = _touching(boxes, frag) & (~fragment_only[:, None] | outranks) & (idx[:, None] != frag[None, :])
    codes = class_codes(out.get("classes"), n)
    if codes is not None:
        cand &= codes[:, None] == codes[frag][None, :]
    k, col = cand.nonzero(as_tuple=True)
    if not len(k):
        return out
    inside = _containment(out, boxes, k, frag[col]) >= containment
    drop = torch.zeros(n, dtype=torch.bool)
    drop[frag[col[inside]]] = True
    if not drop.any():
        return out
    return filter_instances(out, (~drop).nonzero(as_tuple=True)[0])


def _union_polygons(polys: List[Any]) -> torch.Tensor:
    """Outer ring of the union of polygons. Holes are lost, since a segment is a single ring."""
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
    if isinstance(union, MultiPolygon):  # pieces that do not touch: keep the largest
        union = max(union.geoms, key=lambda g: g.area)
    ring = np.asarray(union.simplify(SIMPLIFY_TOLERANCE).exterior.coords[:-1], dtype=np.float32)

    ref = polys[0]
    if isinstance(ref, torch.Tensor):
        return torch.as_tensor(ring, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(ring)
