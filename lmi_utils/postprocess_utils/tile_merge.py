"""Combine detections of one object that tiled inference found in several tiles.

The image is split into overlapping tiles and the detector runs on each tile separately. An object that crosses a
tile edge is only partly visible in that tile, so its box stops at the edge. Terms used in this file:

- cut detection: its box reaches an inner tile edge (not the image border), so the object may continue in the next tile.
- whole detection: its box stays away from every inner tile edge.
- join: mark two detections from different tiles as the same object. Joined detections form a group, and each group
  becomes one output detection. A group can cover any number of tiles.

Input is one image's result dict, already in image coordinates, plus the index of the tile each detection came from.
Supports boxes, masks and segments; not keypoints or oriented boxes.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .mask_crops import MaskCrops, boxes_from_masks
from .nms import class_codes, filter_instances, intersecting_pairs, pairwise_overlap

# Not user settings: they depend on the detector and the tile layout, not on the dataset.
DEFAULT_EDGE_TOLERANCE = 2.0  # a box this many px from an inner tile edge counts as reaching it
LINK_MARGIN = 16.0  # a box this many px from an inner tile edge may be cut and may be joined; limited by the tile overlap
SAME_OBJECT_CONTAINMENT = 0.95  # two whole detections are one object when this share of one lies inside the other
AGREEMENT_IOU = 0.5  # to join, two boxes must reach this IoU within the area both tiles cover
SIMPLIFY_TOLERANCE = 1.0  # px a combined polygon's outline may move when it is simplified


# merge_origin codes: how each output detection was built
ORIGIN_WHOLE = 0  # a whole detection that nothing was joined to
ORIGIN_WHOLE_GROUPED = 1  # a whole detection kept for its group; the group's cut detections were discarded
ORIGIN_UNION = 2  # a group with no whole detection: the output covers all its cut detections combined
ORIGIN_FRAGMENT = 3  # a cut detection that nothing was joined to


def merge_tile_fragments(
    merged: Dict[str, Any],
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    im_size: Tuple[int, int],
    containment: float,
    edge_tolerance: float = DEFAULT_EDGE_TOLERANCE,
    report_origin: bool = False,
) -> Dict[str, Any]:
    """Combine the detections of each object that overlapping tiles found separately.

    Args:
        merged: one image's result dict, in image coordinates.
        tile_idx: (N,) tile index of each detection.
        tile_origins: (T, 2) top-left (y, x) of each tile in the image.
        tile_size: (tile_h, tile_w).
        im_size: (im_h, im_w) of the original image, without padding.
        containment: to join a cut detection with a whole one, at least this share of the cut one must lie inside
            the whole one.
        edge_tolerance: a box this many px from an inner tile edge counts as reaching it. It matches how far the
            detector's box can stop short of a tile edge, so it does not grow with tile size. Used to decide which
            leftover cut detections may be deleted. Joining uses the wider ``LINK_MARGIN``, which is safe because a
            join also needs a matching detection in the other tile.
        report_origin: add a ``merge_origin`` code (``ORIGIN_*``) to each output detection, saying how it was built.
            Off by default, so the result dict keeps its usual keys. Missing when the function returns before merging.

    Returns:
        The dict with each group replaced by one detection per separate object. A group with a whole detection
        outputs its best-scoring whole detection. A group of only cut detections outputs their shapes combined,
        with the best member's score, and is deleted when its detections all reach a tile edge and another tile
        covered its whole area.
    """
    # one detection still gets the cut test and the drop rule
    if not len(tile_idx):
        return merged

    masks = merged.get("masks")
    if isinstance(masks, torch.Tensor) and len(masks):
        crops = MaskCrops.from_masks(masks, tuple(masks.shape[1:]))
        out = merge_tile_fragments(
            {**merged, "masks": crops}, tile_idx, tile_origins, tile_size, im_size, containment, edge_tolerance, report_origin
        )
        return {**out, "masks": out["masks"].paste()}

    boxes = instance_boxes(merged)
    if boxes is None:
        return merged

    codes = class_codes(merged.get("classes"), len(tile_idx))
    margin = _link_margin(tile_origins, tile_size, edge_tolerance)
    cut = _cut_flags(boxes, tile_idx, tile_origins, tile_size, im_size, margin)
    at_edge = _cut_flags(boxes, tile_idx, tile_origins, tile_size, im_size, edge_tolerance)
    links = _links(merged, boxes, cut, tile_idx, tile_origins, tile_size, codes, containment)
    whole = ~cut
    joined = {(int(a), int(b)) for a, b in links.tolist()}
    joined |= {(b, a) for a, b in joined}
    groups = _split_distinct_whole(merged, boxes, _connected_groups(links, len(tile_idx)), whole, joined)
    out = merged
    united = any(len(g) > 1 for g in groups)
    if united:
        out, groups = _union_groups(merged, groups, whole)
    fragment_only = torch.tensor([not bool(whole[torch.tensor(g)].any()) for g in groups])
    if report_origin:
        # nothing was combined, so the output keeps the input order and each code goes at its detection's own index
        codes = torch.zeros(len(groups) if united else len(tile_idx), dtype=torch.uint8)
        for k, g in enumerate(groups):
            if fragment_only[k]:
                codes[k if united else g[0]] = ORIGIN_UNION if len(g) > 1 else ORIGIN_FRAGMENT
            else:
                codes[k if united else g[0]] = ORIGIN_WHOLE_GROUPED if len(g) > 1 else ORIGIN_WHOLE
        out = {**out, "merge_origin": codes}
    # only groups whose detections all reach the edge may be deleted; a whole object that just sits near an edge must stay
    droppable = torch.zeros(len(groups) if united else len(tile_idx), dtype=torch.bool)
    for k, g in enumerate(groups):
        # ``out`` has one row per group only when a union ran; otherwise it keeps the input rows, as the codes above do
        droppable[k if united else g[0]] = bool(at_edge[torch.tensor(g)].all())
    return _drop_fragments_seen_whole(out, droppable, tile_origins, tile_size, im_size, edge_tolerance)


def instance_boxes(merged: Dict[str, Any]) -> Optional[torch.Tensor]:
    """(N, 4) xyxy box of each detection, taken from boxes, else masks, else segments."""
    boxes = merged.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes):
        b = boxes.detach().cpu().float()
        if b.ndim == 3:  # oriented box (N, 4, 2): use the upright box around it
            return torch.cat([b.amin(dim=1), b.amax(dim=1)], dim=1)
        return b

    masks = merged.get("masks")
    if isinstance(masks, MaskCrops) and len(masks):
        return masks.boxes.float()
    if isinstance(masks, torch.Tensor) and len(masks):
        return boxes_from_masks(masks).cpu()

    segments = merged.get("segments")
    if segments is not None and len(segments):
        out = torch.zeros(len(segments), 4)
        for i, s in enumerate(segments):
            pts = s.detach().cpu().float() if isinstance(s, torch.Tensor) else torch.as_tensor(np.asarray(s), dtype=torch.float32)
            if len(pts):
                out[i] = torch.cat([pts.amin(dim=0), pts.amax(dim=0)])
        return out
    return None


def _link_margin(tile_origins: np.ndarray, tile_size: Tuple[int, int], edge_tolerance: float) -> float:
    """Px distance used to flag cut detections for joining: ``LINK_MARGIN``, at most 1/8 of the smallest tile overlap,
    and never below ``edge_tolerance``.

    Two neighbouring tiles each apply this margin on their side of the strip they share. Keeping it small leaves most
    of the strip outside both margins, so an object inside the strip is still whole in at least one tile.
    """
    margin = LINK_MARGIN
    for axis in (0, 1):
        starts = np.unique(np.asarray(tile_origins)[:, axis])
        if len(starts) > 1:
            margin = min(margin, (tile_size[axis] - float(np.diff(starts).min())) / 8)
    return max(margin, edge_tolerance)


def _cut_flags(
    boxes: torch.Tensor,
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    im_size: Tuple[int, int],
    tolerance: float,
) -> torch.Tensor:
    """(N,) bool: the box is within ``tolerance`` px of an inner edge of its own tile, so the object may continue in
    the next tile.

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
    """(L, 2) pairs [j, i]: cut detection i and detection j are the same object and are joined. All must hold:

    1. i is cut.
    2. j comes from another tile and has the same class.
    3. Their boxes reach ``AGREEMENT_IOU`` after both are trimmed to the area both tiles cover.
    4. j is cut too, or at least ``containment`` of i lies inside j. Without this, j could be a short piece that
       stops just before its tile edge and so looks whole; the group would then output that short piece and
       discard i, the longer part of the object.

    Pairs, not an (N, N) matrix: a crowded tiled image carries tens of thousands of detections and the matrix
    of those runs to gigabytes, nearly all of it zero.
    """
    empty = torch.zeros(0, 2, dtype=torch.long)
    if not cut.any():
        return empty
    # every intersecting pair, in both directions, keeping those whose second member is the cut one
    touching = intersecting_pairs(boxes)
    j = torch.cat([touching[:, 0], touching[:, 1]])
    i = torch.cat([touching[:, 1], touching[:, 0]])
    keep = cut[i] & (tile_idx[j] != tile_idx[i])
    if codes is not None:
        keep &= codes[j] == codes[i]
    j, i = j[keep], i[keep]
    if not len(j):
        return empty
    keep = _agree_in_shared_region(boxes, j, i, tile_idx, tile_origins, tile_size)
    j, i = j[keep], i[keep]
    uncut = ~cut[j]
    if uncut.any():
        keep = torch.ones(len(j), dtype=torch.bool)
        keep[uncut] = _containment(merged, boxes, j[uncut], i[uncut]) >= containment
        j, i = j[keep], i[keep]
    return torch.stack([j, i], dim=1) if len(j) else empty


def _agree_in_shared_region(
    boxes: torch.Tensor,
    j: torch.Tensor,
    i: torch.Tensor,
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
) -> torch.Tensor:
    """(K,) bool: boxes j[k] and i[k] reach ``AGREEMENT_IOU`` after both are trimmed to the area both their tiles cover.

    Both tiles see that area fully, so two detections of the same object nearly match there. The bigger box of a
    different, nearby object that merely contains the cut detection does not.
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
    """(K,) share of detection i[k] that lies inside detection j[k]. Uses masks, else segments, else boxes.

    Measures only the given pairs; comparing every pair of masks is slow.
    """
    out = torch.zeros(len(i))
    masks = merged.get("masks")
    if isinstance(masks, MaskCrops) and len(masks) == len(boxes):
        areas = masks.areas()
        for k, (a, b) in enumerate(zip(j.tolist(), i.tolist())):
            if areas[b]:
                out[k] = masks.intersection(a, b) / float(areas[b])
        return out

    segments = merged.get("segments")
    if segments is not None and len(segments) == len(boxes):
        for k, (a, b) in enumerate(zip(j.tolist(), i.tolist())):
            pairs, inter, areas = pairwise_overlap({"segments": [segments[a], segments[b]]})
            if len(pairs) and areas[1] > 0:
                out[k] = float(inter[0] / areas[1])
        return out

    bj, bi = boxes[j], boxes[i]
    inter = (torch.minimum(bj[:, 2:], bi[:, 2:]) - torch.maximum(bj[:, :2], bi[:, :2])).clamp(min=0).prod(dim=1)
    return inter / (bi[:, 2:] - bi[:, :2]).prod(dim=1).clamp(min=1e-9)


def _split_distinct_whole(
    merged: Dict[str, Any], boxes: torch.Tensor, groups: List[List[int]], whole: torch.Tensor, joined: Set[Tuple[int, int]]
) -> List[List[int]]:
    """Split a group so that each separate whole object gets its own group.

    A group outputs only one whole detection, so one wrong join could otherwise delete a real object. Two whole
    detections are the same object when ``SAME_OBJECT_CONTAINMENT`` of either lies inside the other, and detections
    connected through a chain of such pairs are all one object. Each cut detection goes with the object it is joined
    to in ``joined`` (the symmetric set of joined index pairs), through other cut detections if needed; one joined to
    several objects goes with the nearest, then the first.
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
        seat = {m: k for k, m in enumerate(members)}
        edges = torch.tensor([[seat[p], seat[q]] for p, q in same], dtype=torch.long).reshape(-1, 2)
        clusters = [[members[k] for k in c] for c in _connected_groups(edges, len(members))]
        if len(clusters) < 2:
            out.append(g)
            continue
        # breadth-first from every object at once, stepping only through cut detections
        owner = {m: k for k, c in enumerate(clusters) for m in c}
        cuts = [m for m in g if not whole[m]]
        frontier = [m for c in clusters for m in c]
        while frontier:
            reached = []
            for m in frontier:
                for c in cuts:
                    if c not in owner and (m, c) in joined:
                        owner[c] = owner[m]
                        reached.append(c)
            frontier = reached
        out.extend(c + [m for m in cuts if owner.get(m, 0) == k] for k, c in enumerate(clusters))
    return out


def _connected_groups(edges: torch.Tensor, n: int) -> List[List[int]]:
    """Put detections connected through the (E, 2) index pairs into the same group (union-find); returns each group's indices."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges.tolist():
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    return list(buckets.values())


def _union_groups(merged: Dict[str, Any], groups: List[List[int]], whole: torch.Tensor) -> Tuple[Dict[str, Any], List[List[int]]]:
    """Turn each group into one output detection, keeping input order.

    A group with whole detections outputs its best-scoring whole detection. A group of only cut detections outputs
    their shapes combined (box, mask or polygon). Also returns the groups in output order.
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

    # filtering copies boxes and masks, so editing them below leaves ``merged`` unchanged
    out = filter_instances(merged, torch.tensor(reps, dtype=torch.long))

    boxes = merged.get("boxes")
    masks = merged.get("masks")
    for k, g in enumerate(groups):
        if not widen[k]:
            continue
        if isinstance(boxes, torch.Tensor) and len(boxes):
            members = boxes[torch.tensor(g, dtype=torch.long, device=boxes.device)]
            out["boxes"][k, :2] = members[:, :2].amin(dim=0)
            out["boxes"][k, 2:] = members[:, 2:].amax(dim=0)
        if isinstance(masks, MaskCrops) and len(masks):
            out["masks"].set(k, *masks.union(g))

    segments = merged.get("segments")
    if segments is not None and len(segments):
        out["segments"] = [_union_polygons([segments[i] for i in g]) if widen[k] else segments[reps[k]] for k, g in enumerate(groups)]

    return out, groups


def _drop_fragments_seen_whole(
    out: Dict[str, Any],
    droppable: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    im_size: Tuple[int, int],
    tolerance: float,
) -> Dict[str, Any]:
    """Delete leftover cut detections whose whole area another tile covered. Output detection r is deleted when both hold:

    1. ``droppable[r]``: every detection r was built from reaches an inner tile edge within ``tolerance``.
    2. r's box lies inside some tile and stays more than ``tolerance`` from that tile's inner edges.

    That tile saw the whole area, so a real object there would have been detected whole. r is therefore an inaccurate
    extra detection of that object, or a false detection. Only position is checked: boxes of small pieces at a tile
    edge are too inaccurate for an overlap test.
    """
    n = len(droppable)
    boxes = instance_boxes(out) if droppable.any() else None
    if boxes is None or len(boxes) != n:
        return out

    frag = droppable.nonzero(as_tuple=True)[0]
    n_tiles = len(tile_origins)
    b = boxes[frag].repeat_interleave(n_tiles, dim=0)
    tiles = torch.arange(n_tiles).repeat(len(frag))
    lo = torch.as_tensor(tile_origins, dtype=torch.float32)[tiles].flip(1)  # (x, y)
    hi = lo + torch.tensor([tile_size[1], tile_size[0]], dtype=torch.float32)
    inside = (b[:, :2] >= lo).all(dim=1) & (b[:, 2:] <= hi).all(dim=1)
    seen = (inside & ~_cut_flags(b, tiles, tile_origins, tile_size, im_size, tolerance)).view(len(frag), n_tiles).any(dim=1)
    if not seen.any():
        return out
    drop = torch.zeros(n, dtype=torch.bool)
    drop[frag[seen]] = True
    return filter_instances(out, (~drop).nonzero(as_tuple=True)[0])


def _union_polygons(polys: List[Any]) -> torch.Tensor:
    """Outline of the polygons combined into one shape. Holes are lost, because a segment stores a single outline."""
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
    if isinstance(union, MultiPolygon):  # the polygons do not all touch: keep the largest piece
        union = max(union.geoms, key=lambda g: g.area)
    ring = np.asarray(union.simplify(SIMPLIFY_TOLERANCE).exterior.coords[:-1], dtype=np.float32)

    ref = polys[0]
    if isinstance(ref, torch.Tensor):
        return torch.as_tensor(ring, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(ring)
