"""Combine detections of one object that tiled inference found in several tiles.

The image is split into tiles, usually overlapping, and the detector runs on each tile separately. An object that
crosses a tile edge is only partly visible in that tile, so its box stops at the edge. Terms used in this file:

- cut detection: its box reaches an inner tile edge (not the image border), so the object may continue in the next tile.
- whole detection: its box stays away from every inner tile edge.
- join: mark two detections from different tiles as the same object. Joined detections form a group, and each group
  becomes one output detection. A group can cover any number of tiles.

Input is one image's result dict, already in image coordinates, plus the index of the tile each detection came from.
Supports boxes, masks and segments; not keypoints or oriented boxes.

Mask pixels stay on the model's device. The per-detection box algebra runs on the CPU: at the few hundred
detections an image carries, a kernel launch costs more than the arithmetic. The pair scan is the exception and
is sent back, being the one step whose size pays for the launch. Only result vectors cross back.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .mask_crops import MaskCrops, boxes_from_masks
from .nms import box_intersections, class_codes, filter_instances, intersecting_pairs, pairwise_overlap, result_device

# Not user settings: they depend on the detector and the tile layout, not on the dataset.
DEFAULT_EDGE_TOLERANCE = 2.0  # a box this many px from an inner tile edge counts as reaching it
JOIN_MARGIN = 16.0  # a box this many px from an inner tile edge may be cut and may be joined; limited by the tile overlap
SAME_OBJECT_CONTAINMENT = 0.95  # two whole detections are one object when this share of one lies inside the other
AGREEMENT_IOU = 0.5  # to join, two boxes must reach this IoU within the area both tiles cover
SIMPLIFY_TOLERANCE = 1.0  # px a combined polygon's outline may move when it is simplified
_NO_PAIRS = torch.zeros(0, 2, dtype=torch.long)  # an (L, 2) index-pair list with nothing in it
_TILE_TEST_BUDGET = 1_000_000  # (fragment, tile) rows tested at once when looking for a tile that saw a fragment whole


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
            leftover cut detections may be deleted. Joining uses the wider ``JOIN_MARGIN``, which is safe because a
            join also needs a matching detection in the other tile. Without tile overlap it is also the half-width of
            the band the two pieces are compared in, so it should stay small.
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

    grid = _Grid(tile_origins, tile_size, im_size, edge_tolerance)
    cut = _cut_flags(boxes, tile_idx, grid, grid.margin)
    at_edge = _cut_flags(boxes, tile_idx, grid, edge_tolerance)
    whole = ~cut
    touching = _touching_pairs(merged, boxes, cut, grid.pad)
    joins = _joins(merged, boxes, cut, tile_idx, grid, containment, touching)
    groups = _split_distinct_whole(merged, boxes, _connected_groups(joins, len(tile_idx)), whole, joins, touching)

    united = any(len(g) > 1 for g in groups)
    out, groups = _union_groups(merged, groups, whole, grid) if united else (merged, groups)
    rows = _Rows(groups) if united else _Rows(groups, len(tile_idx))
    if report_origin:
        out = {**out, "merge_origin": rows.scatter(_origin_codes(rows, whole))}
    # only groups whose detections all reach the edge may be deleted; a whole object that just sits near an edge must stay
    return _drop_fragments_seen_whole(out, rows.scatter(rows.every(at_edge)), grid)


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


class _Grid:
    """Where the tiles sit, and the margins that follow from how much they overlap.

    ``margin`` is the distance used to flag cut detections for joining: ``JOIN_MARGIN``, at most 1/8 of the
    smallest tile overlap, and never below ``tolerance``. Two neighbouring tiles each apply it on their side of
    the strip they share, so keeping it small leaves most of the strip outside both margins and an object inside
    the strip still whole in at least one tile.
    """

    def __init__(self, origins: np.ndarray, size: Tuple[int, int], im_size: Tuple[int, int], tolerance: float):
        self.origins = np.asarray(origins)
        self.size = size
        self.im_size = im_size
        self.tolerance = tolerance
        overlap = self._overlap()
        self.margin = max(min(JOIN_MARGIN, float(overlap.min()) / 8), tolerance)
        self.pad = self._band_pad(overlap)

    def _overlap(self) -> np.ndarray:
        """(2,) px two neighbouring tiles share on each tile axis (y, x); ``inf`` on an axis with no seam to share across."""
        out = []
        for axis in (0, 1):
            starts = np.unique(self.origins[:, axis])
            out.append(self.size[axis] - float(np.diff(starts).min()) if len(starts) > 1 else np.inf)
        return np.array(out)

    def _band_pad(self, overlap: np.ndarray) -> torch.Tensor:
        """(2,) px in (x, y) to widen the area two tiles cover by, so it is at least ``2 * margin`` wide on each axis.

        Zero wherever the tiles overlap by more than that, the usual case. It reaches ``margin`` for tiles with no
        overlap, whose shared area is a line with no width to measure anything in.
        """
        pad = ((2 * self.margin - overlap) / 2).clip(min=0)
        return torch.tensor(pad[::-1].copy(), dtype=torch.float32)  # (y, x) tile axes -> (x, y) box axes

    def corners(self, tile_idx: torch.Tensor) -> torch.Tensor:
        """(N, 2) top-left (x, y) of the tile each detection came from."""
        return torch.as_tensor(self.origins, dtype=torch.float32, device=tile_idx.device)[tile_idx].flip(1)

    def seams(self) -> Tuple[np.ndarray, np.ndarray]:
        """Inner tile edges in image px: x of the vertical ones, y of the horizontal ones. The image border is not a seam."""
        out = []
        for axis in (1, 0):
            starts = np.unique(self.origins[:, axis])
            edges = np.unique(np.concatenate([starts, starts + self.size[axis]]))
            out.append(edges[(edges > 0) & (edges < self.im_size[axis])])
        return out[0], out[1]


class _Rows:
    """Every group's members in one flat run, and the output row each group's summary belongs on.

    The summaries are then scatters over that run, not a pass per group. ``out`` has one row per group only when
    a union ran; otherwise it keeps the input rows, and a group's summary lands on its first member's row.
    """

    def __init__(self, groups: List[List[int]], n_rows: Optional[int] = None):
        """``n_rows`` is the input row count when no union ran; leave it out for one output row per group."""
        self.sizes = torch.tensor([len(g) for g in groups])
        self.members = torch.tensor([m for g in groups for m in g], dtype=torch.long)
        self.gid = torch.repeat_interleave(torch.arange(len(groups)), self.sizes)
        self.n_rows = len(groups) if n_rows is None else n_rows
        self.at = torch.arange(len(groups)) if n_rows is None else torch.tensor([g[0] for g in groups], dtype=torch.long)

    def any(self, flags: torch.Tensor) -> torch.Tensor:
        """(G,) bool: ``flags`` is set on at least one member of the group."""
        return self._fold(flags, "amax")

    def every(self, flags: torch.Tensor) -> torch.Tensor:
        """(G,) bool: ``flags`` is set on every member of the group."""
        return self._fold(flags, "amin")

    def _fold(self, flags: torch.Tensor, reduce: str) -> torch.Tensor:
        n = len(self.sizes)
        return torch.zeros(n, dtype=torch.bool).scatter_reduce_(0, self.gid, flags[self.members], reduce=reduce, include_self=False)

    def scatter(self, values: torch.Tensor) -> torch.Tensor:
        """(n_rows,) with each group's value on its own output row, zero elsewhere."""
        out = torch.zeros(self.n_rows, dtype=values.dtype)
        out[self.at] = values
        return out


def _origin_codes(rows: _Rows, whole: torch.Tensor) -> torch.Tensor:
    """(G,) ``ORIGIN_*`` code saying how each group's output detection was built."""
    fragment_only = ~rows.any(whole)
    grouped = rows.sizes > 1
    return torch.where(
        fragment_only,
        torch.where(grouped, ORIGIN_UNION, ORIGIN_FRAGMENT),
        torch.where(grouped, ORIGIN_WHOLE_GROUPED, ORIGIN_WHOLE),
    ).to(torch.uint8)


def _mask_crops(merged: Dict[str, Any], boxes: torch.Tensor) -> Optional[MaskCrops]:
    """The detections' masks as crops, when there is one for each box."""
    masks = merged.get("masks")
    return masks if isinstance(masks, MaskCrops) and len(masks) == len(boxes) else None


def _cut_flags(boxes: torch.Tensor, tile_idx: torch.Tensor, grid: "_Grid", tolerance: float) -> torch.Tensor:
    """(N,) bool: the box is within ``tolerance`` px of an inner edge of its own tile, so the object may continue in
    the next tile.

    Tile edges on the image border or in the padding do not count: nothing continues past them.
    """
    tile_h, tile_w = grid.size
    im_h, im_w = grid.im_size
    x0, y0 = grid.corners(tile_idx).unbind(1)
    y1, x1 = y0 + tile_h, x0 + tile_w
    return (
        ((x0 > 0) & (boxes[:, 0] <= x0 + tolerance))
        | ((y0 > 0) & (boxes[:, 1] <= y0 + tolerance))
        | ((x1 < im_w) & (boxes[:, 2] >= x1 - tolerance))
        | ((y1 < im_h) & (boxes[:, 3] >= y1 - tolerance))
    )


def _joins(
    merged: Dict[str, Any],
    boxes: torch.Tensor,
    cut: torch.Tensor,
    tile_idx: torch.Tensor,
    grid: "_Grid",
    containment: float,
    touching: torch.Tensor,
) -> torch.Tensor:
    """(L, 2) pairs [j, i]: cut detection i and detection j are the same object and are joined. All must hold:

    1. i is cut.
    2. j comes from another tile and has the same class.
    3. Their boxes reach ``AGREEMENT_IOU`` after both are trimmed to the area both tiles cover, widened by the grid's pad.
    4. j is cut too, or at least ``containment`` of i lies inside j. Without this, j could be a short piece that
       stops just before its tile edge and so looks whole; the group would then output that short piece and
       discard i, the longer part of the object.

    Pairs, not an (N, N) matrix: a crowded tiled image carries tens of thousands of detections and the matrix
    of those runs to gigabytes, nearly all of it zero.
    """
    if not len(touching):
        return _NO_PAIRS
    # ``touching`` in both directions, keeping the pairs whose second member is the cut one
    j = torch.cat([touching[:, 0], touching[:, 1]])
    i = torch.cat([touching[:, 1], touching[:, 0]])
    keep = cut[i] & (tile_idx[j] != tile_idx[i])
    codes = class_codes(merged.get("classes"), len(cut))
    if codes is not None:
        keep &= codes[j] == codes[i]
    j, i = j[keep], i[keep]
    if not len(j):
        return _NO_PAIRS
    keep = _agree_in_shared_region(merged, boxes, j, i, tile_idx, grid)
    j, i = j[keep], i[keep]
    uncut = ~cut[j]
    if uncut.any():
        keep = torch.ones(len(j), dtype=torch.bool)
        keep[uncut] = _containment(merged, boxes, j[uncut], i[uncut]) >= containment
        j, i = j[keep], i[keep]
    return torch.stack([j, i], dim=1) if len(j) else _NO_PAIRS


def _touching_pairs(merged: Dict[str, Any], boxes: torch.Tensor, cut: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """(L, 2) index pairs i < j whose boxes intersect once grown by ``pad``, so two pieces that only meet at a seam
    are still candidates.

    Scanned once and used by both the join test and the split below. Empty when nothing is cut: there is then
    nothing to join, so every group holds one detection and the split has no work either. The scan is ~12x faster
    on a GPU at tens of thousands of detections, and the pairs come back in one transfer.
    """
    if not cut.any():
        return _NO_PAIRS
    return intersecting_pairs(_grow(boxes, pad).to(result_device(merged))).cpu()


def _grow(boxes: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """xyxy boxes widened by ``pad`` (x, y) px on each side."""
    return boxes + torch.cat([-pad, pad]).to(boxes)  # ``to(tensor)`` matches its device as well as its dtype


def _trimmed(
    merged: Dict[str, Any], boxes: torch.Tensor, idx: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, grid: "_Grid"
) -> torch.Tensor:
    """(K, 4) box of each detection's shape inside the region ``lo``-``hi``, then grown by the grid's pad and held to it.

    A detection whose shape misses the region keeps a zero-area box, so it agrees with nothing. Where two tiles
    only meet at a seam the region is a thin band, and a mask then answers with the slice of the object that
    actually crosses there: a long object cut at an angle has two halves whose boxes barely overlap but whose
    crossings line up. Masks only; a polygon or a plain box answers with the box, as ``_containment`` does.
    """
    inside = torch.minimum(torch.maximum(boxes[idx], lo), hi)
    masks = _mask_crops(merged, boxes)
    if masks is not None:
        clipped = (inside != boxes[idx]).any(dim=1)
        inside[clipped] = masks.boxes_in_regions(idx[clipped], lo[clipped], hi[clipped], inside[clipped]).cpu()
    empty = ((inside[:, 2:] - inside[:, :2]) <= 0).any(dim=1)
    out = torch.minimum(torch.maximum(_grow(inside, grid.pad), lo), hi)
    out[empty] = lo[empty]
    return out


def _agree_in_shared_region(
    merged: Dict[str, Any],
    boxes: torch.Tensor,
    j: torch.Tensor,
    i: torch.Tensor,
    tile_idx: torch.Tensor,
    grid: "_Grid",
) -> torch.Tensor:
    """(K,) bool: detections j[k] and i[k] reach ``AGREEMENT_IOU`` once ``_trimmed`` holds each to the area both
    their tiles cover, widened by the grid's pad.

    Both tiles see that area fully, so two detections of the same object nearly match there. The bigger box of a
    different, nearby object that merely contains the cut detection does not.

    That pad is zero whenever the tiles overlap, and nothing moves. Where they do not, it turns the line they share
    into a band: both pieces then span its full width, and the measure falls back to how far the two agree along
    the seam — all either tile can still say about the other's side.
    """
    if not len(j):
        return torch.zeros(0, dtype=torch.bool)
    tile_h, tile_w = grid.size
    origins = grid.corners(tile_idx)
    pad = grid.pad.to(origins)
    extent = torch.tensor([tile_w, tile_h], dtype=torch.float32, device=origins.device)
    lo = (torch.maximum(origins[j], origins[i]) - pad).repeat(1, 2)
    hi = (torch.minimum(origins[j], origins[i]) + extent + pad).repeat(1, 2)
    a = _trimmed(merged, boxes, j, lo, hi, grid)
    b = _trimmed(merged, boxes, i, lo, hi, grid)
    inter = box_intersections(a, b)
    area_a = (a[:, 2:] - a[:, :2]).prod(dim=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(dim=1)
    return inter / (area_a + area_b - inter).clamp(min=1e-9) >= AGREEMENT_IOU


def _containment(merged: Dict[str, Any], boxes: torch.Tensor, j: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """(K,) share of detection i[k] that lies inside detection j[k]. Uses masks, else segments, else boxes.

    Measures only the given pairs; comparing every pair of masks is slow.
    """
    out = torch.zeros(len(i))
    masks = _mask_crops(merged, boxes)
    if masks is not None:
        areas = masks.areas()[i]
        return torch.where(areas > 0, masks.intersections(j, i) / areas.clamp(min=1e-9), out.to(areas)).cpu()

    segments = merged.get("segments")
    if segments is not None and len(segments) == len(boxes):
        for k, (a, b) in enumerate(zip(j.tolist(), i.tolist())):
            pairs, inter, areas = pairwise_overlap({"segments": [segments[a], segments[b]]})
            if len(pairs) and areas[1] > 0:
                out[k] = float(inter[0] / areas[1])
        return out

    bj, bi = boxes[j], boxes[i]
    return box_intersections(bj, bi) / (bi[:, 2:] - bi[:, :2]).prod(dim=1).clamp(min=1e-9)


def _split_distinct_whole(
    merged: Dict[str, Any],
    boxes: torch.Tensor,
    groups: List[List[int]],
    whole: torch.Tensor,
    joins: torch.Tensor,
    touching: torch.Tensor,
) -> List[List[int]]:
    """Split a group so that each separate whole object gets its own group.

    A group outputs only one whole detection, so one wrong join could otherwise delete a real object. Two whole
    detections are the same object when ``SAME_OBJECT_CONTAINMENT`` of either lies inside the other, and detections
    connected through a chain of such pairs are all one object. Each cut detection goes with the object it is joined
    to in ``joins``, through other cut detections if needed; one joined to several objects goes with the nearest,
    then the first.
    """
    joined = {(int(a), int(b)) for a, b in joins.tolist()}
    joined |= {(b, a) for a, b in joined}  # the walk below is direction-blind
    whole_of = whole.tolist()
    wholes = [[m for m in g if whole_of[m]] for g in groups]
    edges_of = _same_object_edges(merged, boxes, wholes, touching)

    out: List[List[int]] = []
    for k, g in enumerate(groups):
        members = wholes[k]
        if len(members) < 2:
            out.append(g)
            continue
        edges = edges_of.get(k, _NO_PAIRS)
        clusters = [[members[s] for s in c] for c in _connected_groups(edges, len(members))]
        if len(clusters) < 2:
            out.append(g)
            continue
        # breadth-first from every object at once, stepping only through cut detections
        owner = {m: c_i for c_i, c in enumerate(clusters) for m in c}
        cuts = [m for m in g if not whole_of[m]]
        frontier = [m for c in clusters for m in c]
        while frontier:
            reached = []
            for m in frontier:
                for c in cuts:
                    if c not in owner and (m, c) in joined:
                        owner[c] = owner[m]
                        reached.append(c)
            frontier = reached
        out.extend(c + [m for m in cuts if owner.get(m, 0) == c_i] for c_i, c in enumerate(clusters))
    return out


def _same_object_edges(
    merged: Dict[str, Any], boxes: torch.Tensor, wholes: List[List[int]], touching: torch.Tensor
) -> Dict[int, torch.Tensor]:
    """(E, 2) pairs of whole detections that are the same object, as seats in ``wholes[k]``, per group that has any.

    Filtered out of the one global scan rather than rescanning each group: only intersecting detections can
    contain one another, so that scan already holds every pair any group could need, and a group can hold
    thousands whose full cross product costs more memory than everything else here put together.
    """
    multi = [(k, members) for k, members in enumerate(wholes) if len(members) > 1]
    if not multi or not len(touching):
        return {}
    flat = torch.tensor([m for _, members in multi for m in members], dtype=torch.long)
    seat = torch.full((len(boxes),), -1, dtype=torch.long)
    group_of = torch.full((len(boxes),), -1, dtype=torch.long)
    seat[flat] = torch.tensor([s for _, members in multi for s in range(len(members))], dtype=torch.long)
    group_of[flat] = torch.tensor([k for k, members in multi for _ in members], dtype=torch.long)

    ga, gb = group_of[touching[:, 0]], group_of[touching[:, 1]]
    keep = (ga >= 0) & (ga == gb)  # both ends whole, in the same group
    a, b = touching[keep, 0], touching[keep, 1]
    # either one inside the other makes them the same object, and the grouping below is direction-blind
    same = (_containment(merged, boxes, a, b) >= SAME_OBJECT_CONTAINMENT) | (_containment(merged, boxes, b, a) >= SAME_OBJECT_CONTAINMENT)
    a, b = a[same], b[same]

    rows: Dict[int, List[List[int]]] = {}
    for k, p, q in zip(group_of[a].tolist(), seat[a].tolist(), seat[b].tolist()):
        rows.setdefault(k, []).append([p, q])
    return {k: torch.tensor(e, dtype=torch.long) for k, e in rows.items()}


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


def _union_groups(
    merged: Dict[str, Any], groups: List[List[int]], whole: torch.Tensor, grid: Optional["_Grid"] = None
) -> Tuple[Dict[str, Any], List[List[int]]]:
    """Turn each group into one output detection, keeping input order.

    A group with whole detections outputs its best-scoring whole detection. A group of only cut detections outputs
    their shapes combined (box, mask or polygon). Also returns the groups in output order.

    A combined shape is closed across the grid's seams by its pad, (x, y) px and zero wherever tiles overlap, to bridge
    pieces that stop short of a seam. The rest of the shape is left as the model drew it. With masks, the polygon is
    traced from the closed mask.
    """
    gap_xy = [int(np.ceil(v)) for v in grid.pad.tolist()] if grid is not None else [0, 0]
    seams = grid.seams() if grid is not None else (np.zeros(0), np.zeros(0))
    scores = merged.get("scores")
    g = _Rows(groups)
    members, gid = g.members, g.gid
    is_whole = whole[members]
    has_whole = g.any(whole)

    pool = is_whole | ~has_whole[gid]  # the members allowed to represent their group
    usable = isinstance(scores, torch.Tensor) and len(scores) > int(members.max())
    ranked = scores.detach().cpu().float()[members] if usable else torch.zeros(len(members))
    ranked = ranked.masked_fill(~pool, -torch.inf)
    best = torch.full((len(groups),), -torch.inf).scatter_reduce_(0, gid, ranked, reduce="amax", include_self=False)
    # ties go to the group's first member, as the case without scores does
    seat = torch.arange(len(members)).masked_fill(ranked != best[gid], len(members))
    pick = torch.zeros(len(groups), dtype=torch.long).scatter_reduce_(0, gid, seat, reduce="amin", include_self=False)
    reps = members[pick]
    widen = (g.sizes > 1) & ~has_whole

    order = torch.argsort(reps)
    groups = [groups[k] for k in order.tolist()]
    reps, widen = reps[order], widen[order]

    # filtering copies boxes and masks, so editing them below leaves ``merged`` unchanged
    out = filter_instances(merged, reps)

    segments = merged.get("segments")
    has_segments = segments is not None and len(segments) > 0
    traced: Dict[int, Any] = {}  # output row -> polygon traced from its combined mask

    rows = widen.nonzero(as_tuple=True)[0].tolist()
    if rows:
        # one pass over every widened group's members: a round trip per group costs more than the union itself
        wide = _Rows([groups[k] for k in rows])
        flat, row = wide.members, wide.gid
        boxes = merged.get("boxes")
        masks = merged.get("masks")
        if isinstance(boxes, torch.Tensor) and len(boxes):
            src = boxes[flat.to(boxes.device)]
            at = row.to(boxes.device)[:, None].expand(-1, 2)
            blank = torch.zeros(len(rows), 2, dtype=src.dtype, device=boxes.device)  # ignored: include_self=False
            at_rows = torch.tensor(rows, device=boxes.device)
            out["boxes"][at_rows, :2] = blank.scatter_reduce(0, at, src[:, :2], "amin", include_self=False)
            out["boxes"][at_rows, 2:] = blank.scatter_reduce(0, at, src[:, 2:], "amax", include_self=False)
        if isinstance(masks, MaskCrops) and len(masks):
            data, crop_boxes = masks.unions(flat, row, len(rows))
            if any(gap_xy) or has_segments:
                crops = _closed_crops(data, crop_boxes, gap_xy, seams)
                if any(gap_xy):
                    data = torch.from_numpy(np.concatenate([c.reshape(-1) for c in crops])).to(data.device, torch.bool)
                if has_segments:
                    ref = segments[int(reps[rows[0]])]
                    for k, crop, box in zip(rows, crops, crop_boxes.tolist()):
                        ring = _trace(crop, box[0], box[1])
                        if ring is not None:
                            traced[k] = torch.as_tensor(ring, dtype=ref.dtype, device=ref.device) if isinstance(ref, torch.Tensor) else ring
            out["masks"].set_runs(torch.tensor(rows, dtype=torch.long), data, crop_boxes)

    if has_segments:
        rep_l, widen_l = reps.tolist(), widen.tolist()
        out["segments"] = [
            (traced[k] if k in traced else _union_polygons([segments[i] for i in g], gap_xy, seams)) if widen_l[k] else segments[rep_l[k]]
            for k, g in enumerate(groups)
        ]

    return out, groups


def _closed_crops(data: torch.Tensor, boxes: torch.Tensor, gap: List[int], seams: Tuple[np.ndarray, np.ndarray]) -> List[np.ndarray]:
    """Packed mask crops as (h, w) uint8 arrays, each closed with a rectangle reaching ``gap`` (x, y) px from its centre,
    keeping the fill only within ``gap`` of a seam.

    A closing never reaches past the crop's box, so the boxes and the packed layout stay valid.
    """
    import cv2

    host = data.cpu().numpy().astype(np.uint8)  # one transfer for every crop
    gx, gy = gap
    # a rectangle, not an ellipse: seams are axis-aligned, and cv2 shrinks an ellipse one pixel thick to a point
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * gx + 1, 2 * gy + 1)) if gx or gy else None
    out, at = [], 0
    for x0, y0, x1, y1 in boxes.tolist():
        w, h = x1 - x0, y1 - y0
        crop = host[at : at + w * h].reshape(h, w)
        at += w * h
        if kernel is not None and crop.size:
            # zero padding so the dilation is not cut at the crop edge
            closed = cv2.morphologyEx(np.pad(crop, ((gy, gy), (gx, gx))), cv2.MORPH_CLOSE, kernel)[gy : gy + h, gx : gx + w]
            # the pieces' gaps are at the seams; elsewhere a narrow dent or hole is the model's to keep
            band = _near_seam(x0 + np.arange(w), seams[0], gx)[None, :] | _near_seam(y0 + np.arange(h), seams[1], gy)[:, None]
            crop = crop | (closed & band)
        out.append(np.ascontiguousarray(crop))
    return out


def _near_seam(px: np.ndarray, seams: np.ndarray, gap: int) -> np.ndarray:
    """(P,) bool: pixel index within ``gap`` of a seam, on the side either piece may stop short of it."""
    if not gap or not len(seams):
        return np.zeros(len(px), dtype=bool)
    return ((px[:, None] >= seams - gap) & (px[:, None] < seams + gap)).any(axis=1)


def _trace(crop: np.ndarray, x0: int, y0: int) -> Optional[np.ndarray]:
    """Outline of a mask crop placed at (x0, y0), traced as the detectors trace theirs; None when the crop is empty.

    Pieces that still do not touch keep the largest: they are stray pixels or a second object, and a hull over them
    covers the background between.
    """
    import cv2

    if not crop.size:  # a group whose masks are all empty has a zero box
        return None
    contours = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return None
    points = max(contours, key=cv2.contourArea)
    return points.reshape(-1, 2).astype(np.float32) + np.array([x0, y0], dtype=np.float32)


def _seen_whole_by_a_tile(boxes: torch.Tensor, grid: "_Grid") -> torch.Tensor:
    """(F,) bool: some tile holds the box with more than the grid's tolerance px to each of that tile's inner edges.

    Blocked by ``_TILE_TEST_BUDGET``: a crowded image carries a hundred thousand fragments over a grid of several
    hundred tiles, and testing that product in one go runs to tens of gigabytes.
    """
    n_tiles = len(grid.origins)
    tiles = torch.arange(n_tiles)
    origins = grid.corners(tiles)
    extent = torch.tensor([grid.size[1], grid.size[0]], dtype=torch.float32)
    out = torch.zeros(len(boxes), dtype=torch.bool)
    block = max(1, _TILE_TEST_BUDGET // max(1, n_tiles))
    for start in range(0, len(boxes), block):
        chunk = boxes[start : start + block]
        b = chunk.repeat_interleave(n_tiles, dim=0)
        t = tiles.repeat(len(chunk))
        lo = origins[t]
        hi = lo + extent
        inside = (b[:, :2] >= lo).all(dim=1) & (b[:, 2:] <= hi).all(dim=1)
        fits = inside & ~_cut_flags(b, t, grid, grid.tolerance)
        out[start : start + len(chunk)] = fits.view(len(chunk), n_tiles).any(dim=1)
    return out


def _drop_fragments_seen_whole(out: Dict[str, Any], droppable: torch.Tensor, grid: "_Grid") -> Dict[str, Any]:
    """Delete leftover cut detections whose whole area another tile covered. Output detection r is deleted when both hold:

    1. ``droppable[r]``: every detection r was built from reaches an inner tile edge within the grid's tolerance.
    2. r's box lies inside some tile and stays more than that tolerance from that tile's inner edges.

    That tile saw the whole area, so a real object there would have been detected whole. r is therefore an inaccurate
    extra detection of that object, or a false detection. Only position is checked: boxes of small pieces at a tile
    edge are too inaccurate for an overlap test.
    """
    n = len(droppable)
    boxes = instance_boxes(out) if droppable.any() else None
    if boxes is None or len(boxes) != n:
        return out

    frag = droppable.nonzero(as_tuple=True)[0]
    seen = _seen_whole_by_a_tile(boxes[frag], grid)
    if not seen.any():
        return out
    drop = torch.zeros(n, dtype=torch.bool)
    drop[frag[seen]] = True
    return filter_instances(out, (~drop).nonzero(as_tuple=True)[0])


def _union_polygons(polys: List[Any], gap: Sequence[int] = (0, 0), seams: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> torch.Tensor:
    """Outline of the polygons combined into one shape. Holes are lost, because a segment stores a single outline.

    ``gap`` (x, y) closes the shape across ``seams``, the vertical then horizontal seam lines, bridging pieces that stop
    short of one; the rest of the outline is left as it was. Of pieces that still do not touch, the largest is kept.
    """
    from shapely import BufferJoinStyle
    from shapely.geometry import MultiPolygon, Polygon, box
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
    d = max(gap)
    if d > 0 and seams is not None:
        # mitre corners add one vertex each, round ones a whole arc
        mitre = BufferJoinStyle.mitre
        closed = unary_union([g.buffer(d, join_style=mitre) for g in geoms]).buffer(-d, join_style=mitre)
        far = 1e9
        # outlines run through pixel centres, so a piece's last column sits one px before the band a mask would use
        strips = [box(s - gap[0] - 1, -far, s + gap[0], far) for s in seams[0] if gap[0]]
        strips += [box(-far, s - gap[1] - 1, far, s + gap[1]) for s in seams[1] if gap[1]]
        if strips:
            union = unary_union([union, closed.intersection(unary_union(strips))])
    if isinstance(union, MultiPolygon):
        union = max(union.geoms, key=lambda g: g.area)
    ring = np.asarray(union.simplify(SIMPLIFY_TOLERANCE).exterior.coords[:-1], dtype=np.float32)

    ref = polys[0]
    if isinstance(ref, torch.Tensor):
        return torch.as_tensor(ring, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(ring)
