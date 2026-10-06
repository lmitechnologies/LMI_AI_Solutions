"""Combine detections of one object that tiled inference found in several tiles.

The image is split into tiles, usually overlapping, and the detector runs on each tile separately. An object that
crosses a tile edge is only partly visible in that tile, so its box stops at the edge. Terms used in this file:

- seam: an inner tile edge, one not on the image border.
- fragment: a detection whose box reaches a seam of its tile, so the object may continue in the next tile (``cut`` in code).
- whole detection: its box stays away from every seam of its tile.
- join: mark two detections from different tiles as the same object. Joined detections form a group, and each group
  becomes one output detection. A group can cover any number of tiles.
- seam band: ``_Grid.seam_band`` px added either side of a seam where tiles overlap by less than ``2 * margin``; zero otherwise.

Input is one image's result dict, already in image coordinates, plus the index of the tile each detection came from.
Supports boxes and masks. Segments are not merged: trace them from the merged masks with ``trace_crops``.

Mask pixels stay on the model's device. The per-detection box algebra runs on the CPU: at the few hundred
detections an image carries, a kernel launch costs more than the arithmetic. The pair scan is the exception and
is sent back, being the one step whose size pays for the launch. Only result vectors cross back.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .mask_crops import MaskCrops, boxes_from_masks
from .mask_segments import trace_mask
from .nms import box_intersections, class_codes, filter_instances, intersecting_pairs, result_device

# Not user settings: they depend on the detector and the tile layout, not on the dataset.
# px from a seam that counts as reaching it; matches detector box error, so it does not grow with tile size
DEFAULT_EDGE_TOLERANCE = 2.0
JOIN_MARGIN = 16.0  # px from a seam within which a box may be joined; capped by the tile overlap (see ``_Grid``)
CONTAINMENT = 0.8  # a fragment joins a whole detection when this share of it lies inside; F1 barely moves over 0.5-0.95
SAME_OBJECT_CONTAINMENT = 0.95  # two whole detections are one object when this share of one lies inside the other
AGREEMENT_IOU = 0.5  # to join, two boxes must reach this IoU within the area both tiles cover
_NO_PAIRS = torch.zeros(0, 2, dtype=torch.long)  # an (L, 2) index-pair list with nothing in it


# merge_origin codes: how each output detection was built
ORIGIN_WHOLE = 0  # a whole detection that nothing was joined to
ORIGIN_WHOLE_GROUPED = 1  # a whole detection kept for its group; the group's fragments were discarded
ORIGIN_UNION = 2  # a group with no whole detection: the union of its fragments
ORIGIN_FRAGMENT = 3  # a fragment that nothing was joined to


def merge_tile_fragments(
    merged: Dict[str, Any],
    tile_idx: torch.Tensor,
    tile_origins: np.ndarray,
    tile_size: Tuple[int, int],
    im_size: Tuple[int, int],
    containment: float = CONTAINMENT,
    edge_tolerance: float = DEFAULT_EDGE_TOLERANCE,
    report_origin: bool = False,
) -> Dict[str, Any]:
    """Combine the detections of each object that overlapping tiles found separately.

    Args:
        merged: one image's result dict, in image coordinates.
        tile_idx: (N,) tile index of each detection.
        tile_origins: (T, 2) top-left (y, x) of each tile; a full grid, every row start with every column start.
        tile_size: (tile_h, tile_w).
        im_size: (im_h, im_w) of the original image, without padding.
        containment: share of a fragment that must lie inside a whole detection to join it.
        edge_tolerance: see ``DEFAULT_EDGE_TOLERANCE``. Decides which leftover fragments may be deleted; joining uses the
            wider ``JOIN_MARGIN``. Without tile overlap it is also the seam band's half-width, so keep it small.
        report_origin: add a ``merge_origin`` code (``ORIGIN_*``) per output detection. Missing when nothing was merged.

    Returns:
        The dict with one detection per object. A group with a whole detection keeps its best-scoring one; a group of
        only fragments becomes their union with the best score, deleted when another tile saw its whole area.
    """
    # no early return for a single detection: it still needs the cut test and the drop rule
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
    touching = _touching_pairs(merged, boxes, cut, grid.seam_band)
    joins = _joins(merged, boxes, cut, tile_idx, grid, containment, touching)
    groups = _split_distinct_whole(merged, boxes, _connected_groups(joins, len(tile_idx)), whole, joins, touching)

    if any(len(g) > 1 for g in groups):
        out, groups = _union_groups(merged, groups, whole, grid)
        rows = _Rows(groups)
        origin = _origin_codes(rows, whole)
        # only groups whose members all reach a seam may be deleted; a whole object near a seam must stay
        droppable = rows.every(at_edge)
    else:  # nothing joined: every group is one detection, in input order
        out = merged
        origin = torch.where(whole, ORIGIN_WHOLE, ORIGIN_FRAGMENT).to(torch.uint8)
        droppable = at_edge
    if report_origin:
        out = {**out, "merge_origin": origin}
    return _drop_fragments_seen_whole(out, droppable, grid)


def instance_boxes(merged: Dict[str, Any]) -> Optional[torch.Tensor]:
    """(N, 4) xyxy box of each detection, taken from boxes, else masks, else segments."""
    boxes = merged.get("boxes")
    if isinstance(boxes, torch.Tensor) and len(boxes):
        return boxes.detach().cpu().float()

    masks = merged.get("masks")
    if isinstance(masks, MaskCrops) and len(masks):
        return masks.boxes.float().cpu()
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

    ``margin`` flags fragments for joining: ``JOIN_MARGIN``, capped at 1/8 of the smallest overlap, never below
    ``tolerance``. Kept small so an object inside the overlap stays whole in at least one tile.
    """

    def __init__(self, origins: np.ndarray, size: Tuple[int, int], im_size: Tuple[int, int], tolerance: float):
        self.origins = np.asarray(origins)
        self.size = size
        self.im_size = im_size
        self.tolerance = tolerance
        overlap = self._overlap()
        self.margin = max(min(JOIN_MARGIN, float(overlap.min()) / 8), tolerance)
        self.seam_band = self._seam_band(overlap)

    def _overlap(self) -> np.ndarray:
        """(2,) px two neighbouring tiles share on each tile axis (y, x); ``inf`` on an axis with no seam to share across."""
        out = []
        for axis in (0, 1):
            starts = np.unique(self.origins[:, axis])
            out.append(self.size[axis] - float(np.diff(starts).min()) if len(starts) > 1 else np.inf)
        return np.array(out)

    def _seam_band(self, overlap: np.ndarray) -> torch.Tensor:
        """(2,) px in (x, y) that widens the area two tiles share to at least ``2 * margin``: the seam band.

        Zero when tiles overlap by more than that; ``margin`` when they do not overlap.
        """
        pad = ((2 * self.margin - overlap) / 2).clip(min=0)
        return torch.tensor(pad[::-1].copy(), dtype=torch.float32)  # (y, x) tile axes -> (x, y) box axes

    def corners(self, tile_idx: torch.Tensor) -> torch.Tensor:
        """(N, 2) top-left (x, y) of the tile each detection came from."""
        return torch.as_tensor(self.origins, dtype=torch.float32, device=tile_idx.device)[tile_idx].flip(1)

    def seams(self) -> Tuple[np.ndarray, np.ndarray]:
        """Seams in image px: x of the vertical ones, y of the horizontal ones."""
        out = []
        for axis in (1, 0):
            starts = np.unique(self.origins[:, axis])
            edges = np.unique(np.concatenate([starts, starts + self.size[axis]]))
            out.append(edges[(edges > 0) & (edges < self.im_size[axis])])
        return out[0], out[1]


class _Rows:
    """Every group's members in one flat run, so a per-group summary is one scatter over that run, not a pass per group."""

    def __init__(self, groups: List[List[int]]):
        self.sizes = torch.tensor([len(g) for g in groups])
        self.members = torch.tensor([m for g in groups for m in g], dtype=torch.long)
        self.gid = torch.repeat_interleave(torch.arange(len(groups)), self.sizes)

    def any(self, flags: torch.Tensor) -> torch.Tensor:
        """(G,) bool: ``flags`` is set on at least one member of the group."""
        return self._fold(flags, "amax")

    def every(self, flags: torch.Tensor) -> torch.Tensor:
        """(G,) bool: ``flags`` is set on every member of the group."""
        return self._fold(flags, "amin")

    def _fold(self, flags: torch.Tensor, reduce: str) -> torch.Tensor:
        n = len(self.sizes)
        return torch.zeros(n, dtype=torch.bool).scatter_reduce_(0, self.gid, flags[self.members], reduce=reduce, include_self=False)


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
    """(N,) bool: the box is within ``tolerance`` px of a seam of its own tile. Edges on the border or in the padding do not count."""
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
    """(L, 2) pairs [j, i]: fragment i and detection j are the same object and are joined. All must hold:

    1. i is a fragment.
    2. j comes from another tile and has the same class.
    3. Their boxes reach ``AGREEMENT_IOU`` inside the area both tiles cover, widened by the seam band.
    4. j is a fragment too, or at least ``containment`` of i lies inside j. Otherwise a short fragment that stops just
       before its seam looks whole, and its group would keep it and discard the longer i.
    """
    if not len(touching):
        return _NO_PAIRS
    # ``touching`` in both directions, keeping the pairs whose second member is the fragment
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
    """(L, 2) index pairs i < j whose boxes intersect once grown by ``pad``, so fragments that only meet at a seam count.

    Shared by the join test and the split. Empty when there are no fragments: nothing can join. The scan is ~12x faster
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
    """(K, 4) box of each detection's shape inside the region ``lo``-``hi``, grown by the seam band and held to the region.

    A shape that misses the region gets a zero-area box. Masks use their pixels in the region, so two halves of an
    object cut at an angle line up at the seam; polygons and boxes use the box.
    """
    inside = torch.minimum(torch.maximum(boxes[idx], lo), hi)
    masks = _mask_crops(merged, boxes)
    if masks is not None:
        clipped = (inside != boxes[idx]).any(dim=1)
        inside[clipped] = masks.boxes_in_regions(idx[clipped], lo[clipped], hi[clipped], lo[clipped]).cpu()
    empty = ((inside[:, 2:] - inside[:, :2]) <= 0).any(dim=1)
    out = torch.minimum(torch.maximum(_grow(inside, grid.seam_band), lo), hi)
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
    """(K,) bool: j[k] and i[k] reach ``AGREEMENT_IOU`` inside the area both tiles see, widened by the seam band.

    Two views of one object nearly match there; a larger nearby object that only contains i does not.
    """
    if not len(j):
        return torch.zeros(0, dtype=torch.bool)
    tile_h, tile_w = grid.size
    origins = grid.corners(tile_idx)
    band = grid.seam_band.to(origins)
    extent = torch.tensor([tile_w, tile_h], dtype=torch.float32, device=origins.device)
    lo = (torch.maximum(origins[j], origins[i]) - band).repeat(1, 2)
    hi = (torch.minimum(origins[j], origins[i]) + extent + band).repeat(1, 2)
    a = _trimmed(merged, boxes, j, lo, hi, grid)
    b = _trimmed(merged, boxes, i, lo, hi, grid)
    inter = box_intersections(a, b)
    area_a = (a[:, 2:] - a[:, :2]).prod(dim=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(dim=1)
    return inter / (area_a + area_b - inter).clamp(min=1e-9) >= AGREEMENT_IOU


def _containment(merged: Dict[str, Any], boxes: torch.Tensor, j: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """(K,) share of detection i[k] that lies inside detection j[k], for the given pairs only. Uses masks, else boxes."""
    masks = _mask_crops(merged, boxes)
    if masks is not None:
        areas = masks.areas()[i]
        return torch.where(areas > 0, masks.intersections(j, i) / areas.clamp(min=1e-9), torch.zeros_like(areas)).cpu()

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
    """Split a group so each separate whole object gets its own group; otherwise one wrong join deletes a real object.

    Whole detections are one object when ``SAME_OBJECT_CONTAINMENT`` of either lies inside the other, directly or
    through a chain. Each fragment follows the object it is joined to, through other fragments if needed; with
    several, the nearest wins, then the first.
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
        # breadth-first from every object at once, stepping only through fragments
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

    Filtered from the global ``touching`` scan; a per-group cross product can exhaust memory.
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
    merged: Dict[str, Any], groups: List[List[int]], whole: torch.Tensor, grid: "_Grid"
) -> Tuple[Dict[str, Any], List[List[int]]]:
    """Turn each group into one output detection, keeping input order; also returns the groups in output order.

    A group with whole detections keeps its best-scoring one. A group of only fragments becomes their union (box and
    mask), with the mask closed across seams within the seam band to bridge fragments that stop short.
    """
    gap_xy = [int(np.ceil(v)) for v in grid.seam_band.tolist()]
    seams = grid.seams()
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
            if any(gap_xy):
                data = _close_across_seams(data, crop_boxes, gap_xy, seams)
            out["masks"].set_runs(torch.tensor(rows, dtype=torch.long), data, crop_boxes)

    return out, groups


def _host_crops(data: torch.Tensor, boxes: torch.Tensor) -> List[np.ndarray]:
    """Packed mask crops as (h, w) uint8 arrays, in one transfer."""
    host = data.cpu().numpy().astype(np.uint8)
    out, at = [], 0
    for x0, y0, x1, y1 in boxes.tolist():
        w, h = x1 - x0, y1 - y0
        out.append(host[at : at + w * h].reshape(h, w))
        at += w * h
    return out


def _close_across_seams(data: torch.Tensor, boxes: torch.Tensor, gap: List[int], seams: Tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
    """Packed mask crops closed by a ``gap`` (x, y) rectangle, keeping the fill only near seams.

    A closing never reaches past the crop's box, so the boxes and the packed layout stay valid.
    """
    import cv2

    gx, gy = gap
    # a rectangle, not an ellipse: seams are axis-aligned, and cv2 shrinks an ellipse one pixel thick to a point
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * gx + 1, 2 * gy + 1))
    out = []
    for crop, (x0, y0, _, _) in zip(_host_crops(data, boxes), boxes.tolist()):
        h, w = crop.shape
        if crop.size:
            # zero padding so the dilation is not cut at the crop edge
            closed = cv2.morphologyEx(np.pad(crop, ((gy, gy), (gx, gx))), cv2.MORPH_CLOSE, kernel)[gy : gy + h, gx : gx + w]
            # fragment gaps are at seams; elsewhere a narrow dent or hole is the model's to keep
            band = _near_seam(x0 + np.arange(w), seams[0], gx)[None, :] | _near_seam(y0 + np.arange(h), seams[1], gy)[:, None]
            crop = crop | (closed & band)
        out.append(crop.reshape(-1))
    return torch.from_numpy(np.concatenate(out) if out else np.zeros(0, np.uint8)).to(data.device, torch.bool)


def _near_seam(px: np.ndarray, seams: np.ndarray, gap: int) -> np.ndarray:
    """(P,) bool: pixel index within ``gap`` of a seam, on either side."""
    if not gap or not len(seams):
        return np.zeros(len(px), dtype=bool)
    return ((px[:, None] >= seams - gap) & (px[:, None] < seams + gap)).any(axis=1)


def trace_crops(crops: MaskCrops) -> List[torch.Tensor]:
    """(M, 2) outline of each mask in image px, on the masks' device; see ``mask_segments.masks_to_segments``."""
    return [
        torch.as_tensor(trace_mask(crop) + np.array([x0, y0], dtype=np.float32), device=crops.device)
        for crop, (x0, y0, _, _) in zip(_host_crops(crops.data, crops.boxes), crops.boxes.tolist())
    ]


def _seen_whole_by_a_tile(boxes: torch.Tensor, grid: "_Grid") -> torch.Tensor:
    """(F,) bool: some tile holds the box more than ``grid.tolerance`` px from each of its seams.

    The tiles form a full grid, so each axis is checked on its own.
    """
    out = torch.ones(len(boxes), dtype=torch.bool)
    for axis, lo, hi in ((1, 0, 2), (0, 1, 3)):  # tile axis (y, x) -> box columns
        starts = torch.as_tensor(np.unique(grid.origins[:, axis]), dtype=torch.float32)[None, :]
        ends = starts + grid.size[axis]
        a, b = boxes[:, lo, None], boxes[:, hi, None]
        cut = ((starts > 0) & (a <= starts + grid.tolerance)) | ((ends < grid.im_size[axis]) & (b >= ends - grid.tolerance))
        out &= ((a >= starts) & (b <= ends) & ~cut).any(dim=1)
    return out


def _drop_fragments_seen_whole(out: Dict[str, Any], droppable: torch.Tensor, grid: "_Grid") -> Dict[str, Any]:
    """Delete leftover fragments whose whole area another tile saw. Output r is deleted when both hold:

    1. ``droppable[r]``: every detection r was built from reaches a seam within ``grid.tolerance``.
    2. Some tile holds r's box more than that tolerance from its seams.

    That tile would have detected a real object there whole, so r is a duplicate or false. Only position is checked:
    boxes of small fragments at a seam are too inaccurate for an overlap test.
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
