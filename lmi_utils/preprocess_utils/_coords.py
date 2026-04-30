"""Shared helpers for coord-field traversal in geometric Operations.

Every new geometric op (rotate, crop, perspective, ...) needs to walk the same
four fields with the same edge cases: empty tensors, OBB vs xyxy boxes, the
segments-is-a-list quirk, and keypoint visibility splitting. Centralizing that
walk here keeps each Operation focused on its actual transform.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch


def split_visibility(points: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split (N, K, 3) keypoints into (xy, visibility). (N, K, 2) returns (points, None)."""
    if points.shape[-1] == 3:
        return points[..., :2], points[..., -1]
    return points, None


def join_visibility(xy: torch.Tensor, visibility: Optional[torch.Tensor]) -> torch.Tensor:
    """Inverse of split_visibility."""
    if visibility is None:
        return xy
    return torch.cat((xy, visibility.unsqueeze(-1)), dim=-1)


def apply_coord_transform(
    result: Dict[str, Any],
    *,
    xy_fn: Callable[[torch.Tensor], torch.Tensor],
    box_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Apply a geometric transform to every coord field of one result dict.

    Args:
        xy_fn:   (N, 2) -> (N, 2). Required. Used for segments and points,
                 and for boxes when `box_fn` is not provided (via corner flatten).
        box_fn:  Optional override for boxes. Use when `xy_fn` is not safe to
                 apply to flattened corner pairs (e.g. flip needs corner swap)
                 or when the underlying util handles (N, 4) natively.
        mask_fn: Optional masks transform. Default: leave masks untouched.

    Empty / missing fields are skipped. The input dict is not mutated.
    """
    out = dict(result)

    boxes = result.get("boxes")
    if boxes is not None and len(boxes):
        if box_fn is not None:
            out["boxes"] = box_fn(boxes)
        elif boxes.ndim == 3:  # OBB (N, 4, 2)
            n = boxes.shape[0]
            out["boxes"] = xy_fn(boxes.reshape(-1, 2)).reshape(n, 4, 2)
        else:  # xyxy (N, 4) — flatten to corner pairs
            out["boxes"] = xy_fn(boxes.reshape(-1, 2)).reshape(boxes.shape)

    segs = result.get("segments")
    if segs is not None and len(segs):
        out["segments"] = [xy_fn(s) if len(s) else s for s in segs]

    pts = result.get("points")
    if pts is not None and len(pts):
        xy, vis = split_visibility(pts)
        n, k, _ = xy.shape
        new_xy = xy_fn(xy.reshape(n * k, 2)).reshape(n, k, 2)
        out["points"] = join_visibility(new_xy, vis)

    masks = result.get("masks")
    if masks is not None and len(masks) and mask_fn is not None:
        out["masks"] = mask_fn(masks)

    return out
