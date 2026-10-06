"""Compare two models' detections on one image, independent of output order."""

import numpy as np


def _to_numpy(x):
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x)


def box_iou(a, b):
    """Pairwise IoU of xyxy boxes a (N,4) and b (M,4)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def _match_one_way(src, dst, min_score, score_tol, min_box_iou, min_mask_iou, max_point_dist, label):
    iou = box_iou(src["boxes"], dst["boxes"]) if len(src["boxes"]) and len(dst["boxes"]) else np.zeros((len(src["boxes"]), 0))
    used = set()
    for i in np.argsort(-src["scores"]):
        if src["scores"][i] < min_score:
            continue
        candidates = [j for j in range(len(dst["boxes"])) if j not in used and dst["classes"][j] == src["classes"][i]]
        assert candidates, f"{label}: {src['classes'][i]} ({src['scores'][i]:.3f}) has no match"
        j = max(candidates, key=lambda j: iou[i, j])
        used.add(j)
        what = f"{label}: {src['classes'][i]} ({src['scores'][i]:.3f})"
        assert iou[i, j] >= min_box_iou, f"{what}: box IoU {iou[i, j]:.3f} < {min_box_iou}"
        score_diff = abs(src["scores"][i] - dst["scores"][j])
        assert score_diff <= score_tol, f"{what}: score differs by {score_diff:.3f}"
        if min_mask_iou is not None:
            a, b = src["masks"][i] > 0, dst["masks"][j] > 0
            mask_iou = (a & b).sum() / max((a | b).sum(), 1)
            assert mask_iou >= min_mask_iou, f"{what}: mask IoU {mask_iou:.3f} < {min_mask_iou}"
        if max_point_dist is not None:
            visible = src["points"][i][:, 2] >= 0.5
            dist = np.linalg.norm(src["points"][i][visible, :2] - dst["points"][j][visible, :2], axis=-1)
            assert (dist <= max_point_dist).all(), f"{what}: a visible keypoint moved {dist.max():.1f} px > {max_point_dist}"


def assert_detections_match(ref, out, min_score, score_tol, min_box_iou, min_mask_iou=None, max_point_dist=None, label=""):
    """Every detection scoring >= min_score on either side has a same-class match on the other side.

    ``ref`` and ``out`` are one image's outputs with ``boxes`` (xyxy, or (N,4,2) oriented corners, compared by their
    bounding rectangles), ``scores`` and ``classes``, plus ``masks`` when ``min_mask_iou`` is set and ``points`` (N,K,3)
    when ``max_point_dist`` is set; only keypoints visible (>= 0.5) on the side being matched are compared. Predict both at
    ``min_score - score_tol`` so a score that crosses ``min_score`` within tolerance still finds its match.
    """
    keys = (
        ["boxes", "scores", "classes"]
        + (["masks"] if min_mask_iou is not None else [])
        + (["points"] if max_point_dist is not None else [])
    )
    ref, out = ({k: _to_numpy(d[k]) for k in keys} for d in (ref, out))
    for d in (ref, out):
        if d["boxes"].ndim == 3:
            d["boxes"] = np.concatenate([d["boxes"].min(1), d["boxes"].max(1)], axis=-1)
    args = (min_score, score_tol, min_box_iou, min_mask_iou, max_point_dist)
    _match_one_way(ref, out, *args, f"{label} ref->out")
    _match_one_way(out, ref, *args, f"{label} out->ref")
