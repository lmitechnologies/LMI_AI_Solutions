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


def _match_one_way(src, dst, min_score, score_tol, min_box_iou, min_mask_iou, label):
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


def assert_detections_match(ref, out, min_score, score_tol, min_box_iou, min_mask_iou=None, label=""):
    """Every detection scoring >= min_score on either side has a same-class match on the other side.

    ``ref`` and ``out`` are one image's outputs with ``boxes`` (xyxy), ``scores`` and ``classes``, plus ``masks`` when
    ``min_mask_iou`` is set. Predict both at ``min_score - score_tol`` so a score that crosses ``min_score`` within
    tolerance still finds its match.
    """
    keys = ["boxes", "scores", "classes"] + (["masks"] if min_mask_iou is not None else [])
    ref, out = ({k: _to_numpy(d[k]) for k in keys} for d in (ref, out))
    _match_one_way(ref, out, min_score, score_tol, min_box_iou, min_mask_iou, f"{label} ref->out")
    _match_one_way(out, ref, min_score, score_tol, min_box_iou, min_mask_iou, f"{label} out->ref")
