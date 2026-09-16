"""Precision, recall and F1 of object detection predictions against ground truth, both in the LMI dataset json format.

Ground truth comes from the ``annotations`` of a labels json; predictions from the ``predictions`` of the json that
``infer.py --json`` writes. Each prediction, highest score first, matches the unmatched ground truth object of the
same class with the highest IoU, if that IoU reaches the threshold. Results use box IoU, plus mask IoU when every
ground truth object and every prediction is a polygon or mask.

Example:
    python -m lmi_utils.eval_utils.eval_od --labels data/labels.json --preds outputs/predictions.json --iou 0.5 -o outputs/eval
"""

import argparse
import collections
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
from pycocotools import mask as mask_utils

from lmi_utils.dataset_utils.representations import Annotation, AnnotationType, Dataset, FileAnnotations

logger = logging.getLogger(__name__)

SHAPE_TYPES = (AnnotationType.BOX, AnnotationType.POLYGON, AnnotationType.MASK)
ALL = "all"
CURVE_NAMES = ("precision", "recall", "f1")


class Matches:
    """The score and match result of every prediction of one class, plus its ground truth count."""

    def __init__(self):
        self.scores = np.zeros(0)
        self.is_tp = np.zeros(0, dtype=bool)
        self.num_gt = 0

    def add(self, scores, is_tp, num_gt: int) -> None:
        self.scores = np.concatenate([self.scores, scores])
        self.is_tp = np.concatenate([self.is_tp, is_tp])
        self.num_gt += num_gt

    def counts(self, confidences) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tp, fp, fn) when keeping predictions scored at or above each confidence."""
        order = np.argsort(self.scores)
        scores, cum_tp = self.scores[order], np.concatenate([[0], np.cumsum(self.is_tp[order])])
        first_kept = np.searchsorted(scores, np.asarray(confidences, dtype=float), side="left")
        tp = cum_tp[-1] - cum_tp[first_kept]
        fp = (len(scores) - first_kept) - tp
        return tp, fp, self.num_gt - tp


def to_coco_geometry(annotation: Annotation, h: int, w: int, use_mask: bool):
    """COCO RLE when ``use_mask``, else the [x, y, w, h] axis-aligned box that encloses the shape."""
    value = annotation.value
    if use_mask:
        mask = value if annotation.type == AnnotationType.MASK else value.to_mask(h=h, w=w)
        # the stored string is COCO RLE of the transposed mask; IoU is the same either way, so it needs no decoding
        return {"size": [w, h], "counts": mask.mask}
    if annotation.type == AnnotationType.MASK:
        y, x, box_h, box_w = mask_utils.toBbox({"size": [w, h], "counts": value.mask}).tolist()
        return [x, y, max(box_w - 1, 0.0), max(box_h - 1, 0.0)]  # Mask.to_box ends at the last pixel index, not after it
    if annotation.type == AnnotationType.BOX:
        return value.to_coco()
    return value.to_box().to_coco()


def iou_matrix(preds: List, gts: List, use_mask: bool) -> np.ndarray:
    """(P, G) IoU between predictions and ground truth geometry from ``to_coco_geometry``."""
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    if not use_mask:
        preds, gts = np.asarray(preds, dtype=float), np.asarray(gts, dtype=float)
    return np.asarray(mask_utils.iou(preds, gts, [0] * len(gts))).reshape(len(preds), len(gts))


def match_file(gt_file: FileAnnotations, pred_file: FileAnnotations, iou_thres: float, use_mask: bool) -> Dict[str, Tuple]:
    """Match one image's predictions to its ground truth: class -> (scores, is_tp, number of ground truth objects).

    Matching goes from the highest score down, so dropping low scores later never changes a higher score's match.
    """
    h, w = gt_file.height, gt_file.width
    gts = collections.defaultdict(list)
    for a in gt_file.annotations:
        if a.type in SHAPE_TYPES:
            gts[a.label_id].append(to_coco_geometry(a, h, w, use_mask))
    preds = collections.defaultdict(list)
    for a in pred_file.predictions:
        if a.type in SHAPE_TYPES:
            preds[a.label_id].append((1.0 if a.confidence is None else a.confidence, to_coco_geometry(a, h, w, use_mask)))

    out = {}
    for c in set(gts) | set(preds):
        ranked = sorted(preds[c], key=lambda p: -p[0])
        ious = iou_matrix([g for _, g in ranked], gts[c], use_mask)
        matched = np.zeros(len(gts[c]), dtype=bool)
        is_tp = np.zeros(len(ranked), dtype=bool)
        for k, row in enumerate(ious):
            row = np.where(matched, -1.0, row)
            if len(row) and row.max() >= iou_thres:
                matched[row.argmax()] = is_tp[k] = True
        out[c] = (np.array([s for s, _ in ranked]), is_tp, len(gts[c]))
    return out


def pair_files(labels: Dataset, preds: Dataset) -> List[Tuple[FileAnnotations, FileAnnotations]]:
    """Pair each labeled image with its predictions by path, falling back to the file name."""
    by_path = {os.path.normpath(f.path): f for f in preds.files}
    by_name = collections.defaultdict(list)
    for f in preds.files:
        by_name[os.path.basename(f.path)].append(f)

    pairs, missing = [], []
    for gt in labels.files:
        pred = by_path.get(os.path.normpath(gt.path))
        if pred is None and len(by_name[os.path.basename(gt.path)]) == 1:
            pred = by_name[os.path.basename(gt.path)][0]
        if pred is None:
            missing.append(gt.path)
        else:
            pairs.append((gt, pred))
    if missing:
        logger.warning(f"skipped {len(missing)} labeled images with no predictions, e.g. {missing[:3]}")
    if not pairs:
        raise ValueError("no labeled image matches a prediction file path")
    unlabeled = len(preds.files) - len({id(p) for _, p in pairs})
    if unlabeled:
        logger.warning(f"skipped {unlabeled} predicted images that are not in the labels")
    return pairs


def has_masks(pairs: List[Tuple[FileAnnotations, FileAnnotations]]) -> bool:
    """Whether every ground truth object and every prediction is a polygon or mask, with at least one of each.

    Warns when boxes and masks are mixed, since mask IoU is then skipped.
    """

    def count(annotations):
        types = collections.Counter(a.type for a in annotations)
        return types[AnnotationType.BOX], types[AnnotationType.POLYGON] + types[AnnotationType.MASK]

    gt_boxes, gt_masks = count(a for gt, _ in pairs for a in gt.annotations)
    pred_boxes, pred_masks = count(a for _, pred in pairs for a in pred.predictions)
    if gt_boxes == pred_boxes == 0:
        return gt_masks > 0 and pred_masks > 0
    if gt_masks or pred_masks:
        logger.warning(
            f"boxes and masks are mixed: the labels have {gt_boxes} boxes and {gt_masks} polygons or masks, the predictions have "
            f"{pred_boxes} boxes and {pred_masks} polygons or masks. Mask IoU needs a polygon or mask for every label and every "
            "prediction, so only box IoU is reported"
        )
    return False


def match(pairs: List[Tuple[FileAnnotations, FileAnnotations]], iou_thres=0.5, use_mask=False) -> Dict[str, Matches]:
    """Matches per class and over all classes (``all``) for the image pairs from ``pair_files``."""
    matches = collections.defaultdict(Matches)
    for gt, pred in pairs:
        for c, result in match_file(gt, pred, iou_thres, use_mask).items():
            matches[c].add(*result)
            matches[ALL].add(*result)
    return dict(matches)


def curves(matches: Dict[str, Matches], confidences) -> Dict[str, Dict[str, np.ndarray]]:
    """tp, fp, fn, precision, recall and f1 per class at each confidence. A precision or recall with a zero denominator is NaN."""
    out = {}
    for c, m in matches.items():
        tp, fp, fn = m.counts(confidences)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
            r = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
            f1 = np.where(tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        out[c] = dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)
    return out


def metrics_at(matches: Dict[str, Matches], confidence: float) -> Dict[str, Dict[str, float]]:
    """The ``curves`` values at one confidence."""
    return {c: {k: v[0].item() for k, v in m.items()} for c, m in curves(matches, [confidence]).items()}


def evaluate(labels: Dataset, preds: Dataset, iou_thres=0.5, conf_thres=0.0, use_mask=False) -> Dict[str, Dict[str, float]]:
    """Metrics per class and over all classes (``all``) at one confidence, each a dict of tp, fp, fn, precision, recall, f1."""
    return metrics_at(match(pair_files(labels, preds), iou_thres, use_mask), conf_thres)


def plot_curves(confidences, metrics: Dict[str, Dict[str, np.ndarray]], output_dir: str, iou_thres: float, iou_type: str) -> None:
    """Save ``<iou_type>_<name>_curve.png`` for precision, recall and f1 against confidence, one line per class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(c for c in metrics if c != ALL)
    for name in CURVE_NAMES:
        fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
        if len(classes) < 20:  # a legend of more classes hides the plot
            for c in classes:
                ax.plot(confidences, metrics[c][name], linewidth=1, label=c, zorder=3)
        all_values = metrics[ALL][name]
        label = f"all classes @ {iou_type} iou={iou_thres}"
        if name == "f1" and not np.isnan(all_values).all():
            best = np.nanargmax(all_values)
            label += f": {all_values[best]:.3f} at {confidences[best]:.3f}"
        ax.plot(confidences, all_values, linewidth=3, color="blue", label=label)
        ax.set(xlabel="Confidence", ylabel=name.capitalize() if name != "f1" else "F1", xlim=(0, 1), ylim=(0, 1.01))
        ax.grid(alpha=0.3)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.savefig(os.path.join(output_dir, f"{iou_type}_{name}_curve.png"), dpi=250)
        plt.close(fig)


def format_table(metrics: Dict[str, Dict[str, float]]) -> str:
    width = max(len("class"), *(len(c) for c in metrics))
    lines = [f"{'class':<{width}} {'tp':>7} {'fp':>7} {'fn':>7} {'precision':>10} {'recall':>8} {'f1':>8}"]
    for c in sorted(metrics, key=lambda c: (c == ALL, c)):
        m = metrics[c]
        lines.append(f"{c:<{width}} {m['tp']:>7} {m['fp']:>7} {m['fn']:>7} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", required=True, help="the ground truth json")
    parser.add_argument("--preds", required=True, help="the predictions json, e.g. predictions.json from infer.py --json")
    parser.add_argument("--iou", type=float, default=0.5, help="[optional] the IoU a match needs, default=0.5")
    parser.add_argument("-c", "--confidence", type=float, default=0.0, help="[optional] ignore predictions scored below this, default=0.0")
    parser.add_argument("-o", "--output", help="[optional] save the precision, recall and f1 curves against confidence to this folder")
    args = parser.parse_args()

    pairs = pair_files(Dataset.load(args.labels), Dataset.load(args.preds))
    confidences = np.linspace(0, 1, 1001)
    for iou_type in ("box", "mask") if has_masks(pairs) else ("box",):
        matches = match(pairs, args.iou, use_mask=iou_type == "mask")
        print(f"\n{iou_type} iou >= {args.iou}, confidence >= {args.confidence}")
        print(format_table(metrics_at(matches, args.confidence)))

        over_conf = curves(matches, confidences)
        f1 = over_conf[ALL]["f1"]
        if not np.isnan(f1).all():
            best = np.nanargmax(f1)
            print(f"best {iou_type} f1 over all classes: {f1[best]:.4f} at confidence {confidences[best]:.3f}")
        if args.output:
            plot_curves(confidences, over_conf, args.output, args.iou, iou_type)
    if args.output:
        print(f"\nsaved the curves to {args.output}")
