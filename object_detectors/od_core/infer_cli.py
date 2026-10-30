"""Command-line flags and JSON output shared by the object detector ``infer.py`` scripts."""

import argparse
from typing import Any, Dict, List, Optional

from lmi_common import infer_cli as shared
from lmi_common.infer_cli import TILE_PLOT_DIR, predict_tiled  # noqa: F401  re-exported for the detector scripts

JSON_NAME = "predictions.json"

# (merge_origin code, legend text, RGB color); codes mirror tile_merge.ORIGIN_*, not imported since it pulls in torch
MERGE_ORIGIN_LEGEND = [
    (0, "whole: seen whole in a tile", (0, 200, 0)),
    (1, "whole joined with cut pieces", (0, 160, 255)),
    (2, "cut pieces combined", (255, 140, 0)),
    (3, "lone cut piece", (255, 0, 255)),
]


def add_infer_args(
    parser: argparse.ArgumentParser,
    confidence: float,
    weights: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
) -> None:
    """Add the flags every detector script takes. A path left as None becomes a required flag."""
    shared.add_infer_args(
        parser, confidence, weights, input, output, tile_plot="the tile grid and each detection colored by how merging built it"
    )
    parser.add_argument("--json", action="store_true", help=f"[optional] save the predictions to {JSON_NAME} in the output folder")
    parser.add_argument(
        "--no_label", "--no-label", action="store_true", help="[optional] do not draw class names and scores on the output images"
    )
    parser.add_argument(
        "--line_thickness",
        "--line-thickness",
        type=int,
        help="[optional] px width of the drawn boxes. By default it grows with the image size",
    )


def check_infer_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate the flags, exiting through ``parser.error`` on bad input.

    Sets ``args.image_size`` to [h, w] or None, and ``args.tile_step`` to a tile step or None. The tile step adds a
    ``merge_origin`` code per detection, which ``plot_tile_merge`` colors by.
    """
    if args.line_thickness is not None and args.line_thickness <= 0:
        parser.error(f"--line_thickness takes a positive int, got {args.line_thickness}")
    shared.check_infer_args(parser, args, report_merge_origin=True)


def plot_tile_merge(image, outputs: Dict, tile_boxes, hide_label: bool = False, line_thickness: Optional[int] = None) -> Any:
    """Copy of one RGB image with the tile grid and each detection colored by its ``merge_origin``, with a color legend in a
    panel to the right of the image.

    Args:
        image: (H, W, 3) RGB image.
        outputs: that image's results, with ``boxes`` (N, 4) xyxy, ``classes`` and ``merge_origin``. A missing
            ``merge_origin`` means no merging ran, so every detection counts as whole.
        tile_boxes: (T, 4) xyxy box of each tile, as returned by ``predict_tiled``.
        hide_label: do not write the class name on each detection.
        line_thickness: px width of the boxes; None grows it with the image size.
    """
    import cv2
    import numpy as np

    from lmi_utils.label_utils.plot_utils import plot_one_box, plot_tile_grid

    im = np.ascontiguousarray(image).copy()
    plot_tile_grid(tile_boxes, im, line_thickness=2)
    boxes = np.asarray(outputs.get("boxes", []), dtype=float).reshape(-1, 4)
    classes = outputs.get("classes", [])
    codes = outputs.get("merge_origin")
    codes = np.zeros(len(boxes), dtype=int) if codes is None else np.asarray(codes, dtype=int)
    colors = {code: color for code, _, color in MERGE_ORIGIN_LEGEND}
    for box, cls, code in zip(boxes, classes, codes):
        plot_one_box(box, im, color=colors[int(code)], label=None if hide_label else str(cls), line_thickness=line_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, max(im.shape[:2]) / 1600)
    thickness = max(1, round(2 * scale))
    line_h = round(40 * scale)
    margin = line_h // 2
    text_w = max(cv2.getTextSize(text, font, scale, thickness)[0][0] for _, text, _ in MERGE_ORIGIN_LEGEND)
    h, w = im.shape[:2]
    canvas = np.zeros((max(h, line_h * len(MERGE_ORIGIN_LEGEND) + margin), w + text_w + 2 * margin, 3), dtype=im.dtype)
    canvas[:h, :w] = im
    for k, (_, text, color) in enumerate(MERGE_ORIGIN_LEGEND):
        cv2.putText(canvas, text, (w + margin, line_h * (k + 1)), font, scale, color, thickness, cv2.LINE_AA)
    return canvas


def save_tile_plot(
    output_dir: str, name: str, image, outputs: Dict, tile_boxes, hide_label: bool = False, line_thickness: Optional[int] = None
) -> None:
    """Write ``plot_tile_merge`` for one RGB image to ``output_dir/TILE_PLOT_DIR/name``."""
    plot = plot_tile_merge(image, outputs, tile_boxes, hide_label=hide_label, line_thickness=line_thickness)
    shared.write_tile_plot(output_dir, name, plot)


class PredictionsJson:
    """Collect per-image predictions and save them in the LMI dataset json format."""

    def __init__(self, class_names, keypoint_names: Optional[Dict[str, List[str]]] = None):
        """``keypoint_names`` maps a class to its keypoint name per model slot; without it a keypoint is labeled with its box's class."""
        self.class_names = [str(n) for n in class_names]
        self.keypoint_names = {str(c): [str(n) for n in names] for c, names in (keypoint_names or {}).items()}
        self.files = []
        self.annotation_type = None
        self._next_id = 0

    def add(self, path: str, height: int, width: int, outputs: Dict) -> None:
        """Add one image. ``outputs`` holds that image's boxes, scores and classes, and optional masks and points, in image coordinates.

        Masks become bitmask predictions, 4-corner boxes become rotated boxes, and each keypoint links to its box.
        """
        from lmi_utils.dataset_utils.representations import FileAnnotations

        predictions = []
        classes, scores, boxes = outputs["classes"], outputs["scores"], outputs["boxes"]
        masks, points = outputs.get("masks"), outputs.get("points")
        for i in range(len(classes)):
            mask = masks[i] if masks is not None and len(masks) else None
            pts = points[i] if points is not None and len(points) else None
            predictions += self._annotations(str(classes[i]), float(scores[i]), boxes[i], mask, pts)
        self.files.append(FileAnnotations(id=path, path=path, height=int(height), width=int(width), predictions=predictions))

    def save(self, path: str) -> None:
        from lmi_utils.dataset_utils.representations import AnnotationType, Dataset, Label

        kind = self.annotation_type or AnnotationType.BOX
        predictions = [a for f in self.files for a in f.predictions]
        seen = {a.label_id for a in predictions if a.type != AnnotationType.KEYPOINT}
        names = self.class_names + sorted(seen - set(self.class_names))
        labels = [Label(id=n, annotation_type=kind, keypoints=self.keypoint_names.get(n)) for n in names]
        keypoint_ids = {n for layout in self.keypoint_names.values() for n in layout}
        keypoint_ids |= {a.label_id for a in predictions if a.type == AnnotationType.KEYPOINT}
        labels += [Label(id=n, annotation_type=AnnotationType.KEYPOINT) for n in sorted(keypoint_ids - set(names))]
        Dataset(labels=labels, files=self.files).save(path)

    def _annotations(self, label: str, score: float, box, mask, points) -> List:
        import numpy as np

        from lmi_utils.dataset_utils.representations import Annotation, AnnotationType, Box, KeypointAnnotation, Mask, Point2d, Polygon

        box = np.asarray(box, dtype=float)
        if mask is not None:
            kind, value = AnnotationType.MASK, Mask(np.asarray(mask) > 0)
        elif box.size == 8:
            kind, value = AnnotationType.BOX, Polygon(points=box.reshape(4, 2)).to_rbox()
        else:
            kind, value = AnnotationType.BOX, Box(*box[:4], angle=0)
        self.annotation_type = self.annotation_type or kind

        box_id = self._new_id()
        out = [Annotation(id=box_id, label_id=label, type=kind, value=value, confidence=score)]
        slot_names = self.keypoint_names.get(label)
        for slot, pt in enumerate([] if points is None else np.asarray(points, dtype=float)):
            if slot_names is not None and slot >= len(slot_names):
                break  # slots past the class's layout are padding
            name = label if slot_names is None else slot_names[slot]
            # the model's third column is a confidence, not a labeled visibility flag, so it is not written
            out.append(KeypointAnnotation(id=self._new_id(), label_id=name, value=Point2d(pt[0], pt[1]), bounding_box_id=box_id))
        return out

    def _new_id(self) -> str:
        self._next_id += 1
        return str(self._next_id - 1)
