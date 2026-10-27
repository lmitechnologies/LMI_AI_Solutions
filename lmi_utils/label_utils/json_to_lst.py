import argparse
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from lmi_utils.dataset_utils.representations import Annotation, Box, Dataset, Mask, Point2d, Polygon, rle2mask
from lmi_utils.label_utils.lst_brush import encode_mask

logger = logging.getLogger(__name__)

OUT_NAME = "lst.json"
XML_NAME = "labeling_interface.xml"
# must match the control tag names in the Label Studio labeling interface; "label" keeps csv_to_lst projects working
CONTROL_NAMES = {
    "rectanglelabels": ("label", "RectangleLabels"),
    "polygonlabels": ("polygon", "PolygonLabels"),
    "brushlabels": ("brush", "BrushLabels"),
    "keypointlabels": ("keypoint", "KeyPointLabels"),
}
CONTROL_ATTRS = {"polygonlabels": {"strokeWidth": "3", "pointSize": "small", "opacity": "0.9"}}


def annotation_to_lst(annot: Annotation, height: int, width: int, is_pred: bool) -> dict:
    """Convert one annotation to a Label Studio result dict, with coordinates in percent of the image size."""
    shape = annot.value
    if isinstance(shape, Box):
        result_type = "rectanglelabels"
        value = {
            "x": shape.x_min / width * 100,
            "y": shape.y_min / height * 100,
            "width": (shape.x_max - shape.x_min) / width * 100,
            "height": (shape.y_max - shape.y_min) / height * 100,
            "rotation": shape.angle or 0,
        }
    elif isinstance(shape, Polygon):
        result_type = "polygonlabels"
        value = {"points": [[x / width * 100, y / height * 100] for x, y in shape.points]}
    elif isinstance(shape, Mask):
        result_type = "brushlabels"
        value = {"format": "rle", "rle": encode_mask(rle2mask(shape.mask, h=height, w=width))}
    elif isinstance(shape, Point2d):
        result_type = "keypointlabels"
        value = {"x": shape.x / width * 100, "y": shape.y / height * 100, "width": 0.5}
    else:
        raise ValueError(f"unsupported shape: {type(shape).__name__}")

    value[result_type] = [annot.label_id]
    if is_pred:
        value["score"] = 1.0 if annot.confidence is None else annot.confidence
    return {
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": value,
        "from_name": CONTROL_NAMES[result_type][0],
        "to_name": "image",
        "type": result_type,
    }


def to_image_root(path: str) -> str:
    """Return the prefix joined with each image path; a local absolute path becomes a Label Studio local-files URL.

    The first folder of a local path is taken as LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT.
    """
    if path.startswith("gs://"):
        return path.rstrip("/") + "/"
    if not path.startswith("/"):
        raise ValueError("The local storage path must be absolute path starting with /")
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise ValueError("The local storage path must include the document root folder")
    logger.info(f"Assume that LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT is /{parts[0]}")
    return "/data/local-files/?d=" + "".join(p + "/" for p in parts[1:])


def dataset_to_lst(dataset: Dataset, image_root: str) -> List[dict]:
    """Convert a dataset to Label Studio tasks, with annotations and predictions; image_root comes from to_image_root."""
    tasks = []
    for file in sorted(dataset.files, key=lambda f: f.path):
        if file.height is None or file.width is None:
            raise ValueError(f"missing image size for {file.path}")
        task = {"data": {"image": image_root + Path(file.path).as_posix()}}
        if file.annotations:
            results = [annotation_to_lst(a, file.height, file.width, is_pred=False) for a in file.annotations]
            task["annotations"] = [{"result": results}]
        if file.predictions:
            results = [annotation_to_lst(a, file.height, file.width, is_pred=True) for a in file.predictions]
            task["predictions"] = [{"model_version": "prediction", "result": results}]
        tasks.append(task)
    return tasks


def labeling_interface(tasks: List[dict]) -> str:
    """Build a labeling interface XML with one control tag per result type found in the tasks."""
    classes = {}
    for task in tasks:
        for key in ("annotations", "predictions"):
            for result in (r for item in task.get(key, []) for r in item["result"]):
                classes.setdefault(result["type"], set()).update(result["value"][result["type"]])

    root = ET.Element("View")
    ET.SubElement(root, "Image", name="image", value="$image", zoom="true")
    for result_type, (name, tag) in CONTROL_NAMES.items():
        if result_type in classes:
            control = ET.SubElement(root, tag, name=name, toName="image", **CONTROL_ATTRS.get(result_type, {}))
            for label in sorted(classes[result_type]):
                ET.SubElement(control, "Label", value=label)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Convert json annotations to Label Studio import format")
    ap.add_argument("--json", "-j", required=True, type=Path, help="path to the json annotations")
    ap.add_argument(
        "--image_root",
        required=True,
        help="where Label Studio finds the images: a gs:// path or a local absolute path; joined with each file path in the json",
    )
    ap.add_argument("--out_dir", "-o", required=True, type=Path, help="the output directory")
    args = ap.parse_args()

    dataset = Dataset.load(str(args.json))
    tasks = dataset_to_lst(dataset, to_image_root(args.image_root))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / OUT_NAME, "w") as f:
        json.dump(tasks, f, indent=4)
    with open(args.out_dir / XML_NAME, "w") as f:
        f.write(labeling_interface(tasks))
    logger.info(f"wrote {len(tasks)} tasks to {args.out_dir / OUT_NAME} and the labeling interface to {args.out_dir / XML_NAME}")


if __name__ == "__main__":
    main()
