import argparse
import json
import logging
from pathlib import Path
from typing import List

from lmi_utils.dataset_utils.representations import Annotation, Box, Dataset, Mask, Point2d, Polygon, rle2mask
from lmi_utils.label_utils.lst_brush import encode_mask
from lmi_utils.label_utils.lst_project import CONTROL_NAMES, XML_NAME, labeling_interface, to_image_root

logger = logging.getLogger(__name__)

OUT_NAME = "lst.json"


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
    result = {
        "id": annot.id,
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": value,
        "from_name": CONTROL_NAMES[result_type][0],
        "to_name": "image",
        "type": result_type,
    }
    # Label Studio's Regions-panel nesting, which lst_to_json reads back as bounding_box_id
    bounding_box_id = getattr(annot, "bounding_box_id", None)
    if bounding_box_id is not None:
        result["parentID"] = bounding_box_id
    return result


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
