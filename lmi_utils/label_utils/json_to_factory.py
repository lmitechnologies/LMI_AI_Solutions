import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from lmi_utils.dataset_utils.representations import Annotation, AnnotationType, Dataset, FileAnnotations, link_keypoints

logger = logging.getLogger(__name__)

"""
factory dataset format:
for each image:
{
    "id": "2azypeya",
    "width": 640,
    "height": 480,
    "annotations": [
        {
            "id": "abcd1234",
            "type": "Polygon",
            "label_id": "bicycle",
            "value": {
                "points": [
                    [100, 100],
                    [200, 100],
                    [200, 200],
                    [100, 200]
                ]
            }
        }
    ],
    "predictions": [
        {
            "id": "8oe97l4x",
            "type": "Box",
            "label_id": "bicycle",
            "value": {
                "x_min": 81,
                "y_min": 7,
                "x_max": 109,
                "y_max": 133,
                "angle": 0
            }
        },
    ]

"""


DATASET_META_FILE = ".meta.json"
# Factory's LabelNameSchema and LabelColorSchema
NAME_MAX_LENGTH = 100
COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")
# The types a Factory label is drawn as; a keypoint is a point of a box label
LABEL_TYPES = (AnnotationType.BOX, AnnotationType.POLYGON, AnnotationType.MASK)
UNSEEN_POINT_FRACTION = 0.5
# The images Factory's importer reads (ACCEPTED_EXTENSIONS in the SDK's dataset-import grouping)
FACTORY_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_LISTED_PROBLEMS = 20


def to_dict(annot: Annotation):
    """convert it to a dictionary."""
    return json.loads(annot.to_json())


def _annotation_lists(file: FileAnnotations):
    return (file.annotations, file.predictions)


def raise_problems(problems: List[str], summary: str):
    if problems:
        listed = problems[:MAX_LISTED_PROBLEMS]
        more = len(problems) - len(listed)
        raise ValueError(f"{summary}:\n  " + "\n  ".join(listed) + (f"\n  and {more} more" if more else ""))


def link_keypoints_to_boxes(dataset: Dataset):
    """Link each keypoint to its box as json_to_yolo does (see link_keypoints), since Factory refuses a keypoint naming no box.

    Raises:
        ValueError: listing every keypoint that cannot be linked
    """
    problems = []
    for file in dataset.files:
        for annotations in _annotation_lists(file):
            problems += [f"{file.path}: {message}" for _, message in link_keypoints(annotations)]
    raise_problems(problems, "These keypoints cannot be linked to a box")


def check_image_types(dataset: Dataset):
    unread = [file.path for file in dataset.files if Path(file.path).suffix.lower() not in FACTORY_IMAGE_SUFFIXES]
    raise_problems(unread, f"Factory imports only {', '.join(sorted(FACTORY_IMAGE_SUFFIXES))} images, so it would skip")


def build_label_dictionary(dataset: Dataset) -> Dict[str, dict]:
    """The Factory labelDictionary of a dataset, keyed by label name.

    A box label's points are its declared `keypoints`, else the names of the keypoints its boxes hold, sorted. Each
    point sits where it does on average in its upright boxes, as a fraction of the box.
    """
    shape_types = {}
    oriented = set()
    point_names = {}
    fractions = {}
    for file in dataset.files:
        for annotations in _annotation_lists(file):
            boxes = {a.id: a for a in annotations if a.type == AnnotationType.BOX}
            for annotation in annotations:
                if annotation.type != AnnotationType.KEYPOINT:
                    shape_types.setdefault(annotation.label_id, set()).add(annotation.type)
                    if annotation.type == AnnotationType.BOX and annotation.value.angle:
                        oriented.add(annotation.label_id)
                    continue
                box = boxes[annotation.parent_id]
                point_names.setdefault(box.label_id, set()).add(annotation.label_id)
                v = box.value
                if not v.angle and v.x_max > v.x_min and v.y_max > v.y_min:
                    fx = (annotation.value.x - v.x_min) / (v.x_max - v.x_min)
                    fy = (annotation.value.y - v.y_min) / (v.y_max - v.y_min)
                    fractions.setdefault((box.label_id, annotation.label_id), []).append((fx, fy))

    declared = {label.id: label for label in dataset.labels}
    names = list(declared) + [name for name in shape_types if name not in declared]
    dictionary = {}
    problems = []
    for name in names:
        label = declared.get(name)
        stated = label.annotation_type if label is not None else None
        drawn = shape_types.get(name, set())
        try:
            label_type = AnnotationType(stated) if stated is not None else next(iter(drawn), None)
        except ValueError:
            problems.append(f"label '{name}' has annotation type '{stated}', which Factory does not draw")
            continue
        if label_type == AnnotationType.KEYPOINT or (label_type is None and any(name in n for n in point_names.values())):
            continue
        if label_type is None:
            logger.warning(f"Leaving label '{name}' out of the {DATASET_META_FILE}: no annotation says what it is drawn as")
            continue
        if label_type not in LABEL_TYPES or drawn - {label_type}:
            kinds = ", ".join(sorted(t.value for t in drawn | {label_type}))
            problems.append(f"label '{name}' is drawn as {kinds}; a Factory label is one Box, Polygon or Bitmask")
            continue
        check_name(name, f"label '{name}'", problems)
        entry = {"id": name, "annotationType": label_type.value, "order": len(dictionary)}
        color = label.color if label is not None else None
        if color and COLOR_PATTERN.match(color):
            entry["color"] = color
        elif color:
            logger.warning(f"Dropping label '{name}''s color {color}, which is not #rrggbb")
        if name in oriented:
            entry["oriented"] = True
        observed = point_names.get(name, set())
        layout = list(label.keypoints) if label is not None and label.keypoints else sorted(observed)
        unknown = sorted(observed - set(layout))
        if unknown:
            problems.append(f"label '{name}' declares keypoints {layout}, but its boxes hold {unknown}")
        if layout:
            if label_type != AnnotationType.BOX:
                problems.append(f"label '{name}' is a {label_type.value}, and only a box holds keypoints")
            for point in layout:
                check_name(point, f"keypoint '{point}' of label '{name}'", problems)
            if len(set(layout)) != len(layout):
                problems.append(f"label '{name}' names a keypoint twice: {layout}")
            entry["points"] = [_point(point, fractions.get((name, point))) for point in layout]
        dictionary[name] = entry
    raise_problems(problems, "Factory refuses these labels")
    return dictionary


def check_name(name: str, where: str, problems: List[str]):
    if name != name.strip() or not 0 < len(name) <= NAME_MAX_LENGTH:
        problems.append(f"{where} must be 1 to {NAME_MAX_LENGTH} characters with no surrounding spaces")


def _point(name: str, fractions: Optional[List[tuple]]) -> dict:
    if not fractions:
        return {"name": name, "x": UNSEEN_POINT_FRACTION, "y": UNSEEN_POINT_FRACTION}
    mean = [min(max(sum(axis) / len(fractions), 0.0), 1.0) for axis in zip(*fractions)]
    return {"name": name, "x": round(mean[0], 3), "y": round(mean[1], 3)}


def convert_json_to_factory(dataset: Dataset, image_dir: Path, output_dir: Path):
    """
    Convert a JSON file to a Factory dataset.

    Args:
        dataset (Dataset): Dataset object representing the input JSON file.
        image_dir (Path): Path to the directory containing the images.
        output_dir (Path): Path to the output Factory dataset directory.
    """
    check_image_types(dataset)
    link_keypoints_to_boxes(dataset)
    label_dictionary = build_label_dictionary(dataset)
    fname_to_list = {}
    # Factory reads image.* and annotation.* as one item per folder only when every file is named that way
    item_folders = all(Path(file.path).stem == "image" for file in dataset.files)
    for file in dataset.files:
        image_path = image_dir / file.path
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image file {image_path}.")

        h0, w0 = image.shape[:2]
        annots = {
            "width": w0,
            "height": h0,
            "annotations": [],
            "predictions": [],
        }
        for annotation in file.annotations:
            annots["annotations"].append(to_dict(annotation))
        for prediction in file.predictions:
            annots["predictions"].append(to_dict(prediction))

        rel_path = image_path.relative_to(image_dir)
        outname = rel_path.with_name("annotation.label.json") if item_folders else rel_path.with_suffix(".label.json")
        fname_to_list[outname] = annots

    # write annotation files
    for fname, annots in fname_to_list.items():
        out_path = output_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(annots, f, indent=4)

    # copy images to output_dir
    for file in dataset.files:
        image_path = image_dir / file.path
        rel_path = image_path.relative_to(image_dir)
        out_image_path = output_dir / rel_path
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, out_image_path)

    if label_dictionary:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / DATASET_META_FILE, "w") as f:
            json.dump({"labelDictionary": label_dictionary}, f, indent=4)

    logger.info(f"Successfully converted {len(dataset.files)} files and saved to {output_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Convert a JSON file to a Factory dataset.")
    parser.add_argument("--image_dir", "-i", type=Path, required=True, help="Path to the directory containing the images.")
    parser.add_argument(
        "--json_file", "-j", type=Path, default=None, help="Path to the input JSON file. Default is <image_dir>/labels.json"
    )
    parser.add_argument("--output_dir", "-o", type=Path, required=True, help="Path to the output Factory dataset directory.")
    args = parser.parse_args()

    if args.json_file is None:
        args.json_file = args.image_dir / "labels.json"

    dataset = Dataset.load(str(args.json_file))
    convert_json_to_factory(dataset, args.image_dir, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
