import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2

from lmi_utils.dataset_utils.representations import UNASSIGNED_KEYPOINT_POLICIES, Annotation, AnnotationType, Dataset, Label

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

# The root `.meta.json` carries the dataset-level annotation contract. Unlike the per-image annotation files it is
# camelCase, matching the Factory API's own document format.
DATASET_META_FILE = ".meta.json"
POSE_SCHEMA_TYPE = "Pose"
POSE_SCHEMA_VERSION = 1
# COCO's x, y, visibility. Factory standardizes every declared pose schema on 3: a two-dimensional source's
# points import without visibility values and default to visible at training export.
COORDINATE_DIMENSIONS = 3


def _without_nulls(value):
    """Drop unset optional fields. Factory reads an absent field as undeclared, but rejects an explicit null."""
    if isinstance(value, dict):
        return {key: _without_nulls(entry) for key, entry in value.items() if entry is not None}
    if isinstance(value, list):
        return [_without_nulls(entry) for entry in value]
    return value


def to_dict(annot: Annotation):
    """convert it to a dictionary."""
    return _without_nulls(json.loads(annot.to_json()))


def collect_pose_schema_issues(schema: dict) -> List[str]:
    """Every structural problem in a pose schema, so a bad conversion is fixed in one pass instead of one error per retry."""
    issues = []
    if schema.get("coordinateDimensions") not in (2, 3):
        issues.append(f"coordinateDimensions is {schema.get('coordinateDimensions')}, which must be 2 or 3")
    classes = schema.get("classes") or {}
    if not classes:
        issues.append("the schema declares no classes")

    for class_id, class_schema in classes.items():
        where = f"class '{class_id}'"
        keypoints = class_schema.get("keypoints") or []
        if not keypoints:
            issues.append(f"{where} declares no keypoints")
        duplicates = sorted({name for name in keypoints if keypoints.count(name) > 1})
        if duplicates:
            issues.append(f"{where} declares keypoint(s) {', '.join(duplicates)} more than once")

        seen_edges = set()
        for edge in class_schema.get("skeleton") or []:
            start, end = edge
            if not (0 <= start < len(keypoints)) or not (0 <= end < len(keypoints)):
                issues.append(f"{where} has skeleton edge [{start}, {end}] outside its {len(keypoints)} keypoint slots")
            elif start == end:
                issues.append(f"{where} has skeleton self-edge [{start}, {end}]")
            elif (min(start, end), max(start, end)) in seen_edges:
                issues.append(f"{where} has duplicate skeleton edge [{start}, {end}]")
            else:
                seen_edges.add((min(start, end), max(start, end)))

        # Disjoint named swaps are self-inverse by construction, so mirroring twice restores every keypoint with
        # no permutation invariant left to check; only membership and disjointness can go wrong.
        swapped = set()
        for pair in class_schema.get("horizontalFlipPairs") or []:
            if len(pair) != 2:
                issues.append(f"{where} has a horizontal flip entry {list(pair)} that is not a pair of keypoints")
                continue
            first, second = pair
            undeclared = [name for name in pair if name not in keypoints]
            if undeclared:
                issues.append(f"{where} has a horizontal flip pair naming {', '.join(undeclared)}, which it does not declare")
            elif first == second:
                issues.append(f"{where} pairs keypoint '{first}' with itself under a horizontal flip")
            elif swapped & {first, second}:
                issues.append(f"{where} uses {', '.join(sorted(swapped & {first, second}))} in more than one horizontal flip pair")
            else:
                swapped.update(pair)

    return issues


def build_pose_schema(labels: Iterable[Label]) -> Optional[dict]:
    """The declared pose schema of the labels carrying a keypoint layout, or None when none do.

    A layout is a declaration, never a tally of observations: a class keeps every slot it declares even when no
    image in this dataset observes it. A label with no declared symmetry gets `horizontalFlipPairs: null`, which
    imports but prevents horizontal flipping during training.

    Raises:
        ValueError: if the resulting schema is structurally invalid.
    """
    classes = {}
    for label in labels:
        if not label.keypoints:
            continue
        class_schema = {
            "keypoints": list(label.keypoints),
            "horizontalFlipPairs": (
                [[str(first), str(second)] for first, second in label.horizontal_flip_pairs]
                if label.horizontal_flip_pairs is not None
                else None
            ),
        }
        if label.skeleton:
            class_schema["skeleton"] = [[int(start), int(end)] for start, end in label.skeleton]
        classes[label.id] = class_schema

    if not classes:
        return None

    schema = {
        "type": POSE_SCHEMA_TYPE,
        "version": POSE_SCHEMA_VERSION,
        "coordinateDimensions": COORDINATE_DIMENSIONS,
        "classes": classes,
    }
    issues = collect_pose_schema_issues(schema)
    if issues:
        raise ValueError(f"Invalid pose schema: {'; '.join(issues)}")
    return schema


def scaffold_pose_schema(dataset: Dataset) -> dict:
    """A draft pose schema surveyed from what a dataset's annotations happen to show, for a human to finish.

    This is authoring help, never a contract. Slot membership and order here are observations, so they are wrong
    the moment a class owns a slot that no image in this dataset shows, and the order is only the order the
    annotator worked in. The mirror pairs are always left null because no annotation can reveal them -- that `left_eye`
    mirrors to `right_eye` is knowledge about the object, not about the data. Edit the result, then declare it.

    Keypoints in more than one box, or in none, are skipped rather than guessed at.
    """
    classes: Dict[str, List[str]] = {}
    for file in dataset.files:
        boxes = [a for a in file.annotations if a.type == AnnotationType.BOX]
        boxes_by_id = {box.id: box for box in boxes}
        for annotation in file.annotations:
            if annotation.type != AnnotationType.KEYPOINT:
                continue
            owner = boxes_by_id.get(annotation.bounding_box_id)
            if owner is None:
                containing = [box for box in boxes if box.value.point_in_box(annotation.value.x, annotation.value.y)]
                if len(containing) != 1:
                    logger.warning(f"Skipped keypoint '{annotation.label_id}' in {file.path}: {len(containing)} boxes contain it")
                    continue
                owner = containing[0]
            keypoints = classes.setdefault(owner.label_id, [])
            if annotation.label_id not in keypoints:
                keypoints.append(annotation.label_id)

    return {
        "type": POSE_SCHEMA_TYPE,
        "version": POSE_SCHEMA_VERSION,
        "coordinateDimensions": COORDINATE_DIMENSIONS,
        "classes": {class_id: {"keypoints": keypoints, "horizontalFlipPairs": None} for class_id, keypoints in classes.items()},
    }


def convert_json_to_factory(
    dataset: Dataset,
    image_dir: Path,
    output_dir: Path,
    annotation_schema: Optional[dict] = None,
    unlinked_keypoints: str = "error",
):
    """
    Convert a JSON file to a Factory dataset.

    Keypoints are resolved to their owning box before writing, so every keypoint reaches Factory as part of an
    instance rather than as a loose point. The dataset's declared pose schema is written to a `.meta.json` in
    every directory the result can be imported from; without it Factory can import the images but cannot train
    pose on them. Every declared pose schema is written with the standard COORDINATE_DIMENSIONS, whatever the
    source stored per point.

    Args:
        dataset (Dataset): Dataset object representing the input JSON file.
        image_dir (Path): Path to the directory containing the images.
        output_dir (Path): Path to the output Factory dataset directory.
        annotation_schema (dict): Annotation contract to declare. Defaults to None, which derives the pose schema
            from the dataset's labels.
        unlinked_keypoints (str): What becomes of a keypoint no box owns -- "error", "drop" or "keep". A pose
            model trains on box-linked keypoints only, so a kept one reaches Factory but not training.
    """
    if annotation_schema is None:
        annotation_schema = build_pose_schema(dataset.labels)
    elif annotation_schema.get("type") == POSE_SCHEMA_TYPE:
        annotation_schema = {**annotation_schema, "coordinateDimensions": COORDINATE_DIMENSIONS}

    fname_to_list = {}
    for file in dataset.files:
        image_path = image_dir / file.path
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image file {image_path}.")

        h0, w0 = image.shape[:2]
        if file.get_annotations_by_type(AnnotationType.KEYPOINT):
            file.assign_keypoints(unassigned=unlinked_keypoints)
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
        outname = rel_path.with_suffix(".label.json")
        fname_to_list[outname] = annots

    # write annotation files
    for fname, annots in fname_to_list.items():
        out_path = output_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(annots, f, indent=4)

    if annotation_schema is not None:
        meta_dirs = dataset_meta_dirs(output_dir, fname_to_list)
        for meta_dir in meta_dirs:
            write_dataset_meta(meta_dir, annotation_schema)
        logger.info(
            f"Declared a {annotation_schema['type']} schema of {len(annotation_schema['classes'])} class(es) in "
            f"{len(meta_dirs)} {DATASET_META_FILE} file(s)."
        )

    # copy images to output_dir
    for file in dataset.files:
        image_path = image_dir / file.path
        rel_path = image_path.relative_to(image_dir)
        out_image_path = output_dir / rel_path
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, out_image_path)

    logger.info(f"Successfully converted {len(dataset.files)} files and saved to {output_dir}.")


def dataset_meta_dirs(output_dir: Path, annotation_paths: Iterable[Path]) -> List[Path]:
    """Every directory the converted dataset can be imported from.

    Factory reads the contract from a `.meta.json` at the exact root of the folder it is given and never searches
    parent directories, so each split nested below the output root (`images/val/`) needs its own copy; without one
    its keypoints import with no declared layout and are rejected. The output root is always included so a dataset
    written flat still gets one.
    """
    dirs = {output_dir}
    for path in annotation_paths:
        dirs.add((output_dir / path).parent)
    return sorted(dirs)


def write_dataset_meta(output_dir: Path, annotation_schema: dict, meta: Optional[Dict] = None):
    """Write one `.meta.json` declaring the dataset's annotation contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    document = dict(meta or {})
    document["annotationSchema"] = annotation_schema
    with open(output_dir / DATASET_META_FILE, "w") as f:
        json.dump(document, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Convert a JSON file to a Factory dataset.")
    parser.add_argument("--image_dir", "-i", type=Path, required=True, help="Path to the directory containing the images.")
    parser.add_argument(
        "--json_file", "-j", type=Path, default=None, help="Path to the input JSON file. Default is <image_dir>/labels.json"
    )
    parser.add_argument("--output_dir", "-o", type=Path, required=True, help="Path to the output Factory dataset directory.")
    parser.add_argument(
        "--unlinked_keypoints",
        choices=UNASSIGNED_KEYPOINT_POLICIES,
        default="error",
        help="[optional] what becomes of a keypoint no box owns. Default is to reject the dataset.",
    )
    args = parser.parse_args()

    if args.json_file is None:
        args.json_file = args.image_dir / "labels.json"

    dataset = Dataset.load(str(args.json_file))
    convert_json_to_factory(
        dataset,
        args.image_dir,
        args.output_dir,
        unlinked_keypoints=args.unlinked_keypoints,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
