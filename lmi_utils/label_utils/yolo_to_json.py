"""Convert an Ultralytics YOLO dataset (detect, segment, oriented box or pose) to the AIS dataset json."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from lmi_utils.dataset_utils.file_utils import IMG_FORMATS, update_file_dimensions
from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    BoxAnnotation,
    Dataset,
    FileAnnotations,
    KeypointAnnotation,
    Label,
    Point2d,
    Polygon,
    PolygonAnnotation,
)
from lmi_utils.label_utils.bbox_utils import get_rotated_bbox

logger = logging.getLogger(__name__)

DEFAULT_JSON_NAME = "labels.json"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {f".{ext}" for ext in IMG_FORMATS}
BOX_VALUES = 4
OBB_VALUES = 8


def class_names(names) -> Dict[int, str]:
    """The yaml's `names`, in its mapping or list form, as class index -> class name."""
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    classes = {int(index): str(name) for index, name in entries}
    if len(set(classes.values())) != len(classes):
        raise ValueError(f"names repeats a class name: {names}")
    return classes


def keypoint_shape(config: dict) -> tuple:
    """The yaml's `kpt_shape` as (keypoint count, coordinate dimensions), or (0, 0) for a dataset without keypoints."""
    shape = config.get("kpt_shape")
    if shape is None:
        return 0, 0
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) not in (2, 3):
        raise ValueError(f"kpt_shape must be [keypoint count, 2 or 3], got {shape}")
    return int(shape[0]), int(shape[1])


def keypoint_layouts(config: dict, classes: Dict[int, str], keypoint_names: Optional[List[str]] = None) -> Dict[int, List[str]]:
    """Each class index's keypoint names in slot order.

    The yaml's `kpt_names` (keyed by class index, as json_to_yolo writes it) wins; without it every class takes
    `keypoint_names`, or `point-<slot>` names.
    """
    count, _ = keypoint_shape(config)
    declared = config.get("kpt_names")
    if not count:
        if declared or keypoint_names:
            raise ValueError("keypoint names are given for a dataset without kpt_shape")
        return {}
    if declared is None:
        # zero-padded so sorting the names keeps slot order
        layout = keypoint_names or [f"point-{slot:0{len(str(count - 1))}d}" for slot in range(count)]
        if len(layout) != count:
            raise ValueError(f"{len(layout)} keypoint names are given for kpt_shape's {count} keypoints")
        _check_layout(layout, "the keypoint names")
        return {index: list(layout) for index in classes}
    if keypoint_names:
        raise ValueError("keypoint names are given for a yaml that declares kpt_names")

    layouts = {index: [] for index in classes}
    for key, layout in declared.items():
        try:
            index = int(key)
        except ValueError:
            raise ValueError(f"kpt_names is keyed by class index, not '{key}'") from None
        if index not in classes:
            raise ValueError(f"kpt_names declares class index {index}, which names does not")
        layout = list(layout or [])
        if len(layout) > count:
            raise ValueError(f"kpt_names gives class '{classes[index]}' {len(layout)} keypoints, more than kpt_shape's {count}")
        _check_layout(layout, f"kpt_names of class '{classes[index]}'")
        layouts[index] = layout
    return layouts


def _check_layout(layout: List[str], where: str):
    if any(not isinstance(name, str) or not name for name in layout):
        raise ValueError(f"{where} must be non-empty strings: {layout}")
    if len(set(layout)) != len(layout):
        raise ValueError(f"{where} repeat a name: {layout}")


def dataset_root(config: dict, yaml_dir: Path) -> Path:
    """The directory the yaml's splits resolve against: its `path`, or the yaml's own directory when that is missing."""
    declared = config.get("path")
    if not declared:
        return yaml_dir
    root = Path(declared) if Path(declared).is_absolute() else yaml_dir / declared
    if not root.is_dir():
        # `path` often names where Ultralytics downloaded the dataset, which is stale once it is moved
        logger.warning(f"The yaml's path {root} does not exist; resolving the splits against {yaml_dir}")
        return yaml_dir
    return root


def split_dirs(config: dict, root: Path) -> List[Path]:
    """The image directories of the yaml's splits."""
    dirs = []
    for split in SPLITS:
        declared = config.get(split)
        if not declared:
            continue
        for entry in declared if isinstance(declared, list) else [declared]:
            path = Path(entry) if Path(entry).is_absolute() else root / entry
            if path.is_file():
                raise ValueError(f"Split '{split}' lists its images in {path}; only image directories are supported")
            if not path.is_dir():
                raise FileNotFoundError(f"Split '{split}' points at {path}, which does not exist")
            dirs.append(path)
    return dirs


def label_path(image_path: Path) -> Optional[Path]:
    """An image's label file: in the `labels` directory that pairs with its last `images` directory, as Ultralytics
    looks for it, else beside the image; None when neither exists."""
    images, labels = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    paired = Path(labels.join(str(image_path).rsplit(images, 1))).with_suffix(".txt")
    beside = image_path.with_suffix(".txt")
    return next((path for path in (paired, beside) if path.is_file()), None)


def parse_rows(
    text: str,
    file: FileAnnotations,
    classes: Dict[int, str],
    layouts: Dict[int, List[str]],
    shape: tuple,
    where: str,
    obb: bool = False,
) -> List[Annotation]:
    """One label file's rows as annotations, in pixels of the file's image."""
    count, dims = shape
    annotations = []
    for row, line in enumerate(text.splitlines()):
        fields = line.split()
        if not fields:
            continue
        at = f"{where}:{row + 1}"
        try:
            # Ultralytics reads the class as a float, so some exporters write `0.0`
            index = float(fields[0])
            values = [float(field) for field in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{at}: {error}") from None
        if index not in classes:
            raise ValueError(f"{at}: class index {fields[0]} is not in the yaml's names")
        index = int(index)
        annotation_id = f"{file.id}-{row}"
        if obb:
            if len(values) != OBB_VALUES:
                raise ValueError(f"{at}: an oriented box row holds {OBB_VALUES} values after the class, not {len(values)}")
            annotations.append(_oriented_box(annotation_id, classes[index], values, file))
        elif count:
            expected = BOX_VALUES + count * dims
            if len(values) != expected:
                raise ValueError(f"{at}: a pose row holds {expected} values after the class, not {len(values)}")
            box = _box(annotation_id, classes[index], values[:BOX_VALUES], file)
            annotations.append(box)
            annotations.extend(_keypoints(box, layouts[index], values[BOX_VALUES:], dims, file, at))
        elif len(values) == BOX_VALUES:
            annotations.append(_box(annotation_id, classes[index], values, file))
        elif len(values) >= 6 and len(values) % 2 == 0:
            points = [[values[i] * file.width, values[i + 1] * file.height] for i in range(0, len(values), 2)]
            annotations.append(PolygonAnnotation(id=annotation_id, label_id=classes[index], value=Polygon(points=points)))
        else:
            raise ValueError(f"{at}: {len(values)} values after the class are neither a box nor a polygon")
    return annotations


def _box(annotation_id: str, label_id: str, values: List[float], file: FileAnnotations) -> BoxAnnotation:
    x, y, w, h = values
    return BoxAnnotation(
        id=annotation_id,
        label_id=label_id,
        value=Box(
            x_min=(x - w / 2) * file.width,
            y_min=(y - h / 2) * file.height,
            x_max=(x + w / 2) * file.width,
            y_max=(y + h / 2) * file.height,
            angle=0,
        ),
    )


def _oriented_box(annotation_id: str, label_id: str, values: List[float], file: FileAnnotations) -> BoxAnnotation:
    corners = [[values[i] * file.width, values[i + 1] * file.height] for i in range(0, len(values), 2)]
    x, y, w, h, angle = get_rotated_bbox(np.array(corners))
    return BoxAnnotation(id=annotation_id, label_id=label_id, value=Box(x_min=x, y_min=y, x_max=x + w, y_max=y + h, angle=angle))


def _keypoints(box: BoxAnnotation, layout: List[str], values: List[float], dims: int, file: FileAnnotations, at: str):
    """The row's labeled keypoints; Ultralytics writes an unlabeled one as zeros, or with visibility 0."""
    keypoints = []
    for slot in range(len(values) // dims):
        x, y = values[slot * dims], values[slot * dims + 1]
        visibility = round(values[slot * dims + 2]) if dims == 3 else None
        if visibility == 0 or (visibility is None and x == 0 and y == 0):
            continue
        if visibility not in (None, 1, 2):
            raise ValueError(f"{at}: keypoint slot {slot} has visibility {values[slot * dims + 2]}, not 0, 1 or 2")
        if slot >= len(layout):
            raise ValueError(f"{at}: keypoint slot {slot} is labeled, but class '{box.label_id}' names {len(layout)} keypoints")
        keypoints.append(
            KeypointAnnotation(
                id=f"{box.id}-{layout[slot]}",
                label_id=layout[slot],
                value=Point2d(x=x * file.width, y=y * file.height, visibility=visibility),
                parent_id=box.id,
            )
        )
    return keypoints


def build_dataset(
    config: dict,
    yaml_dir: Path,
    keypoint_names: Optional[List[str]] = None,
    images: Optional[Path] = None,
    obb: bool = False,
) -> tuple:
    """Read a YOLO dataset into a Dataset, with image paths relative to the returned dataset root.

    Args:
        images: a directory to read in place of the yaml's splits, which then becomes the dataset root
        obb: read 8-value rows as oriented boxes rather than 4-point polygons

    Returns:
        tuple: the Dataset and its root directory
    """
    classes = class_names(config["names"])
    shape = keypoint_shape(config)
    if obb and shape[0]:
        raise ValueError("An oriented box dataset cannot have kpt_shape")
    layouts = keypoint_layouts(config, classes, keypoint_names)
    if images is not None:
        if not images.is_dir():
            raise FileNotFoundError(f"{images} is not a directory")
        root = images.resolve()
        dirs = [root]
    else:
        # absolute, so label_path finds the `images` directory of a dataset given relative to the working directory
        root = dataset_root(config, yaml_dir).resolve()
        dirs = split_dirs(config, root)
        if not dirs:
            raise ValueError(f"The yaml names none of the splits {', '.join(SPLITS)}; give an image directory instead")

    files = {}
    for directory in dirs:
        for image in sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
            try:
                path = image.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                raise ValueError(f"{image} is outside the dataset root {root}") from None
            # splits may share images, such as a val that reuses train; ids name files json_to_yolo writes, so hold no path
            files.setdefault(path, FileAnnotations(id=str(len(files)), path=path, height=None, width=None))
    dataset = Dataset(labels=[], files=list(files.values()))
    update_file_dimensions(dataset, str(root))

    types = {}
    unlabeled = 0
    for file in dataset.files:
        labels_file = label_path(root / file.path)
        if labels_file is None:
            unlabeled += 1
            continue
        file.annotations = parse_rows(labels_file.read_text(), file, classes, layouts, shape, str(labels_file), obb)
        for annotation in file.annotations:
            if annotation.type == AnnotationType.KEYPOINT:
                continue
            if types.setdefault(annotation.label_id, annotation.type) != annotation.type:
                raise ValueError(f"Class '{annotation.label_id}' is labeled with both boxes and polygons ({labels_file})")

    if unlabeled:
        logger.warning(f"{unlabeled} of {len(dataset.files)} images have no label file, beside them or under labels/")
    unseen_type = AnnotationType.POLYGON if AnnotationType.POLYGON in types.values() else AnnotationType.BOX
    dataset.labels = [
        Label(id=name, annotation_type=types.get(name, unseen_type), keypoints=layouts.get(index) or None)
        for index, name in sorted(classes.items())
    ]
    keypoint_ids = {name for layout in layouts.values() for name in layout} - set(classes.values())
    dataset.labels += [Label(id=name, annotation_type=AnnotationType.KEYPOINT) for name in sorted(keypoint_ids)]
    return dataset, root


def convert_yolo_to_json(
    dataset_yaml: Path,
    output_json: Optional[Path] = None,
    keypoint_names: Optional[List[str]] = None,
    images: Optional[Path] = None,
    obb: bool = False,
) -> Path:
    """Write the AIS dataset json of a YOLO dataset; json_to_factory then takes the dataset root as its image directory.

    Returns:
        Path: the json written, by default `labels.json` in the dataset root (the image directory, when one is given)
    """
    with open(dataset_yaml) as f:
        config = yaml.safe_load(f)
    dataset, root = build_dataset(config, Path(dataset_yaml).parent, keypoint_names, images, obb)
    output_json = output_json or root / DEFAULT_JSON_NAME
    dataset.save(str(output_json))
    logger.info(f"Converted {len(dataset.files)} images to {output_json}; their paths are relative to {root}")
    return output_json


def main():
    parser = argparse.ArgumentParser(description="Convert a YOLO dataset (detect, segment, oriented box or pose) to the AIS dataset json.")
    parser.add_argument(
        "--dataset_yaml", "-y", type=Path, required=True, help="the YOLO dataset yaml; only its classes and keypoints with -i"
    )
    parser.add_argument(
        "--images",
        "-i",
        type=Path,
        default=None,
        help="[optional] an image directory to convert in place of the yaml's splits; its labels sit beside the images or in "
        "the matching labels directory",
    )
    parser.add_argument("--obb", action="store_true", help="read 8-value rows as oriented boxes rather than 4-point polygons")
    parser.add_argument(
        "--output_json", "-o", type=Path, default=None, help=f"[optional] default=<dataset root, or -i>/{DEFAULT_JSON_NAME}"
    )
    parser.add_argument(
        "--keypoint_names",
        default=None,
        help="[optional] comma separated keypoint names in slot order, for a yaml without kpt_names; default=point-<slot>",
    )
    args = parser.parse_args()
    keypoint_names = [name.strip() for name in args.keypoint_names.split(",")] if args.keypoint_names else None
    convert_yolo_to_json(args.dataset_yaml, args.output_json, keypoint_names, args.images, args.obb)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
