import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml

from lmi_utils.dataset_utils.file_utils import IMG_FORMATS
from lmi_utils.dataset_utils.representations import (
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

logger = logging.getLogger(__name__)

DATASET_YAML = "dataset.yaml"
DEFAULT_JSON_NAME = "labels.json"
# Ultralytics locates a split's labels by swapping this component of its image path.
IMAGE_DIR_COMPONENT = "images"
LABEL_DIR_COMPONENT = "labels"
# A pose model has one global slot layout, so a class that owns fewer slots pads the rest. Padding is named with
# this prefix by the Factory exporter and is not part of the class's declared layout.
UNUSED_SLOT_PREFIX = "__unused_"
BOX_FIELDS = 5


def load_dataset_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def class_ids_by_index(names) -> Dict[int, str]:
    """The yaml's `names` as an index to class id map, accepting either the mapping or the list form."""
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    class_ids = {}
    for raw_index, raw_class_id in entries:
        index = int(raw_index)
        class_id = str(raw_class_id)
        if index in class_ids:
            raise ValueError(f"names declares class index {index} more than once")
        if class_id in class_ids.values():
            raise ValueError(f"names declares class id '{class_id}' more than once")
        class_ids[index] = class_id
    if not class_ids:
        raise ValueError("names declares no classes")
    return class_ids


def keypoint_names_by_class(class_ids: Dict[int, str], kpt_names: Optional[dict], keypoint_count: int) -> Optional[Dict[str, List[str]]]:
    """Normalize Ultralytics and Factory `kpt_names` maps to Factory class ids.

    Ultralytics' built-in datasets key layouts by numeric class index, while Factory-exported datasets key them
    by class name. YAML may also deserialize a numeric index as either an integer or a string, so all three forms
    are accepted. Conflicting aliases are rejected rather than choosing one silently.
    """
    if kpt_names is None:
        return None
    if not isinstance(kpt_names, dict):
        raise ValueError("kpt_names must map each class index or class id to an ordered keypoint list")

    normalized = {}
    used_keys = set()
    for index, class_id in class_ids.items():
        candidates = []
        for key in dict.fromkeys((index, str(index), class_id)):
            if key in kpt_names:
                candidates.append((key, kpt_names[key]))
        if not candidates:
            raise ValueError(f"kpt_names declares no layout for class '{class_id}' (index {index})")

        layouts = []
        for key, value in candidates:
            if not isinstance(value, list) or any(not isinstance(name, str) or not name for name in value):
                raise ValueError(f"kpt_names entry {key!r} must be a list of non-empty strings")
            if len(value) != keypoint_count:
                raise ValueError(f"kpt_names for '{class_id}' has {len(value)} entries for a {keypoint_count} slot model")
            layouts.append(value)
            used_keys.add(key)
        if any(layout != layouts[0] for layout in layouts[1:]):
            aliases = ", ".join(repr(key) for key, _ in candidates)
            raise ValueError(f"kpt_names entries {aliases} disagree for class '{class_id}'")
        normalized[class_id] = list(layouts[0])

    unused = [key for key in kpt_names if key not in used_keys]
    if unused:
        raise ValueError(f"kpt_names declares layout(s) for unknown class key(s): {unused}")
    return normalized


def class_slots(class_id: str, kpt_names: Optional[Dict[str, List[str]]], keypoint_count: int) -> List[int]:
    """The model slots this class owns, in slot order.

    Without `kpt_names` a YOLO file states one layout for the whole model, so every class owns every slot. The
    per-class layouts the Factory exporter writes name a class's unowned slots with `UNUSED_SLOT_PREFIX`.
    """
    if kpt_names is None:
        return list(range(keypoint_count))
    if class_id not in kpt_names:
        raise ValueError(f"kpt_names declares no layout for class '{class_id}'")
    names = kpt_names[class_id]
    if len(names) != keypoint_count:
        raise ValueError(f"kpt_names for '{class_id}' has {len(names)} entries for a {keypoint_count} slot model")
    slots = [slot for slot, name in enumerate(names) if not name.startswith(UNUSED_SLOT_PREFIX)]
    if not slots:
        raise ValueError(f"kpt_names for '{class_id}' declares no owned slots")
    return slots


def local_flip(slots: List[int], flip_idx: Optional[List[int]]) -> Optional[List[int]]:
    """Translate the model's global flip into this class's local order, or None when it does not close over the class.

    A slot whose mirror the class does not own leaves the whole class's flip undeclared: a partial mapping would
    silently drop keypoints under a horizontal flip rather than mirror them.
    """
    if flip_idx is None:
        return None
    local_by_slot = {slot: local for local, slot in enumerate(slots)}
    flip = []
    for slot in slots:
        target = flip_idx[slot]
        if target not in local_by_slot:
            logger.warning(f"Slot {slot} mirrors to slot {target}, which the class does not own; declaring no flip for it")
            return None
        flip.append(local_by_slot[target])
    return flip


def build_labels(
    class_ids: Dict[int, str],
    kpt_names: Optional[Dict[str, List[str]]],
    keypoint_count: int,
    flip_idx: Optional[List[int]],
    default_layout: List[str],
) -> Tuple[Dict[int, Label], Dict[int, List[int]]]:
    """The Factory label of each class index, with the model slots each one owns."""
    labels = {}
    slots_by_index = {}
    for index, class_id in class_ids.items():
        if not keypoint_count:
            labels[index] = Label(id=class_id, annotation_type=AnnotationType.BOX)
            continue
        slots = class_slots(class_id, kpt_names, keypoint_count)
        layout = [kpt_names[class_id][slot] for slot in slots] if kpt_names else [default_layout[slot] for slot in slots]
        labels[index] = Label(
            id=class_id,
            annotation_type=AnnotationType.BOX,
            keypoints=layout,
            horizontal_flip=local_flip(slots, flip_idx),
        )
        slots_by_index[index] = slots
    return labels, slots_by_index


def split_image_dirs(config: dict, root: Path, splits: List[str]) -> Dict[str, Path]:
    """The image directory of each split the yaml declares, keyed by split name."""
    base = Path(config["path"]) if config.get("path") else root
    if not base.is_absolute():
        base = root / base
    if not base.is_dir():
        # `path` records where the dataset was written, which is stale once it has been moved or unpacked elsewhere.
        logger.warning(f"The yaml's path {base} does not exist; resolving the splits against {root} instead")
        base = root
    dirs = {}
    for split in splits:
        declared = config.get(split)
        if not declared:
            continue
        path = Path(declared)
        path = path if path.is_absolute() else base / path
        if not path.is_dir():
            logger.warning(f"Split '{split}' points at {path}, which does not exist; skipping it")
            continue
        dirs[split] = path
    return dirs


def label_path(image_path: Path) -> Path:
    """The label file Ultralytics pairs with an image."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == IMAGE_DIR_COMPONENT:
            parts[index] = LABEL_DIR_COMPONENT
            break
    return Path(*parts).with_suffix(".txt")


def parse_rows(
    text: str,
    labels: Dict[int, Label],
    slots_by_index: Dict[int, List[int]],
    keypoint_count: int,
    coordinate_dimensions: int,
    height: int,
    width: int,
    file_id: str,
) -> List:
    """Convert one label file's rows into Factory annotations, denormalized against the image size."""
    annotations = []
    for row_index, line in enumerate(text.splitlines()):
        fields = line.split()
        if not fields:
            continue
        class_index = int(fields[0])
        if class_index not in labels:
            raise ValueError(f"Row {row_index} names class index {class_index}, which the yaml does not declare")
        label = labels[class_index]
        values = [float(field) for field in fields[1:]]
        annotation_id = f"{file_id}-{row_index}"

        if not keypoint_count:
            annotations.append(_detection_annotation(annotation_id, label.id, values, height, width))
            continue

        expected = BOX_FIELDS - 1 + keypoint_count * coordinate_dimensions
        if len(values) != expected:
            raise ValueError(f"Row {row_index} has {len(values)} values for the {expected} a {keypoint_count} keypoint model writes")
        box = _box_annotation(annotation_id, label.id, values[: BOX_FIELDS - 1], height, width)
        annotations.append(box)
        annotations.extend(
            _keypoint_annotations(box, label, slots_by_index[class_index], values[BOX_FIELDS - 1 :], coordinate_dimensions, height, width)
        )
    return annotations


def _box_annotation(annotation_id: str, label_id: str, values: List[float], height: int, width: int) -> BoxAnnotation:
    center_x, center_y, box_width, box_height = values
    return BoxAnnotation(
        id=annotation_id,
        label_id=label_id,
        value=Box(
            x_min=(center_x - box_width / 2) * width,
            y_min=(center_y - box_height / 2) * height,
            x_max=(center_x + box_width / 2) * width,
            y_max=(center_y + box_height / 2) * height,
        ),
    )


def _detection_annotation(annotation_id: str, label_id: str, values: List[float], height: int, width: int):
    """A non-pose row: four values are a box, more are a segmentation polygon."""
    if len(values) == BOX_FIELDS - 1:
        return _box_annotation(annotation_id, label_id, values, height, width)
    if len(values) < 6 or len(values) % 2:
        raise ValueError(f"A row of {len(values)} values is neither a box nor a polygon")
    points = [[values[index] * width, values[index + 1] * height] for index in range(0, len(values), 2)]
    return PolygonAnnotation(id=annotation_id, label_id=label_id, value=Polygon(points=points))


def _keypoint_annotations(
    box: BoxAnnotation, label: Label, slots: List[int], values: List[float], coordinate_dimensions: int, height: int, width: int
) -> List[KeypointAnnotation]:
    """This row's observed keypoints, read out of the slots the class owns.

    An unobserved slot is written zeroed by Ultralytics, so it is dropped here and stays an empty declared slot
    in Factory rather than becoming a keypoint at the image origin.
    """
    annotations = []
    for local, slot in enumerate(slots):
        offset = slot * coordinate_dimensions
        x, y = values[offset], values[offset + 1]
        visibility = int(values[offset + 2]) if coordinate_dimensions == 3 else None
        if visibility == 0 or (visibility is None and x == 0 and y == 0):
            continue
        annotations.append(
            KeypointAnnotation(
                id=f"{box.id}-{label.keypoints[local]}",
                label_id=label.keypoints[local],
                value=Point2d(x=x * width, y=y * height, visibility=visibility),
                bounding_box_id=box.id,
            )
        )
    return annotations


def build_dataset(config: dict, root: Path, splits: List[str], default_keypoint_names: Optional[List[str]]) -> Dataset:
    """Convert a YOLO dataset into the representation the Factory writer consumes.

    Every split is read into one dataset, with each image keeping its path relative to the dataset root so images
    of the same name in different splits stay distinct.
    """
    class_ids = class_ids_by_index(config["names"])
    kpt_shape = config.get("kpt_shape")
    if kpt_shape is not None:
        if not isinstance(kpt_shape, (list, tuple)) or len(kpt_shape) != 2:
            raise ValueError("kpt_shape must be [keypoint count, coordinate dimensions]")
        keypoint_count, coordinate_dimensions = int(kpt_shape[0]), int(kpt_shape[1])
        if keypoint_count <= 0:
            raise ValueError("kpt_shape must declare at least one keypoint")
        if coordinate_dimensions not in (2, 3):
            raise ValueError(f"kpt_shape coordinate dimensions must be 2 or 3, got {coordinate_dimensions}")
    else:
        keypoint_count, coordinate_dimensions = 0, 0

    raw_flip = config.get("flip_idx")
    flip_idx = [int(target) for target in raw_flip] if raw_flip is not None else None
    if flip_idx is not None:
        if not keypoint_count:
            raise ValueError("flip_idx is declared without kpt_shape")
        if len(flip_idx) != keypoint_count:
            raise ValueError(f"flip_idx has {len(flip_idx)} entries for a {keypoint_count} keypoint model")
        if sorted(flip_idx) != list(range(keypoint_count)):
            raise ValueError(f"flip_idx must be a permutation of 0 through {keypoint_count - 1}")
        not_involution = next((slot for slot, target in enumerate(flip_idx) if flip_idx[target] != slot), None)
        if not_involution is not None:
            raise ValueError(
                f"flip_idx is not an involution: slot {not_involution} maps to {flip_idx[not_involution]}, "
                f"which maps to {flip_idx[flip_idx[not_involution]]}"
            )

    kpt_names = keypoint_names_by_class(class_ids, config.get("kpt_names"), keypoint_count)
    if kpt_names is not None and not keypoint_count:
        raise ValueError("kpt_names is declared without kpt_shape")

    default_layout = default_keypoint_names or [f"point-{slot}" for slot in range(keypoint_count)]
    if keypoint_count and len(default_layout) != keypoint_count:
        raise ValueError(f"{len(default_layout)} keypoint names were given for a {keypoint_count} keypoint model")
    labels, slots_by_index = build_labels(class_ids, kpt_names, keypoint_count, flip_idx, default_layout)

    files = []
    for split, image_dir in split_image_dirs(config, root, splits).items():
        for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lstrip(".").lower() in IMG_FORMATS):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image file {image_path}.")
            height, width = image.shape[:2]
            file_id = str(image_path.relative_to(image_dir).with_suffix(""))
            labels_file = label_path(image_path)
            text = labels_file.read_text() if labels_file.is_file() else ""
            if not text:
                logger.debug(f"No label file for {image_path}; importing it as a background image")
            files.append(
                FileAnnotations(
                    id=f"{split}-{file_id}",
                    path=str(image_path.relative_to(root)),
                    height=height,
                    width=width,
                    annotations=parse_rows(
                        text, labels, slots_by_index, keypoint_count, coordinate_dimensions, height, width, f"{split}-{file_id}"
                    ),
                )
            )

    return Dataset(labels=list(labels.values()), files=files, coordinate_dimensions=coordinate_dimensions or None)


def convert_yolo_to_json(
    dataset_yaml: Path,
    output_json: Optional[Path] = None,
    root: Optional[Path] = None,
    splits: Optional[List[str]] = None,
    keypoint_names: Optional[List[str]] = None,
) -> Dataset:
    """Convert a YOLO dataset into the AIS dataset json, from which json_to_factory writes a Factory dataset.

    Image paths are written relative to the dataset root, so json_to_factory takes that root as its image
    directory and every split keeps its own items.
    """
    root = root or dataset_yaml.parent
    config = load_dataset_yaml(dataset_yaml)
    dataset = build_dataset(config, root, splits or ["train", "val", "test"], keypoint_names)
    dataset.save(str(output_json or root / DEFAULT_JSON_NAME))
    logger.info(f"Converted {len(dataset.files)} file(s) and {len(dataset.labels)} label(s) to {output_json or root / DEFAULT_JSON_NAME}.")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert a YOLO dataset to the AIS dataset json.")
    parser.add_argument("--dataset_yaml", "-y", type=Path, required=True, help=f"Path to the YOLO {DATASET_YAML}.")
    parser.add_argument(
        "--output_json",
        "-o",
        type=Path,
        default=None,
        help=f"[optional] path of the output json. Default is <root>/{DEFAULT_JSON_NAME}, where json_to_factory reads it.",
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="[optional] the directory relative paths resolve against. Default is the yaml's directory."
    )
    parser.add_argument("--splits", default="train,val,test", help="[optional] the comma separated splits to convert.")
    parser.add_argument(
        "--keypoint_names",
        default=None,
        help="[optional] the comma separated keypoint names of a file that declares no kpt_names, in slot order. "
        "Default is to name the slots positionally.",
    )
    args = parser.parse_args()

    convert_yolo_to_json(
        args.dataset_yaml,
        args.output_json,
        root=args.root,
        splits=args.splits.split(","),
        keypoint_names=args.keypoint_names.split(",") if args.keypoint_names else None,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
