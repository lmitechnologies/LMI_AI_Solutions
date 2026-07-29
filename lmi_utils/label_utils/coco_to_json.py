import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pycocotools import mask as coco_mask

from lmi_utils.dataset_utils.coco_dataset import CocoCategory, CocoDataset
from lmi_utils.dataset_utils.pose_identifiers import derive_pose_id
from lmi_utils.dataset_utils.representations import (
    AnnotationType,
    Box,
    BoxAnnotation,
    Dataset,
    FileAnnotations,
    KeypointAnnotation,
    Label,
    Mask,
    MaskAnnotation,
    Point2d,
    Polygon,
    PolygonAnnotation,
)

logger = logging.getLogger(__name__)

DEFAULT_JSON_NAME = "labels.json"

# COCO stores keypoints as x, y, visibility.
COORDINATE_DIMENSIONS = 3
# Visibility 0 means the instance does not observe the slot. It is dropped rather than written at the origin, so
# the slot stays empty in Factory instead of claiming a keypoint at (0, 0).
VISIBILITY_UNOBSERVED = 0


def load_class_map(path: Optional[Path]) -> Dict[str, str]:
    """A COCO category name to the name it should carry in Factory; empty when no file is given."""
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def load_flip_map(path: Optional[Path]) -> Dict[str, List]:
    """A class name to the keypoint-name pairs that exchange places when an image is mirrored."""
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def build_labels(
    categories: List[CocoCategory],
    class_map: Dict[str, str],
    flip_map: Dict[str, List],
    skeleton_base: int,
) -> Tuple[Dict[int, Label], Dict[str, str]]:
    """The Factory label of each COCO category, keyed by category id, and the keypoint vocabulary they draw on.

    COCO names classes and keypoints for people to read, so each name is kept as the display name and its
    identity is derived from it. `class_map` and `flip_map` are written in those same names, after any
    renaming `class_map` does -- nothing asks the caller for an id.

    COCO carries no flip symmetry, so a class reaches Factory with `horizontalFlipPairs: null` -- importable but
    not horizontally flippable -- unless `flip_map` supplies the mirror pairs.
    """
    labels = {}
    keypoint_names: Dict[str, str] = {}
    for category in categories:
        class_name = class_map.get(category.name, category.name)
        skeleton = None
        if category.skeleton:
            skeleton = [[int(start) - skeleton_base, int(end) - skeleton_base] for start, end in category.skeleton]
        flip = flip_map.get(class_name)
        if flip is not None and not category.keypoints:
            raise ValueError(f"A horizontal flip was supplied for '{class_name}', which declares no keypoints")
        for keypoint_name in category.keypoints or []:
            keypoint_names[derive_pose_id(keypoint_name)] = keypoint_name
        labels[category.id] = Label(
            id=derive_pose_id(class_name),
            name=class_name,
            annotation_type=AnnotationType.BOX,
            keypoint_ids=[derive_pose_id(name) for name in category.keypoints] if category.keypoints else None,
            horizontal_flip_pairs=[[derive_pose_id(first), derive_pose_id(second)] for first, second in flip] if flip else flip,
            skeleton=skeleton,
        )
    return labels, keypoint_names


def _segmentation_annotation(annotation, label_id: str, height: int, width: int):
    """The instance's segmentation as a polygon or bitmask annotation, or None when it carries none."""
    segmentation = annotation.segmentation
    if not segmentation:
        return None
    if isinstance(segmentation, dict):
        decoded = coco_mask.decode(coco_mask.frPyObjects(segmentation, height, width))
        return MaskAnnotation(id=str(annotation.id), label_id=label_id, value=Mask(mask=np.ascontiguousarray(decoded).astype(np.uint8)))
    # A COCO polygon is a flat x, y list per part; Factory holds one polygon, so only the first part is kept.
    points = np.array(segmentation[0], dtype=float).reshape(-1, 2).tolist()
    return PolygonAnnotation(id=str(annotation.id), label_id=label_id, value=Polygon(points=points))


def build_dataset(
    coco: CocoDataset, class_map: Dict[str, str], flip_map: Dict[str, List], skeleton_base: int, segmentation: bool
) -> Dataset:
    """Convert a loaded COCO dataset into the AIS dataset representation.

    Each instance becomes one box (the anchor its keypoints link to) plus one keypoint annotation per observed
    slot. With `segmentation`, instances of classes declaring no keypoints become polygons or bitmasks instead.
    """
    labels, keypoint_names = build_labels(coco.categories, class_map, flip_map, skeleton_base)
    files = []
    for image in coco.images:
        annotations = []
        for annotation in coco.get_annotations_by_image_id(image.id):
            label = labels.get(annotation.category_id)
            if label is None:
                raise ValueError(f"Annotation {annotation.id} references undeclared category {annotation.category_id}")

            instance = None
            if segmentation and not label.keypoint_ids:
                instance = _segmentation_annotation(annotation, label.id, image.height, image.width)
            if instance is None:
                x, y, w, h = annotation.bbox
                instance = BoxAnnotation(id=str(annotation.id), label_id=label.id, value=Box(x_min=x, y_min=y, x_max=x + w, y_max=y + h))
            annotations.append(instance)

            if not annotation.keypoints:
                continue
            if not label.keypoint_ids:
                raise ValueError(f"Annotation {annotation.id} has keypoints but category '{label.display_name}' declares none")
            observed = len(annotation.keypoints) // COORDINATE_DIMENSIONS
            if observed != len(label.keypoint_ids):
                raise ValueError(
                    f"Annotation {annotation.id} has {observed} keypoints for the "
                    f"{len(label.keypoint_ids)} declared by '{label.display_name}'"
                )
            for index, keypoint_id in enumerate(label.keypoint_ids):
                x, y, visibility = annotation.keypoints[index * COORDINATE_DIMENSIONS : (index + 1) * COORDINATE_DIMENSIONS]
                if visibility == VISIBILITY_UNOBSERVED:
                    continue
                annotations.append(
                    KeypointAnnotation(
                        id=f"{annotation.id}-{keypoint_id}",
                        label_id=keypoint_id,
                        value=Point2d(x=x, y=y, visibility=int(visibility)),
                        bounding_box_id=instance.id,
                    )
                )

        files.append(
            FileAnnotations(id=str(image.id), path=image.file_name, height=image.height, width=image.width, annotations=annotations)
        )

    pose = any(label.keypoint_ids for label in labels.values())
    return Dataset(
        labels=list(labels.values()),
        files=files,
        coordinate_dimensions=COORDINATE_DIMENSIONS if pose else None,
        keypoints=keypoint_names,
    )


def convert_coco_to_json(
    coco_file: Path,
    output_json: Path,
    class_map_file: Optional[Path] = None,
    flip_map_file: Optional[Path] = None,
    skeleton_base: int = 1,
    segmentation: bool = False,
) -> Dataset:
    """Convert a COCO dataset into the AIS dataset json, from which json_to_factory writes a Factory dataset."""
    coco = CocoDataset.load_from_json(str(coco_file))
    dataset = build_dataset(
        coco,
        load_class_map(class_map_file),
        load_flip_map(flip_map_file),
        skeleton_base,
        segmentation,
    )
    dataset.save(str(output_json))
    logger.info(f"Converted {len(dataset.files)} file(s) and {len(dataset.labels)} label(s) to {output_json}.")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert a COCO dataset to the AIS dataset json.")
    parser.add_argument("--coco_file", "-j", type=Path, required=True, help="Path to the input COCO annotations file.")
    parser.add_argument("--image_dir", "-i", type=Path, required=True, help="Path to the directory the file names are relative to.")
    parser.add_argument(
        "--output_json",
        "-o",
        type=Path,
        default=None,
        help=f"[optional] path of the output json. Default is <image_dir>/{DEFAULT_JSON_NAME}, where json_to_factory reads it.",
    )
    parser.add_argument(
        "--class_map",
        type=Path,
        default=None,
        help="[optional] a json file mapping COCO category names to Factory class ids. Default is to use the names.",
    )
    parser.add_argument(
        "--flip_map",
        type=Path,
        default=None,
        help="[optional] a json file mapping a Factory class id to its horizontal mirror pairs, each a pair of "
        "keypoint names. Classes left out declare no symmetry, which blocks horizontal flipping during training.",
    )
    parser.add_argument(
        "--skeleton_base",
        type=int,
        choices=(0, 1),
        default=1,
        help="[optional] the first keypoint index used by the file's skeletons. Default is 1, the COCO convention.",
    )
    parser.add_argument(
        "--segmentation",
        action="store_true",
        help="convert instances of classes declaring no keypoints to polygons or bitmasks instead of boxes",
    )
    args = parser.parse_args()

    convert_coco_to_json(
        args.coco_file,
        args.output_json or args.image_dir / DEFAULT_JSON_NAME,
        class_map_file=args.class_map,
        flip_map_file=args.flip_map,
        skeleton_base=args.skeleton_base,
        segmentation=args.segmentation,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
