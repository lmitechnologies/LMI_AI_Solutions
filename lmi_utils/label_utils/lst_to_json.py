import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from lmi_utils.dataset_utils.pose_identifiers import pose_id_from_label_alias
from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    Dataset,
    FileAnnotations,
    KeypointAnnotation,
    Label,
    Mask,
    Point2d,
    Polygon,
)
from lmi_utils.label_utils.bbox_utils import convert_from_ls
from lmi_utils.label_utils.json_to_factory import DATASET_META_FILE, scaffold_pose_schema
from lmi_utils.system_utils.path_utils import get_relative_paths

logger = logging.getLogger(__name__)


LABEL_NAME = "labels.json"
PRED_NAME = "preds.json"
IMAGES_DIR = "images"
# Older exports may express ownership as a relation. Native KeyPointLabels nesting uses `parentID`.
RELATION_TYPE = "relation"


def lst_to_shape(result: dict, fname: str, load_confidence=False):
    """parse the result from label studio result dict, return a Shape object"""
    result_type = result["type"]
    labels = result["value"][result_type]
    if len(labels) > 1:
        raise Exception("Not support more than one labels in a bbox/polygon")
    if len(labels) == 0:
        logger.warning(f"found empty label in {fname}, skip")
        return None, None, None, None

    # A project Factory generated stores each label under its own alias, so a region names an id rather than the
    # text the annotator saw -- which is what lets a class or keypoint be renamed without stranding annotations.
    # A value from any other project is outside that namespace and passes through as it stands.
    label = pose_id_from_label_alias(labels[0])
    conf = result["value"].get("score", 1.0) if load_confidence else 1.0
    if result_type == "rectanglelabels":
        # get bbox
        x, y, w, h, angle = convert_from_ls(result)
        x1, y1, w, h = list(map(int, [x, y, w, h]))
        x2, y2 = x1 + w, y1 + h
        box = Box(x_min=x1, y_min=y1, x_max=x2, y_max=y2, angle=angle)
        return box, label, conf, AnnotationType.BOX
    elif result_type == "polygonlabels":
        points = result["value"]["points"]
        points_np = np.array(points)
        xs = (points_np[:, 0] / 100 * result["original_width"]).astype(np.int32)
        ys = (points_np[:, 1] / 100 * result["original_height"]).astype(np.int32)
        points_np = np.stack([xs, ys], axis=1)
        return (
            Polygon(points=points_np.astype(int).tolist()),
            label,
            conf,
            AnnotationType.POLYGON,
        )
    elif result_type == "brushlabels":
        # Only brush masks need the Label Studio SDK, so every other export converts without it installed.
        from label_studio_sdk.converter.brush import decode_rle

        rle = result["value"]["rle"]
        h, w = result["original_height"], result["original_width"]
        img = decode_rle(rle).reshape(h, w, 4)[:, :, 3]
        mask = img > 128
        return Mask(mask=mask), label, conf, AnnotationType.MASK
    elif result_type == "keypointlabels":
        dt = result["value"]
        x, y = (
            dt["x"] / 100 * result["original_width"],
            dt["y"] / 100 * result["original_height"],
        )
        return Point2d(x=x, y=y), label, conf, AnnotationType.KEYPOINT
    else:
        logger.warning(f"unsupported result type: {result_type}, skip")
        return None, None, None, None


def load_pose_schema(path: Optional[Union[str, Path]]) -> Optional[dict]:
    """Read a declared pose schema from an `annotationSchema` object, or from the `.meta.json` holding one.

    A Label Studio export states no keypoint layout: its labeling configuration declares one flat keypoint
    vocabulary for the whole project, with no per-class order, flip or skeleton. The declaration therefore has to
    come from outside the export -- normally the `.meta.json` of the Factory dataset the project was built from.
    """
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        path = path / DATASET_META_FILE
    with open(path) as f:
        data = json.load(f)
    return data.get("annotationSchema", data)


def pose_labels(pose_schema: Optional[dict]) -> Tuple[List[Label], Set[str], Dict[str, str], Dict[AnnotationType, Dict[str, str]]]:
    """Return the labels, slots, names and display-name-to-id mappings a pose schema declares."""
    labels: List[Label] = []
    vocabulary: Set[str] = set()
    ids_by_name: Dict[AnnotationType, Dict[str, str]] = {AnnotationType.BOX: {}, AnnotationType.KEYPOINT: {}}
    keypoint_names = {
        str(keypoint_id): (definition or {}).get("name") or str(keypoint_id)
        for keypoint_id, definition in ((pose_schema or {}).get("keypoints") or {}).items()
    }
    for keypoint_id, name in keypoint_names.items():
        previous = ids_by_name[AnnotationType.KEYPOINT].setdefault(name, keypoint_id)
        if previous != keypoint_id:
            raise ValueError(f"Pose schema keypoints '{previous}' and '{keypoint_id}' are both named '{name}'")
    for class_id, declaration in (pose_schema or {}).get("classes", {}).items():
        class_id = str(class_id)
        class_name = declaration.get("name") or class_id
        keypoint_ids = [str(keypoint_id) for keypoint_id in declaration.get("keypointIds") or []]
        labels.append(
            Label(
                id=class_id,
                name=class_name,
                annotation_type=AnnotationType.BOX,
                keypoint_ids=keypoint_ids or None,
                horizontal_flip_pairs=declaration.get("horizontalFlipPairs"),
                skeleton=declaration.get("skeleton"),
            )
        )
        previous = ids_by_name[AnnotationType.BOX].setdefault(class_name, class_id)
        if previous != class_id:
            raise ValueError(f"Pose schema classes '{previous}' and '{class_id}' are both named '{class_name}'")
        vocabulary.update(keypoint_ids)
    return labels, vocabulary, keypoint_names, ids_by_name


def link_keypoints(relations: List[dict], by_region_id: dict):
    """Resolve legacy Label Studio relations into keypoint-to-box links.

    Either direction links the pair; relations between other region types are ignored. Native `parentID`
    ownership is authoritative, so a relation only fills a still-unlinked keypoint.
    """
    for relation in relations:
        ends = [by_region_id.get(relation.get("from_id")), by_region_id.get(relation.get("to_id"))]
        keypoint = next((a for a in ends if isinstance(a, KeypointAnnotation)), None)
        box = next((a for a in ends if a is not None and a.type == AnnotationType.BOX), None)
        if keypoint is not None and box is not None and keypoint.bounding_box_id is None:
            keypoint.bounding_box_id = box.id


def link_keypoint_parents(parent_links: List[Tuple[KeypointAnnotation, str]], by_region_id: dict):
    """Resolve native Label Studio `parentID` ownership after every region in the result has been indexed."""
    for keypoint, parent_id in parent_links:
        parent = by_region_id.get(parent_id)
        if parent is None:
            raise ValueError(f"Keypoint region {keypoint.id} has parentID '{parent_id}', but that region does not exist")
        if parent.type != AnnotationType.BOX:
            raise ValueError(f"Keypoint region {keypoint.id} has parentID '{parent_id}', which is not a box")
        keypoint.bounding_box_id = parent.id


def check_keypoint_vocabulary(files: List[FileAnnotations], vocabulary: Set[str]):
    """Every keypoint must name a slot some class declares, otherwise it reaches Factory as a stray label."""
    unknown = sorted(
        {
            annotation.label_id
            for file in files
            for annotation in file.annotations + file.predictions
            if annotation.type == AnnotationType.KEYPOINT and annotation.label_id not in vocabulary
        }
    )
    if unknown:
        raise ValueError(f"Keypoints {unknown} are not declared by any class in the pose schema")


def to_linux_path(path: Union[str, Path]):
    """convert windows path to linux (POSIX) path"""
    return Path(path).as_posix()


def build_image_index(images_dir) -> dict:
    """Local images keyed by every name an export might identify them by: the path relative to `images_dir`, its
    basename, and its basename without the extension."""
    index = {}
    for relative in get_relative_paths(images_dir):
        relative = to_linux_path(relative)
        name = os.path.basename(relative)
        for key in (relative, name, os.path.splitext(name)[0]):
            index.setdefault(key, relative)
    return index


def resolve_image(url: str, common_prefix: str, index: dict) -> str:
    """The local image an exported task refers to, as a path relative to the image directory.

    Label Studio keeps whatever the import gave it, which is a file path for a local import but an API URL for a
    project Factory created -- `.../items/<item id>/image`, whose own tail is a fixed word. The component before
    it identifies the item, so it is tried too.
    """
    candidates = []
    if common_prefix and url.startswith(common_prefix):
        candidates.append(url[len(common_prefix) :].lstrip("/"))
    parts = [part for part in to_linux_path(url).split("/") if part]
    if parts:
        candidates += [parts[-1], os.path.splitext(parts[-1])[0]]
    if len(parts) > 1:
        candidates.append(parts[-2])

    for candidate in candidates:
        resolved = index.get(candidate)
        if resolved is not None:
            return resolved
    raise FileNotFoundError(
        f"No image matches '{url}'. Tried {candidates}. Check that the image directory holds the images this "
        f"project was annotated on, named as the export identifies them."
    )


def declared_size(task: dict) -> Optional[Tuple[int, int]]:
    """The image size the export recorded, or None when none of the task's results state one."""
    for group in list(task.get("annotations", [])) + list(task.get("predictions", [])):
        for result in group.get("result") or []:
            if "original_width" in result and "original_height" in result:
                return result["original_width"], result["original_height"]
    return None


def generate_file_ids(files: List[str]):
    file_id = {}
    for i, f in enumerate(files):
        file_id[to_linux_path(f)] = i
    return file_id


def collect_results(
    results,
    out_list,
    counter,
    label_dict,
    labels,
    fname,
    load_confidence=False,
    keypoint_vocabulary=None,
    label_ids_by_name=None,
):
    """Parse results, append Annotations to out_list, return updated counter.

    Region ids and relations are scoped to the one completion these results belong to, so links are resolved here
    rather than across a whole file.
    """
    keypoint_vocabulary = keypoint_vocabulary or set()
    label_ids_by_name = label_ids_by_name or {}
    by_region_id = {}
    relations = []
    parent_links = []
    for result in results:
        if result.get("type") == RELATION_TYPE:
            relations.append(result)
            continue
        shape, label, conf, annot_type = lst_to_shape(result, fname, load_confidence=load_confidence)
        if shape is None:
            continue
        label = label_ids_by_name.get(annot_type, {}).get(label, label)
        if label not in label_dict and label not in keypoint_vocabulary:
            # A declared keypoint names a slot inside its class's layout, not a class of its own.
            label_dict.add(label)
            labels.append(Label(id=str(label), annotation_type=annot_type))
        fields = dict(id=str(counter), label_id=str(label), value=shape, confidence=conf if load_confidence else None)
        if annot_type == AnnotationType.KEYPOINT:
            annotation = KeypointAnnotation(**fields)
        else:
            annotation = Annotation(type=annot_type, **fields)
        out_list.append(annotation)
        if result.get("id") is not None:
            by_region_id[result["id"]] = annotation
        if isinstance(annotation, KeypointAnnotation) and result.get("parentID") is not None:
            parent_links.append((annotation, str(result["parentID"])))
        counter += 1
    link_keypoint_parents(parent_links, by_region_id)
    link_keypoints(relations, by_region_id)
    return counter


def get_annotations_from_json(path_json, images_dir, background=False, pose_schema: Optional[dict] = None):
    """read annotation from label studio json file.

    Args:
        path_json (str): the path to a directory of label studio json files
        pose_schema (dict): the declared pose schema, whose classes seed the labels and whose keypoints are
            read as slots of those classes rather than as classes of their own

    Returns:
        dict: a map <image name, a list of Rect objects>
    """
    if os.path.splitext(path_json)[1] == ".json":
        json_files = [path_json]
    else:
        json_files = glob.glob(os.path.join(path_json, "*.json"))
    json_files = [path for path in json_files if os.path.basename(path) not in (LABEL_NAME, PRED_NAME)]
    if not json_files:
        raise FileNotFoundError(
            f"No Label Studio JSON exports found at '{path_json}'. Files named '{LABEL_NAME}' and '{PRED_NAME}' are reserved outputs."
        )

    labels, keypoint_vocabulary, keypoint_names, label_ids_by_name = pose_labels(pose_schema)
    annotations: List[FileAnnotations] = []

    label_set = {label.id for label in labels}
    file_id_dict = generate_file_ids(get_relative_paths(images_dir))
    image_index = build_image_index(images_dir)
    # Guards the loose name matching in resolve_image: two tasks resolving to one image is a mismatched directory.
    claimed_by = {}
    processed_files = set()

    for path_json in json_files:
        logger.info(f"Extracting labels from: {path_json}")
        logger.info(f"dir_path : {images_dir}")
        with open(path_json) as f:
            li = json.load(f)
        if not isinstance(li, list):
            raise ValueError(
                f"'{path_json}' is not a Label Studio JSON export: expected a top-level array of tasks, got {type(li).__name__}."
            )

        cnt_anno = 0
        cnt_image = 0
        cnt_pred = 0
        cnt_wrong = 0

        # collect all the files
        files = [dt["data"]["image"] for dt in li if "data" in dt]
        # find common string-based prefix to handle cloud path and local path.
        common_prefix = os.path.dirname(os.path.commonprefix(files))
        logger.info(f"common prefix of image paths in json: {common_prefix}")

        for dt in li:
            # load file name
            if "data" not in dt:
                raise ValueError('missing "data" in json file. Ensure that the label studio export format is not JSON-MIN.')
            f = dt["data"]["image"]  # image web path. already in linux path format
            file_annotations: List[Annotation] = []
            pred_annotations: List[Annotation] = []

            if "annotations" in dt:
                cnt = 0
                for annot in dt["annotations"]:
                    if len(annot["result"]) > 0:
                        cnt += 1
                    cnt_anno = collect_results(
                        annot["result"],
                        file_annotations,
                        cnt_anno,
                        label_set,
                        labels,
                        f,
                        keypoint_vocabulary=keypoint_vocabulary,
                        label_ids_by_name=label_ids_by_name,
                    )

                    if "prediction" in annot and "result" in annot["prediction"]:
                        cnt_pred = collect_results(
                            annot["prediction"]["result"],
                            pred_annotations,
                            cnt_pred,
                            label_set,
                            labels,
                            f,
                            load_confidence=True,
                            keypoint_vocabulary=keypoint_vocabulary,
                            label_ids_by_name=label_ids_by_name,
                        )
                if cnt == 0 and dt.get("total_annotations", 0) > 0:
                    cnt_wrong += 1
                    logger.warning(f"found 0 annotation in {f}, but lst claims total_annotations = {dt.get('total_annotations')}")

            if "predictions" in dt:
                for pred in dt["predictions"]:
                    if isinstance(pred, dict):
                        cnt_pred = collect_results(
                            pred["result"],
                            pred_annotations,
                            cnt_pred,
                            label_set,
                            labels,
                            f,
                            load_confidence=True,
                            keypoint_vocabulary=keypoint_vocabulary,
                            label_ids_by_name=label_ids_by_name,
                        )

            url, f = f, resolve_image(f, common_prefix, image_index)
            if f in claimed_by:
                raise ValueError(f"Tasks '{claimed_by[f]}' and '{url}' both resolve to the image '{f}'")
            claimed_by[f] = url
            updated_fp = os.path.join(images_dir, f)

            file_id = file_id_dict.get(f)
            if file_id is None:
                raise KeyError(f"key {f} not found in dict. Sample keys: {list(file_id_dict.keys())[:5]}.")

            image = cv2.imread(updated_fp, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"failed to read image: {updated_fp}")
            height, width = image.shape[:2]
            # Every coordinate is a percentage of the size the export recorded, so a different image here is not
            # a near miss -- it rescales the whole task.
            declared = declared_size(dt)
            if declared is not None and declared != (width, height):
                raise ValueError(f"'{f}' is {width}x{height}, but '{url}' was annotated on a {declared[0]}x{declared[1]} image")
            if file_annotations or pred_annotations:
                file_record = FileAnnotations(
                    id=str(file_id),
                    path=f,
                    height=height,
                    width=width,
                    annotations=file_annotations,
                    predictions=pred_annotations,
                )
                # A keypoint drawn in Label Studio is a top-level region: `parentID` exists only where a task was
                # seeded with linked keypoints or an annotator nested them by hand, so most exports state
                # ownership through geometry alone. Neither unowned nor ambiguous keypoints stop the conversion
                # here -- Label Studio projects legitimately hold standalone keypoints, and the export is worth
                # reading whole -- but they reach Factory unlinked, which rejects them by name.
                file_record.assign_keypoints(unassigned="keep", ambiguous="keep")
                annotations.append(file_record)
                if file_annotations:
                    cnt_image += 1
                else:
                    logger.warning(f"no annotation found in {f}")
            else:
                logger.warning(f"no annotation found in {f}")
                if background:
                    annotations.append(FileAnnotations(id=str(file_id), path=f, height=height, width=width))

            processed_files.add(f)

        logger.info(f"{cnt_image} out of {len(li)} images have annotations")
        if cnt_wrong > 0:
            logger.info(f"{cnt_wrong} images with total_annotations > 0, but found 0 annotation")
        logger.info(f"total {cnt_anno} annotations")
        logger.info(f"total {cnt_pred} predictions")
    # save background images not present in any json file
    if background:
        for f in file_id_dict:
            if f in processed_files:
                continue
            updated_fp = os.path.join(images_dir, f)
            if not os.path.isfile(updated_fp):
                raise FileNotFoundError(
                    f"Not found '{f}' in '{images_dir}'. Check if the folder structure of the images is the same as the path in json file."
                )
            file_id = file_id_dict[f]
            image = cv2.imread(updated_fp, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"failed to read image: {updated_fp}")
            height, width = image.shape[:2]
            annotations.append(FileAnnotations(id=str(file_id), path=f, height=height, width=width))

    logger.info(f"total {len(annotations)} images")
    logger.info(f"total {len(labels)} labels")

    if pose_schema is not None:
        check_keypoint_vocabulary(annotations, keypoint_vocabulary)
    elif any(file.get_annotations_by_type(AnnotationType.KEYPOINT) for file in annotations):
        # Factory rejects keypoints with no declared layout, so callers using this function directly need the
        # same warning the CLI turns into an error below.
        logger.warning(
            "This export has keypoints but no pose schema was given. The result is not a valid Factory pose dataset; "
            "pass -ps, or use --scaffold_schema to draft and apply the latest pose schema."
        )

    return annotations, labels, keypoint_names


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser("Convert label studio json file to json format")
    ap.add_argument(
        "-i",
        "--path_json",
        required=True,
        help="the directory of label-studio json files",
    )
    ap.add_argument("-imgs", "--path_images", required=True, help="the root directory of images")
    ap.add_argument("-of", "--path_out_json", required=True, help="path to store the json file")
    ap.add_argument("-bg", "--background", action="store_true", help="save background")
    ap.add_argument(
        "-ps",
        "--pose_schema",
        help="the declared pose schema: a Factory dataset directory, its .meta.json, or an annotationSchema json",
    )
    ap.add_argument(
        "--scaffold_schema",
        help="[optional] also write a draft pose schema surveyed from the annotations, to edit and pass back as -ps",
    )
    args = ap.parse_args()

    pose_schema = load_pose_schema(args.pose_schema)
    files, labels, keypoint_names = get_annotations_from_json(
        args.path_json, args.path_images, background=args.background, pose_schema=pose_schema
    )

    dataset = Dataset(
        labels=labels,
        files=files,
        coordinate_dimensions=(pose_schema or {}).get("coordinateDimensions"),
        keypoints=keypoint_names,
    )

    if args.scaffold_schema:
        draft = scaffold_pose_schema(dataset)
        with open(args.scaffold_schema, "w") as f:
            json.dump({"annotationSchema": draft}, f, indent=4)
        logger.warning(
            f"Wrote a draft schema of {len(draft['classes'])} class(es) to {args.scaffold_schema}. Its keypoint order "
            f"is only what the annotations showed and horizontalFlipPairs is null for every class: review it, then pass it as -ps."
        )
        if pose_schema is None and any(file.get_annotations_by_type(AnnotationType.KEYPOINT) for file in files):
            pose_schema = draft
            files, labels, keypoint_names = get_annotations_from_json(
                args.path_json, args.path_images, background=args.background, pose_schema=pose_schema
            )
            dataset = Dataset(
                labels=labels,
                files=files,
                coordinate_dimensions=pose_schema.get("coordinateDimensions"),
                keypoints=keypoint_names,
            )

    if pose_schema is None and any(file.get_annotations_by_type(AnnotationType.KEYPOINT) for file in files):
        raise ValueError("Keypoint exports require -ps/--pose_schema or --scaffold_schema in the latest AIS JSON format.")

    out_path = args.path_out_json
    if not out_path.endswith(".json"):
        raise Exception("output path should end with .json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dataset.save(out_path)
    logger.info(f"saved to {out_path}")


if __name__ == "__main__":
    main()
