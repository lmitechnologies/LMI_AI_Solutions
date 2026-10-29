import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

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
from lmi_utils.label_utils.bbox_utils import convert_from_ls, ls_box_to_pixels, ls_points_to_pixels
from lmi_utils.label_utils.lst_brush import decode_mask
from lmi_utils.system_utils.path_utils import get_relative_paths

logger = logging.getLogger(__name__)


LABEL_NAME = "labels.json"
PRED_NAME = "preds.json"
IMAGES_DIR = "images"


def lst_to_shape(result: dict, fname: str, load_confidence=False):
    """parse the result from label studio result dict, return a Shape object"""
    result_type = result["type"]
    labels = result["value"][result_type]
    if len(labels) > 1:
        raise Exception("Not support more than one labels in a bbox/polygon")
    if len(labels) == 0:
        logger.warning(f"found empty label in {fname}, skip")
        return None, None, None, None

    label = labels[0]
    conf = result["value"].get("score", 1.0) if load_confidence else 1.0
    if result_type == "rectanglelabels":
        # get bbox
        x1, y1, x2, y2, angle = ls_box_to_pixels(result)
        box = Box(x_min=x1, y_min=y1, x_max=x2, y_max=y2, angle=angle)
        return box, label, conf, AnnotationType.BOX
    elif result_type == "polygonlabels":
        xs, ys = ls_points_to_pixels(result)
        points_np = np.stack([xs, ys], axis=1)
        return (
            Polygon(points=points_np.astype(int).tolist()),
            label,
            conf,
            AnnotationType.POLYGON,
        )
    elif result_type == "brushlabels":
        rle = result["value"]["rle"]
        h, w = result["original_height"], result["original_width"]
        mask = decode_mask(rle, h, w)
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


def to_linux_path(path: Union[str, Path]):
    """convert windows path to linux (POSIX) path"""
    return Path(path).as_posix()


def generate_file_ids(files: List[str]):
    file_id = {}
    for i, f in enumerate(files):
        file_id[to_linux_path(f)] = i
    return file_id


def collect_results(results, out_list, counter, label_dict, labels, fname, load_confidence=False):
    """Parse one Label Studio result list, append its Annotations to out_list, and link its keypoints to their boxes.

    Returns:
        tuple: the updated counter, and a message for each keypoint that could not be linked (see link_keypoints)
    """
    regions = {}
    boxes = []
    keypoints = []
    for result in results:
        shape, label, conf, annot_type = lst_to_shape(result, fname, load_confidence=load_confidence)
        if shape is None:
            continue
        if label not in label_dict:
            label_dict.add(label)
            labels.append(Label(id=str(label), annotation_type=annot_type))
        annotation_cls = KeypointAnnotation if annot_type == AnnotationType.KEYPOINT else Annotation
        annotation = annotation_cls(
            id=str(counter),
            label_id=str(label),
            type=annot_type,
            value=shape,
            confidence=conf if load_confidence else None,
        )
        out_list.append(annotation)
        counter += 1
        if result.get("id") is not None:
            # a shared id names no single region
            regions[result["id"]] = None if result["id"] in regions else annotation
        if annot_type == AnnotationType.BOX:
            x, y, w, h, _ = convert_from_ls(result)
            boxes.append((annotation, (x, y, x + w, y + h)))
        elif annot_type == AnnotationType.KEYPOINT:
            keypoints.append((annotation, result.get("parentID")))
    return counter, link_keypoints(keypoints, boxes, regions, fname)


def link_keypoints(keypoints, boxes, regions, fname) -> List[str]:
    """Set each keypoint's bounding_box_id: the box it is nested under in Label Studio, else the one box that contains it.

    Args:
        keypoints (list): (KeypointAnnotation, the Label Studio id of its parent region or None)
        boxes (list): (box Annotation, unrounded (x1, y1, x2, y2)); truncated bounds put a keypoint on a shared edge in both boxes
        regions (dict): Label Studio result id to its Annotation, or None for an id several results share
        fname (str): the image, for messages

    Returns:
        list: a message for each keypoint left unlinked; keypoints repeating a name on one box are all unlinked
    """
    errors = []
    rotated = [box for box, _ in boxes if box.value.angle]
    for keypoint, parent_id in keypoints:
        where = f"{fname}: keypoint '{keypoint.label_id}' at ({keypoint.value.x:.1f}, {keypoint.value.y:.1f})"
        if parent_id is not None:
            if parent_id not in regions:
                errors.append(f"{where} is nested under region {parent_id}, which is not a labeled region of the same annotation")
                continue
            box = regions[parent_id]
            if box is None:
                errors.append(f"{where} is nested under region {parent_id}, an id several regions share")
                continue
            if box.type != AnnotationType.BOX:
                errors.append(f"{where} is nested under a '{box.label_id}' region, which is not a box")
                continue
            if box.value.angle:
                errors.append(f"{where} is nested under a rotated '{box.label_id}' box; keypoints on rotated boxes are not supported")
                continue
        elif rotated:
            errors.append(f"{where} is not nested under a box, and cannot be linked by position next to rotated boxes")
            continue
        else:
            x, y = keypoint.value.x, keypoint.value.y
            inside = [box for box, (x1, y1, x2, y2) in boxes if x1 <= x <= x2 and y1 <= y <= y2]
            if len(inside) != 1:
                found = ", ".join(f"'{b.label_id}' {_bounds(b)}" for b in inside) if inside else "no box"
                errors.append(f"{where} is not nested under a box and lies inside {found}")
                continue
            box = inside[0]
        keypoint.bounding_box_id = box.id

    boxes_by_id = {box.id: box for box, _ in boxes}
    by_slot = {}
    for keypoint, _ in keypoints:
        if keypoint.bounding_box_id is not None:
            by_slot.setdefault((keypoint.bounding_box_id, keypoint.label_id), []).append(keypoint)
    for (box_id, name), repeated in by_slot.items():
        if len(repeated) > 1:
            box = boxes_by_id[box_id]
            errors.append(f"{fname}: the '{box.label_id}' box {_bounds(box)} has {len(repeated)} keypoints '{name}'")
            for keypoint in repeated:
                keypoint.bounding_box_id = None
    return errors


def _bounds(box: Annotation) -> str:
    return f"({box.value.x_min:.0f}, {box.value.y_min:.0f}, {box.value.x_max:.0f}, {box.value.y_max:.0f})"


def get_annotations_from_json(path_json, images_dir, background=False, keep_unlinked_keypoints=False):
    """read annotation from label studio json file.

    Args:
        path_json (str): the path to a directory of label studio json files
        keep_unlinked_keypoints (bool): warn about annotated keypoints that cannot be linked to a box, instead of raising

    Returns:
        dict: a map <image name, a list of Rect objects>

    Raises:
        ValueError: if an annotated keypoint cannot be linked to its box, listing every such keypoint
    """
    if os.path.splitext(path_json)[1] == ".json":
        json_files = [path_json]
    else:
        json_files = glob.glob(os.path.join(path_json, "*.json"))

    labels: List[Label] = []
    annotations: List[FileAnnotations] = []

    label_set = set()
    link_errors = []
    file_id_dict = generate_file_ids(get_relative_paths(images_dir))
    processed_files = set()

    for path_json in json_files:
        if path_json.endswith(LABEL_NAME) or path_json.endswith(PRED_NAME):
            continue
        logger.info(f"Extracting labels from: {path_json}")
        logger.info(f"dir_path : {images_dir}")
        with open(path_json) as f:
            li = json.load(f)

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
                    cnt_anno, errors = collect_results(annot["result"], file_annotations, cnt_anno, label_set, labels, f)
                    link_errors += errors

                    if "prediction" in annot and "result" in annot["prediction"]:
                        cnt_pred, errors = collect_results(
                            annot["prediction"]["result"], pred_annotations, cnt_pred, label_set, labels, f, load_confidence=True
                        )
                        _warn_unlinked_predictions(errors)
                if cnt == 0 and dt.get("total_annotations", 0) > 0:
                    cnt_wrong += 1
                    logger.warning(f"found 0 annotation in {f}, but lst claims total_annotations = {dt.get('total_annotations')}")

            if "predictions" in dt:
                for pred in dt["predictions"]:
                    if isinstance(pred, dict):
                        cnt_pred, errors = collect_results(
                            pred["result"], pred_annotations, cnt_pred, label_set, labels, f, load_confidence=True
                        )
                        _warn_unlinked_predictions(errors)

            f = f[len(common_prefix) :].lstrip("/")
            updated_fp = os.path.join(images_dir, f)
            if not os.path.isfile(updated_fp):
                raise FileNotFoundError(
                    f"Not found '{f}' in '{images_dir}'. Check if the folder structure of images_dir is the same as the path in json file."
                )

            file_id = file_id_dict.get(f)
            if file_id is None:
                raise KeyError(f"key {f} not found in dict. Sample keys: {list(file_id_dict.keys())[:5]}.")

            image = cv2.imread(updated_fp, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"failed to read image: {updated_fp}")
            height, width = image.shape[:2]
            if file_annotations or pred_annotations:
                annotations.append(
                    FileAnnotations(
                        id=str(file_id),
                        path=f,
                        height=height,
                        width=width,
                        annotations=file_annotations,
                        predictions=pred_annotations,
                    )
                )
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

    if link_errors:
        for error in link_errors:
            (logger.warning if keep_unlinked_keypoints else logger.error)(error)
        if not keep_unlinked_keypoints:
            raise ValueError(
                f"{len(link_errors)} keypoint problem(s), listed above. Nest each keypoint under its box in Label Studio's Regions "
                "panel, or pass --keep_unlinked_keypoints to write them unlinked."
            )

    logger.info(f"total {len(annotations)} images")
    logger.info(f"total {len(labels)} labels")

    return annotations, labels


def _warn_unlinked_predictions(errors):
    # a model's keypoints may fall outside its boxes; only the ground truth must link
    for error in errors:
        logger.warning(f"prediction {error}")


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
        "--keep_unlinked_keypoints",
        action="store_true",
        help="write keypoints that cannot be linked to a box unlinked, with a warning, instead of failing; for datasets whose "
        "keypoints are unused, or are linked later by json_to_yolo --target_classes",
    )
    args = ap.parse_args()

    annotations, labels = get_annotations_from_json(
        args.path_json, args.path_images, background=args.background, keep_unlinked_keypoints=args.keep_unlinked_keypoints
    )

    annotations = Dataset(labels=labels, files=annotations)

    out_path = args.path_out_json
    if not out_path.endswith(".json"):
        raise Exception("output path should end with .json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    annotations.save(out_path)
    logger.info(f"saved to {out_path}")


if __name__ == "__main__":
    main()
