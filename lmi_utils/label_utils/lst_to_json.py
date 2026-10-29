import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import List, Union

import cv2

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
    link_keypoints,
)
from lmi_utils.label_utils.bbox_utils import convert_from_ls, convert_points_from_ls
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
        x, y, w, h, angle = convert_from_ls(result)
        box = Box(x_min=x, y_min=y, x_max=x + w, y_max=y + h, angle=angle)
        return box, label, conf, AnnotationType.BOX
    elif result_type == "polygonlabels":
        return Polygon(points=convert_points_from_ls(result)), label, conf, AnnotationType.POLYGON
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

    A keypoint links to the box it is nested under in Label Studio, else as representations.link_keypoints does.

    Returns:
        tuple: the updated counter, and a message for each keypoint left unlinked
    """
    regions = {}
    parents = []
    listed = []
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
        listed.append(annotation)
        counter += 1
        if result.get("id") is not None:
            # a shared id names no single region
            regions[result["id"]] = None if result["id"] in regions else annotation
        if annot_type == AnnotationType.KEYPOINT and result.get("parentID") is not None:
            parents.append((annotation, result["parentID"]))
    out_list.extend(listed)

    errors = []
    unlinkable = set()
    for keypoint, parent_id in parents:
        where = f"keypoint '{keypoint.label_id}' at ({keypoint.value.x:.1f}, {keypoint.value.y:.1f})"
        parent = regions.get(parent_id)
        if parent_id not in regions:
            errors.append(f"{where} is nested under region {parent_id}, which is not a labeled region of the same annotation")
        elif parent is None:
            errors.append(f"{where} is nested under region {parent_id}, an id several regions share")
        elif parent.type != AnnotationType.BOX:
            errors.append(f"{where} is nested under a '{parent.label_id}' region, which is not a box")
        else:
            keypoint.bounding_box_id = parent.id
            continue
        unlinkable.add(keypoint.id)
    for keypoint, message in link_keypoints([a for a in listed if a.id not in unlinkable]):
        keypoint.bounding_box_id = None
        if message not in errors:
            errors.append(message)
    return counter, [f"{fname}: {error}" for error in errors]


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
