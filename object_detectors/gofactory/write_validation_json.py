import argparse
import json
import logging
import os

import cv2
import numpy as np
import torch
from ultralytics.utils import nms, ops
from ultralytics.utils.metrics import box_iou, kpt_iou, mask_iou

from lmi_utils.dataset_utils.ops.dataset_pad import pad_annotated_image
from lmi_utils.dataset_utils.ops.dataset_resize import resize_annotated_image
from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    Dataset,
    Mask,
    Point2d,
    Polygon,
)
from object_detectors.ultralytics_lmi.yolo.model import Yolo, YoloObb, YoloPose, YoloSeg

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "ObjectDetection": Yolo,
    "InstanceSegmentation": YoloSeg,
    "OrientedObjectDetection": YoloObb,
    "KeypointDetection": YoloPose,
}


def parse_annotations(annotations: list[Annotation], h: int, w: int, model_type: str) -> dict:
    """parse label annotations from a list.

    Args:
        annotations (list[Annotation]): a list of Annotation objects
        h (int): image height
        w (int): image width
        model_type (str): the model type

    Returns:
        dict: a dictionary contains 'classes','boxes','masks'
    """
    boxes = []
    masks = []
    points = []
    label_names = []
    for annot in annotations:
        label_names.append(annot.label_id)
        if annot.type == AnnotationType.BOX:
            if model_type == "OrientedObjectDetection":
                boxes.append(annot.value.to_polygon().to_numpy())  # 4 corners (xyxyxyxy) for rotated iou
            elif model_type == "InstanceSegmentation":
                # treat box-class instances as rectangular masks, matching the
                mask = annot.value.to_mask(h=h, w=w).to_numpy(h=h, w=w)
                masks.append(mask)
            else:
                boxes.append(annot.value.to_numpy())
        elif annot.type == AnnotationType.MASK:
            mask = annot.value.to_numpy(h=h, w=w)
            masks.append(mask)
        elif annot.type == AnnotationType.POLYGON:
            if model_type == "InstanceSegmentation":
                obj = annot.value.to_mask(h=h, w=w)
                mask = obj.to_numpy(h=h, w=w)
                masks.append(mask)
            elif model_type == "OrientedObjectDetection":
                poly = annot.value.to_numpy()
                boxes.append(poly)
            else:
                logger.warning(f"Not support loading polygons for the model type: {model_type}, skip")
        elif annot.type == AnnotationType.KEYPOINT:
            points.append(annot.value.to_numpy())
        else:
            raise Exception(f"Not supported type: {type(annot.type)}")
    return {
        "boxes": np.array(boxes),
        "masks": np.array(masks),
        "points": np.array(points),
        "classes": np.array(label_names),
    }


def update_annotation_ids(annotations: list[Annotation], start_id=0):
    """reassign sequential ids to annotations in place, starting from start_id."""
    for i, annot in enumerate(annotations):
        annot.id = str(start_id + i)


def build_prediction_annotations(preds: dict, model_type: str, start_id: int) -> list[Annotation]:
    """build prediction Annotation objects from a model's per-image outputs.

    Args:
        preds (dict): per-image predictions with 'classes','boxes','scores' and optional 'masks','points'
        model_type (str): the model type
        start_id (int): first annotation id; ids are sequential and tentative (reassigned after unpadding)

    Returns:
        list[Annotation]: prediction annotations in padded coordinates
    """
    preds_padded = []
    current_id = start_id
    for i in range(len(preds["classes"])):
        box = preds["boxes"][i]
        mask = preds["masks"][i] if "masks" in preds else None
        label_name = preds["classes"][i]
        score = preds["scores"][i].item()

        if model_type == "InstanceSegmentation":
            dt = dict(
                id=str(current_id),
                label_id=label_name,
                type=AnnotationType.MASK,
                value=Mask(mask),
                confidence=score,
            )
            preds_padded.append(Annotation(**dt))
            current_id += 1
        elif model_type in ["ObjectDetection", "KeypointDetection"]:
            dt = dict(
                id=str(current_id),
                label_id=label_name,
                type=AnnotationType.BOX,
                value=Box(*box, angle=0),
                confidence=score,
            )
            preds_padded.append(Annotation(**dt))
            current_id += 1

            if model_type == "KeypointDetection":
                pts = preds["points"][i]
                for j in range(len(pts)):
                    pt = np.squeeze(pts[j])[:2]  # keep (x, y); drop visibility when kpt_shape is [N, 3]
                    dt = dict(
                        id=str(current_id),
                        label_id=label_name,
                        type=AnnotationType.KEYPOINT,
                        value=Point2d(*pt),
                    )
                    preds_padded.append(Annotation(**dt))
                    current_id += 1
        elif model_type == "OrientedObjectDetection":
            dt = dict(
                id=str(current_id),
                label_id=label_name,
                type=AnnotationType.BOX,
                value=Polygon(points=box).to_rbox(),  # model outputs corners (xyxyxyxy)
                confidence=score,
            )
            preds_padded.append(Annotation(**dt))
            current_id += 1
    return preds_padded


def compute_ious(labels: dict, preds: dict, model_type: str, model) -> dict:
    """compute the iou matrix between parsed labels and predictions.

    Args:
        labels (dict): parsed label annotations (see parse_annotations)
        preds (dict): parsed prediction annotations (see parse_annotations)
        model_type (str): the model type
        model: the loaded model (used for device and kpt_shape)

    Returns:
        dict: {'n_gt','n_pred','ious'}; for KeypointDetection also
            {'n_gt_kpt','n_pred_kpt','ious_kpt'}. 'ious'/'ious_kpt' are tensors or None.
    """
    ious = None
    if model_type == "InstanceSegmentation":
        n_gt = len(labels["masks"])
        n_pred = len(preds["masks"])
        if n_gt and n_pred:
            gt_masks = torch.from_numpy(labels["masks"]).float().to(model.device)
            pred_masks = torch.from_numpy(preds["masks"]).float().to(model.device)
            ious = mask_iou(
                gt_masks.view(gt_masks.shape[0], -1),
                pred_masks.view(pred_masks.shape[0], -1),
            )
        return {"n_gt": n_gt, "n_pred": n_pred, "ious": ious}

    if model_type in ["ObjectDetection", "KeypointDetection"]:
        n_gt = len(labels["boxes"])
        n_pred = len(preds["boxes"])
        gt_boxes = None
        if n_gt and n_pred:
            gt_boxes = torch.from_numpy(labels["boxes"][:, :-1]).to(model.device)
            pred_boxes = torch.from_numpy(preds["boxes"][:, :-1]).to(model.device)
            ious = box_iou(gt_boxes, pred_boxes)
        result = {"n_gt": n_gt, "n_pred": n_pred, "ious": ious}
        if model_type == "KeypointDetection":
            ious_kpt = None
            n_gt_kpt = len(labels["points"])
            n_pred_kpt = len(preds["points"])
            if n_gt_kpt and n_pred_kpt:
                nkpt = model.model.kpt_shape[0]
                # parsed points are always (x, y); reshape with 2, independent of kpt_shape's visibility dim
                labels["points"] = labels["points"].reshape(-1, nkpt, 2)  # (N, n_kp, 2)
                preds["points"] = preds["points"].reshape(-1, nkpt, 2)  # (M, n_kp, 2)
                # add ones to the last dimension for visibility
                # TODO: update point2d in data schema to include visibility
                gt_points = torch.from_numpy(labels["points"]).to(model.device)
                gt_points = torch.cat((gt_points, torch.ones_like(gt_points[..., :-1])), dim=-1)  # (N, n_kp, 3)

                pred_points = torch.from_numpy(preds["points"]).to(model.device)
                pred_points = torch.cat((pred_points, torch.ones_like(pred_points[..., :-1])), dim=-1)  # (M, n_kp, 3)
                # `0.53` is from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L181
                area = ops.xyxy2xywh(gt_boxes)[:, 2:].prod(1) * 0.53
                sigma = np.ones(nkpt) / nkpt
                ious_kpt = kpt_iou(gt_points, pred_points, sigma=sigma, area=area)
            result.update(n_gt_kpt=n_gt_kpt, n_pred_kpt=n_pred_kpt, ious_kpt=ious_kpt)
        return result

    if model_type == "OrientedObjectDetection":
        n_gt = len(labels["boxes"])
        n_pred = len(preds["boxes"])
        if n_gt and n_pred:
            gt = labels["boxes"].astype(np.int32)
            gt = torch.from_numpy(gt).to(model.device)
            gt2 = ops.xyxyxyxy2xywhr(gt)
            pred = preds["boxes"].astype(np.int32)
            pred = torch.from_numpy(pred).to(model.device)
            pred2 = ops.xyxyxyxy2xywhr(pred)
            ious = nms.batch_probiou(gt2, pred2)
        return {"n_gt": n_gt, "n_pred": n_pred, "ious": ious}

    raise Exception(f"Not supported model type: {model_type}")


def write_json(
    model_path,
    model_type,
    config_path,
    image_dir,
    label_path,
    out_pred_json,
    out_image_dir,
    out_iou_dir,
    image_size: tuple[int, int] | None,
    confidence=0.01,
    iou=0.45,
    max_det=600,
):
    """write predictions and labels to a json file

    Args:
        model_path (str): a path to a model weights file
        model_type (str): a type of the model, either "ObjectDetection", "OrientedObjectDetection",
            "InstanceSegmentation", "KeypointDetection"
        config_path (str): a path to a model configuration file
        image_dir (str): a input image directory, where each image should have the same dimension as training images
        label_path (str): a path to a label json file
        out_pred_json (str): a full output json file path
        out_image_dir (str): path to save output images
        out_iou_dir (str): a full output folder for iou matrix json files
        image_size (tuple[int] | None, optional): a target image size for the model. Defaults to None.
        confidence (float, optional): a confidence threshold. Defaults to 0.01.
        iou (float, optional): an iou threshold for NMS. Defaults to 0.45.
        max_det (int, optional): the max number of detections. Defaults to 600.

    """
    # load the model by model type
    if model_type not in MODEL_CLASSES:
        raise Exception(f"Not supported model type: {model_type}")
    model = MODEL_CLASSES[model_type](model_path)

    dataset = Dataset.load(label_path)
    pred_annot_id = 0
    for file_annot in dataset.files:
        fname = os.path.basename(file_annot.path)
        p = os.path.join(image_dir, file_annot.path)
        im = cv2.imread(p)
        if im is None:
            raise Exception(f"Could not read image {p}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        h_train, w_train = image_size if image_size is not None else (h, w)

        # get labels
        im_resized, annotations_resized = resize_annotated_image(im, file_annot.annotations, w_train, h_train, maintain_aspect_ratio=True)
        im_padded, annotations_padded, _ = pad_annotated_image(im_resized, annotations_resized, w_train, h_train)

        # run inference
        preds_batch, _ = model.predict(im_padded, confidence, iou=iou, max_det=max_det)
        preds = {k: v[0] for k, v in preds_batch.items()}

        # build prediction annotations (padded coords); ids are tentative and reassigned after unpadding
        logger.info(f"Found {len(preds['classes'])} predictions for {fname}")
        preds_padded = build_prediction_annotations(preds, model_type, pred_annot_id)

        # remove padding (keep resized); predictions in the padded region are dropped here
        h_unpad, w_unpad = im_resized.shape[:2]
        im_unpadded, preds_unpadded, is_deleted = pad_annotated_image(im_padded, preds_padded, w_unpad, h_unpad)
        _, annotations_unpadded, _ = pad_annotated_image(im_padded, annotations_padded, w_unpad, h_unpad)
        if is_deleted:
            update_annotation_ids(preds_unpadded, start_id=pred_annot_id)  # keep ids contiguous after deletion
        pred_annot_id += len(preds_unpadded)

        # compute ious on the unpadded annotations so the matrix matches the saved output
        labels = parse_annotations(annotations_unpadded, h_unpad, w_unpad, model_type)
        preds = parse_annotations(preds_unpadded, h_unpad, w_unpad, model_type)
        iou_result = compute_ious(labels, preds, model_type, model)
        n_gt = iou_result["n_gt"]
        n_pred = iou_result["n_pred"]

        # the iou matrix must align with the saved annotations; for keypoints, n_gt/n_pred count boxes (each box also has keypoints)
        if model_type == "KeypointDetection":
            n_gt_unpad = sum(a.type == AnnotationType.BOX for a in annotations_unpadded)
            n_pred_unpad = sum(a.type == AnnotationType.BOX for a in preds_unpadded)
        else:
            n_gt_unpad = len(annotations_unpadded)
            n_pred_unpad = len(preds_unpadded)
        if n_gt_unpad != n_gt:
            raise Exception(f"Invalid number of labels after unpadding: {n_gt_unpad} vs n_gt: {n_gt}")
        if n_pred_unpad != n_pred:
            raise Exception(f"Invalid number of predictions after unpadding: {n_pred_unpad} vs n_pred: {n_pred}")
        if model_type == "KeypointDetection":
            # every box must keep its full set of keypoints (none dropped independently during unpadding)
            nkpt = model.model.kpt_shape[0]
            n_gt_kpt = iou_result["n_gt_kpt"]
            n_pred_kpt = iou_result["n_pred_kpt"]
            if n_gt_kpt != n_gt * nkpt:
                raise Exception(f"Invalid number of label keypoints after unpadding: {n_gt_kpt} vs expected {n_gt * nkpt}")
            if n_pred_kpt != n_pred * nkpt:
                raise Exception(f"Invalid number of prediction keypoints after unpadding: {n_pred_kpt} vs expected {n_pred * nkpt}")

        # create iou json
        ious = iou_result["ious"]
        ious_out = [] if ious is None else ious.cpu().numpy().tolist()
        iou_json = dict(
            n_gt=n_gt,
            n_pred=n_pred,
            iou=ious_out,  # a shape of n_gt x n_pred
        )
        if model_type == "KeypointDetection":
            ious_kpt = iou_result["ious_kpt"]
            iou_json["kpt_iou"] = [] if ious_kpt is None else ious_kpt.cpu().numpy().tolist()
            iou_json["n_gt_kpt"] = iou_result["n_gt_kpt"]
            iou_json["n_pred_kpt"] = iou_result["n_pred_kpt"]

        # write ious to a json file
        os.makedirs(out_iou_dir, exist_ok=True)
        out_iou_path = os.path.join(out_iou_dir, file_annot.id + ".json")
        with open(out_iou_path, "w") as f:
            json.dump(iou_json, f)

        im_out = cv2.cvtColor(im_unpadded, cv2.COLOR_RGB2BGR)
        out_image_path = os.path.join(out_image_dir, file_annot.path)
        os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
        cv2.imwrite(out_image_path, im_out)

        # add predictions to dataset
        file_annot.width = im_unpadded.shape[1]
        file_annot.height = im_unpadded.shape[0]
        file_annot.annotations = annotations_unpadded
        file_annot.predictions = preds_unpadded

    # write out dataset
    dataset.save(out_pred_json)
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="a path to a model weights file")
    parser.add_argument(
        "--model_type",
        required=True,
        help="a type of the model, either ObjectDetection, OrientedObjectDetection, InstanceSegmentation, KeypointDetection",
    )
    parser.add_argument("--config_path", default=None, help="[optional] a path to a model config file")
    parser.add_argument("--img_dir", required=True, help="a input image directory")
    parser.add_argument("--label_path", required=True, help="a path to a label json file")
    parser.add_argument(
        "--out_pred_json",
        required=True,
        help="a full output json file path for predictions and labels",
    )
    parser.add_argument("--out_image_dir", required=True, help="a path to save output images")
    parser.add_argument(
        "--out_iou_dir",
        required=True,
        help="a full output folder for saving iou json files",
    )
    parser.add_argument(
        "--image_size",
        default=None,
        help="[optional] a target image size for the model, either w,h or single number",
    )
    parser.add_argument(
        "--confidence",
        default=0.01,
        type=float,
        help="[optional] confidence threshold, defaults to 0.01",
    )
    parser.add_argument(
        "--iou",
        default=0.45,
        type=float,
        help="[optional] iou NMS threshold, defaults to 0.45",
    )
    parser.add_argument(
        "--max_det",
        default=600,
        type=int,
        help="[optional] the max number of detections per image, default to 600",
    )
    ap = parser.parse_args()

    image_size = None
    if ap.image_size is not None:
        parsed_size = ap.image_size.split(",")
        if len(parsed_size) == 1:
            image_size = (int(parsed_size[0]), int(parsed_size[0]))
        elif len(parsed_size) == 2:
            image_size = (int(parsed_size[1]), int(parsed_size[0]))  # w,h -> (h,w)
        else:
            raise Exception(f"Invalid image size: {ap.image_size}; must be either w,h or a single number")

    write_json(
        ap.model_path,
        ap.model_type,
        ap.config_path,
        ap.img_dir,
        ap.label_path,
        ap.out_pred_json,
        ap.out_image_dir,
        ap.out_iou_dir,
        image_size,
        ap.confidence,
        ap.iou,
        ap.max_det,
    )
