"""RF-DETR backend for ``gofactory.write_validation_json`` (``--package RfDetr``).

The YOLO backend in ``write_validation_json`` letterboxes its input, so it must resize/pad the image and
annotations before inference and undo the padding afterwards. RF-DETR stretch-resizes its input internally
and returns predictions in input-image coordinates, so no letterbox/unpad bookkeeping is needed and the
input images and annotations pass through unchanged. Predictions are generated with the RfdetrModel
wrapper; box and mask IoU are computed with torch/torchvision.

Outputs (the same contract as the YOLO backend):
  * ``out_pred_json``: the input label dataset with per-file ``predictions`` added.
  * ``out_iou_dir``: one ``<file id>.json`` per file with ``{n_gt, n_pred, iou}`` where ``iou`` is the
    ``n_gt x n_pred`` matrix aligned with the saved annotation and prediction order.
  * ``out_image_dir``: the (unmodified) input images, copied so downstream consumers have a single source tree.
"""

import json
import logging
import os
import shutil

import cv2
import numpy as np
import torch

from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    Dataset,
    Mask,
)

logger = logging.getLogger(__name__)

MODEL_TYPE_OBJECT_DETECTION = "ObjectDetection"
MODEL_TYPE_INSTANCE_SEGMENTATION = "InstanceSegmentation"


def parse_label_boxes(annotations: list[Annotation]) -> np.ndarray:
    """Extract an (N, 4) xyxy array from label annotations for box IoU.

    Non-box annotations are reduced to their bounding box so the row count always matches the
    annotation count (the IoU matrix must align with the saved annotations).
    """
    boxes = []
    for annot in annotations:
        if annot.type == AnnotationType.BOX:
            box = annot.value
        else:
            box = annot.value.to_box()
        boxes.append(box.to_numpy()[:4])
    return np.array(boxes)


def parse_label_masks(annotations: list[Annotation], h: int, w: int) -> np.ndarray:
    """Extract an (N, H, W) binary mask array from label annotations for mask IoU.

    Box and polygon annotations are rasterized so instance-segmentation metrics treat them as filled
    regions, matching the YOLO backend behavior.
    """
    masks = []
    for annot in annotations:
        if annot.type == AnnotationType.MASK:
            masks.append(annot.value.to_numpy(h=h, w=w))
        else:
            masks.append(annot.value.to_mask(h=h, w=w).to_numpy(h=h, w=w))
    return np.array(masks)


def build_prediction_annotations(preds: dict, model_type: str, start_id: int) -> list[Annotation]:
    """Build prediction Annotation objects from a model's per-image outputs.

    Args:
        preds: per-image predictions with 'classes', 'boxes', 'scores' and, for segmentation, 'masks'.
        model_type: the model type, either ObjectDetection or InstanceSegmentation.
        start_id: first annotation id; ids are sequential across the dataset.

    Returns:
        Prediction annotations in input-image coordinates.
    """
    predictions = []
    for i in range(len(preds["classes"])):
        label_name = str(preds["classes"][i])
        score = float(preds["scores"][i])
        if model_type == MODEL_TYPE_INSTANCE_SEGMENTATION:
            value = Mask(np.asarray(preds["masks"][i]).astype(np.uint8))
            annotation_type = AnnotationType.MASK
        else:
            value = Box(*preds["boxes"][i][:4], angle=0)
            annotation_type = AnnotationType.BOX
        predictions.append(
            Annotation(
                id=str(start_id + i),
                label_id=label_name,
                type=annotation_type,
                value=value,
                confidence=score,
            )
        )
    return predictions


def box_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray, device) -> torch.Tensor:
    """Compute the pairwise xyxy box IoU matrix."""
    from torchvision.ops import box_iou

    gt = torch.from_numpy(gt_boxes.astype(np.float32)).to(device)
    pred = torch.from_numpy(np.asarray(pred_boxes)[:, :4].astype(np.float32)).to(device)
    return box_iou(gt, pred)


def mask_iou_matrix(gt_masks: np.ndarray, pred_masks, device) -> torch.Tensor:
    """Compute the pairwise binary mask IoU matrix."""
    gt = torch.from_numpy(gt_masks.astype(np.float32)).to(device).flatten(1)
    pred = torch.as_tensor(np.asarray(pred_masks), dtype=torch.float32, device=device).flatten(1)
    intersection = gt @ pred.T
    union = gt.sum(dim=1, keepdim=True) + pred.sum(dim=1) - intersection
    return intersection / union.clamp(min=1e-9)


def compute_ious(file_annotations: list[Annotation], preds: dict, model_type: str, h: int, w: int, device):
    """Compute the label-vs-prediction IoU matrix for one file.

    Returns (n_gt, n_pred, matrix) where matrix is None when either side is empty.
    """
    n_gt = len(file_annotations)
    n_pred = len(preds["classes"])
    if n_gt == 0 or n_pred == 0:
        return n_gt, n_pred, None
    if model_type == MODEL_TYPE_INSTANCE_SEGMENTATION:
        return n_gt, n_pred, mask_iou_matrix(parse_label_masks(file_annotations, h, w), preds["masks"], device)
    return n_gt, n_pred, box_iou_matrix(parse_label_boxes(file_annotations), preds["boxes"], device)


def write_json(
    model_path: str,
    model_type: str,
    variant: str,
    image_size: tuple[int, int],
    img_dir: str,
    label_path: str,
    out_pred_json: str,
    out_image_dir: str,
    out_iou_dir: str,
    confidence: float,
):
    """Run inference over a labeled dataset and write predictions, images, and IoU matrices.

    Args:
        model_path: a path to a model weights file
        model_type: the model type, either ObjectDetection or InstanceSegmentation
        variant: the RF-DETR model variant (e.g. small, seg-small)
        image_size: the model input size as (h, w)
        img_dir: an input image directory
        label_path: a path to a label json file
        out_pred_json: a full output json file path for predictions and labels
        out_image_dir: a path to save output images
        out_iou_dir: a full output folder for iou matrix json files
        confidence: a confidence threshold
    """
    from object_detectors.rf_detr_lmi.model import RfdetrModel

    if model_type not in (MODEL_TYPE_OBJECT_DETECTION, MODEL_TYPE_INSTANCE_SEGMENTATION):
        raise ValueError(f"Not supported model type: {model_type}")

    model = RfdetrModel(model_path, model_type=variant, image_size=list(image_size))
    model.warmup()

    os.makedirs(out_iou_dir, exist_ok=True)

    dataset = Dataset.load(label_path)
    pred_annot_id = 0
    for file_annot in dataset.files:
        image_path = os.path.join(img_dir, file_annot.path)
        im = cv2.imread(image_path)
        if im is None:
            raise Exception(f"Could not read image {image_path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]

        preds_batch, _ = model.predict(im, configs=confidence)
        preds = {k: v[0] for k, v in preds_batch.items()}
        logger.info(f"Found {len(preds['classes'])} predictions for {os.path.basename(file_annot.path)}")

        predictions = build_prediction_annotations(preds, model_type, pred_annot_id)
        pred_annot_id += len(predictions)

        annotations = file_annot.annotations or []
        n_gt, n_pred, ious = compute_ious(annotations, preds, model_type, h, w, model.device)
        iou_json = dict(
            n_gt=n_gt,
            n_pred=n_pred,
            iou=[] if ious is None else ious.cpu().numpy().tolist(),  # a shape of n_gt x n_pred
        )
        out_iou_path = os.path.join(out_iou_dir, file_annot.id + ".json")
        with open(out_iou_path, "w") as f:
            json.dump(iou_json, f)

        out_image_path = os.path.join(out_image_dir, file_annot.path)
        os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
        shutil.copy(image_path, out_image_path)

        file_annot.width = w
        file_annot.height = h
        file_annot.predictions = predictions

    dataset.save(out_pred_json)
