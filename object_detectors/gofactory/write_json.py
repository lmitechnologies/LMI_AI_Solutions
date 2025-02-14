import os
import cv2
import numpy as np
import scipy.optimize
import torch
from ultralytics.utils.metrics import box_iou, mask_iou
import scipy
import logging

from ultralytics_lmi.yolo.model import Yolo, YoloPose, YoloObb
from dataset_utils.representations import Dataset, Annotation, AnnotationType, MaskType, Box, Mask, Link


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def match_predictions(preds, labels, cls_to_id, ious):
    """match predictions to labels

    Args:
        preds (dict): a dictionary of model outputs
        labels (dict): a dictionary of labels
        cls_to_id (dict): a map <class_name : class_idx>
        ious (numpy | None): a matrix of shape (N,M), where N is the number of labels, M is the number of predictions

    Returns:
        dict: a map <prediction_idx : label_idx>
    """
    if ious is None:
        return {}
    
    pred_ids = np.array([cls_to_id[l] for l in preds['classes']])
    label_ids = np.array([cls_to_id[l] for l in labels['classes']])
    class_match = label_ids[:,None] == pred_ids
    ious = ious.cpu().numpy()
    ious = ious*class_match # zero out ious for non-matching classes, (n_gt, n_pred)
    
    # Hungarian matching
    gt_idx, pred_idx = scipy.optimize.linear_sum_assignment(ious, maximize=True)
    valid = ious[gt_idx, pred_idx] > 0
    gt_idx = gt_idx[valid]
    pred_idx = pred_idx[valid]
    pred_to_gt = {pred_idx[i]:gt_idx[i] for i in range(len(gt_idx))}
    return pred_to_gt
    

def parse_annotations(annotations:list[Annotation], h:int, w:int):
    """parse label annotations from a list. Only support Box and Mask annotation objects.

    Args:
        annotations (list[Annotation]): a list of Annotation objects (Box and Mask)
        h (int): image height
        w (int): image width

    Returns:
        dict: a dictionary contain 'classes','boxes','masks'
    """
    boxes = []
    masks = []
    labels = []
    for annot in annotations:
        labels.append(annot.label_id)
        if annot.type == AnnotationType.BOX:
            boxes.append(annot.value.to_numpy())
        elif annot.type == AnnotationType.MASK:
            if annot.value.type == MaskType.BITMASK:
                mask = annot.value.to_numpy()
            elif annot.value.type == MaskType.POLYGON:
                mask = np.zeros((h, w), dtype=np.uint8)
                xy = annot.value.to_numpy().astype(np.int32)
                cv2.fillPoly(mask, [xy], 1)
            masks.append(mask)
        else:
            raise Exception(f'Not supported type: {type(annot.type)}')
    return {
        'boxes': np.array(boxes),
        'masks': np.array(masks),
        'classes': labels
    }


def write_json(model_path, config_path, image_dir, label_path, output_path, confidence=0.01, max_det=600):
    """write predictions and labels to a json file

    Args:
        model_path (str): a path to a model weights file
        config_path (str): a path to a model configuration file
        image_dir (str): a input image directory, where each image should have the same dimension as training images
        label_path (str): a path to a label json file
        output_path (str): a full output json file path
        confidence (float, optional): a confidence threshold. Defaults to 0.01.
        max_det (int, optional): the max number of detections. Defaults to 600.
        
    """
    model = Yolo(model_path)
    dataset = Dataset.load(label_path)
    cls_to_id = {l.id:l.index for l in dataset.labels}
    
    pred_annot_id = 0 # sum([len(f.annotations) for f in dataset.files])
    for file_annot in dataset.files:
        fname = os.path.basename(file_annot.path)
        p = os.path.join(image_dir, fname)
        im = cv2.imread(p)
        if im is None:
            raise Exception(f'Could not read image {p}')
        
        # update to relative path
        file_annot.path = os.path.relpath(p,image_dir)
        
        # get labels and preds
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h,w = im.shape[:2]
        labels = parse_annotations(file_annot.annotations, h, w)
        preds,_ = model.predict(im, confidence, max_det=max_det, return_segments=False)
        
        # get ious
        ious = None
        if 'masks' in preds:
            if len(labels['masks']) and len(preds['masks']):
                gt_masks = torch.from_numpy(labels['masks']).float().to(model.device)
                pred_masks = torch.from_numpy(preds['masks']).to(model.device)
                ious = mask_iou(gt_masks.view(gt_masks.shape[0], -1),pred_masks.view(pred_masks.shape[0],-1))
        else:
            if len(labels['boxes']) and len(preds['boxes']):
                gt_boxes = torch.from_numpy(labels['boxes'][:,:-1]).to(model.device)
                pred_boxes = torch.from_numpy(preds['boxes']).to(model.device)
                ious = box_iou(gt_boxes, pred_boxes)
            
        pred_to_gt = match_predictions(preds, labels, cls_to_id, ious)
        
        # add predictions to dataset
        logger.info(f'Found {len(preds["classes"])} predictions for {fname}')
        for i in range(len(preds['classes'])):
            box = preds['boxes'][i]
            mask = preds['masks'][i] if 'masks' in preds else None
            label = preds['classes'][i]
            score = preds['scores'][i].item()
            label_annot_id = None
            iou = 0
            if i in pred_to_gt:
                gt_id = pred_to_gt[i]
                iou = ious[gt_id, i].item()
                label_annot_id = file_annot.annotations[gt_id].id
            
            if mask is not None:
                dt = dict(
                    id=str(pred_annot_id), label_id=label, type=AnnotationType.MASK, value=Mask(MaskType.BITMASK,mask,h,w), 
                    link=Link(annotation_id=label_annot_id), confidence=score, iou=iou
                )
                file_annot.predictions.append(Annotation(**dt))
                pred_annot_id += 1
            else:
                dt = dict(
                    id=str(pred_annot_id), label_id=label, type=AnnotationType.BOX, value=Box(*box,angle=0), 
                    link=Link(annotation_id=label_annot_id), confidence=score, iou=iou
                )
                file_annot.predictions.append(Annotation(**dt))
                pred_annot_id += 1
                
    # write out dataset
    dataset.save(output_path)
    return


if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',required=True,help='a path to a model weights file')
    parser.add_argument('--config_path',default=None,help='[optional] a path to a model config file')
    parser.add_argument('--img_dir',required=True,help='a input image directory')
    parser.add_argument('--label_path',required=True,help='a path to a label json file')
    parser.add_argument('--out_path',required=True,help='a full path to a output json file')
    parser.add_argument('--confidence',default=0.01,type=float,help='[optional] confidence threshold, defaults to 0.01')
    parser.add_argument('--max_det',default=600,type=int,help='[optional] the max number of detections per image, default to 600')
    ap = parser.parse_args()
    
    write_json(ap.model_path, ap.config_path, ap.img_dir, ap.label_path, ap.out_path, ap.confidence, ap.max_det)
    