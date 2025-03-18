import os
import cv2
import numpy as np
import torch
from ultralytics.utils.metrics import box_iou, mask_iou
import logging
import json

from ultralytics_lmi.yolo.model import Yolo, YoloPose, YoloObb
from dataset_utils.representations import Dataset, Annotation, AnnotationType, Box, Mask


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def parse_annotations(annotations:list[Annotation], h:int, w:int):
    """parse label annotations from a list. Only support Box and Mask annotation objects.

    Args:
        annotations (list[Annotation]): a list of Annotation objects (Box and Mask)
        h (int): image height
        w (int): image width

    Returns:
        dict: a dictionary contains 'ids','boxes','masks'
    """
    boxes = []
    masks = []
    label_ids = []
    for annot in annotations:
        label_ids.append(int(annot.label_id))
        if annot.type == AnnotationType.BOX:
            boxes.append(annot.value.to_numpy())
        elif annot.type == AnnotationType.MASK:
            mask = annot.value.to_numpy(h=h,w=w)
            masks.append(mask)
        elif annot.type == AnnotationType.POLYGON:
            obj = annot.value.to_mask(h=h, w=w)
            mask = obj.to_numpy(h=h,w=w)
            masks.append(mask)
        else:
            raise Exception(f'Not supported type: {type(annot.type)}')
    return {
        'boxes': np.array(boxes),
        'masks': np.array(masks),
        'ids': np.array(label_ids)
    }


def write_json(model_path, config_path, image_dir, label_path, out_pred_json, out_iou_dir, confidence=0.01, iou=0.45, max_det=600):
    """write predictions and labels to a json file

    Args:
        model_path (str): a path to a model weights file
        config_path (str): a path to a model configuration file
        image_dir (str): a input image directory, where each image should have the same dimension as training images
        label_path (str): a path to a label json file
        output_path (str): a full output json file path
        out_iou_dir (str): a full output folder for iou matrix json files
        confidence (float, optional): a confidence threshold. Defaults to 0.01.
        iou (float, optional): an iou threshold for NMS. Defaults to 0.45.
        max_det (int, optional): the max number of detections. Defaults to 600.
        
    """
    model = Yolo(model_path)
    dataset = Dataset.load(label_path)
    cls_to_id = {l.name:int(l.id) for l in dataset.labels}
    
    pred_annot_id = 0 # sum([len(f.annotations) for f in dataset.files])
    for file_annot in dataset.files:
        fname = os.path.basename(file_annot.path)
        p = os.path.join(image_dir, file_annot.path)
        im = cv2.imread(p)
        if im is None:
            raise Exception(f'Could not read image {p}')
        
        # get labels and preds
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h,w = im.shape[:2]
        labels = parse_annotations(file_annot.annotations, h, w)
        preds,_ = model.predict(im, confidence, iou=iou, max_det=max_det, return_segments=False)
        
        # get ious
        ious = None
        if 'masks' in preds:
            n_gt = len(labels['masks'])
            n_pred = len(preds['masks'])
            if n_gt and n_pred:
                gt_masks = torch.from_numpy(labels['masks']).float().to(model.device)
                pred_masks = torch.from_numpy(preds['masks']).to(model.device)
                ious = mask_iou(gt_masks.view(gt_masks.shape[0], -1),pred_masks.view(pred_masks.shape[0],-1))
        else:
            n_gt = len(labels['boxes'])
            n_pred = len(preds['boxes'])
            if n_gt and n_pred:
                gt_boxes = torch.from_numpy(labels['boxes'][:,:-1]).to(model.device)
                pred_boxes = torch.from_numpy(preds['boxes']).to(model.device)
                ious = box_iou(gt_boxes, pred_boxes)
                
        # write ious to a json file
        ious_out = [] if ious is None else ious.cpu().numpy().tolist()
        iou_json = dict(
            impath=p,
            n_gt=n_gt,
            n_pred=n_pred,
            iou=ious_out
        )
        os.makedirs(out_iou_dir, exist_ok=True)
        out_iou_path = os.path.join(out_iou_dir, file_annot.id + '.json')
        with open(out_iou_path, 'w') as f:
            json.dump(iou_json, f)
        
        # add predictions to dataset
        logger.info(f'Found {len(preds["classes"])} predictions for {fname}')
        for i in range(len(preds['classes'])):
            box = preds['boxes'][i]
            mask = preds['masks'][i] if 'masks' in preds else None
            label = preds['classes'][i]
            label_id = cls_to_id[label]
            score = preds['scores'][i].item()
            
            if mask is not None:
                dt = dict(
                    id=str(pred_annot_id), label_id=str(label_id), type=AnnotationType.MASK, value=Mask(mask), 
                    confidence=score, 
                )
                file_annot.predictions.append(Annotation(**dt))
                pred_annot_id += 1
            else:
                dt = dict(
                    id=str(pred_annot_id), label_id=str(label_id), type=AnnotationType.BOX, value=Box(*box,angle=0), 
                    confidence=score
                )
                file_annot.predictions.append(Annotation(**dt))
                pred_annot_id += 1
                
    # write out dataset
    dataset.save(out_pred_json)
    return


if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',required=True,help='a path to a model weights file')
    parser.add_argument('--config_path',default=None,help='[optional] a path to a model config file')
    parser.add_argument('--img_dir',required=True,help='a input image directory')
    parser.add_argument('--label_path',required=True,help='a path to a label json file')
    parser.add_argument('--out_pred_json',required=True,help='a full output json file path for predictions and labels')
    parser.add_argument('--out_iou_dir',required=True,help='a full output folder for saving iou json files')
    parser.add_argument('--confidence',default=0.01,type=float,help='[optional] confidence threshold, defaults to 0.01')
    parser.add_argument('--iou',default=0.45,type=float,help='[optional] iou NMS threshold, defaults to 0.45')
    parser.add_argument('--max_det',default=600,type=int,help='[optional] the max number of detections per image, default to 600')
    ap = parser.parse_args()
    
    write_json(ap.model_path, ap.config_path, ap.img_dir, ap.label_path, ap.out_pred_json, ap.out_iou_dir, ap.confidence, ap.iou, ap.max_det)
    