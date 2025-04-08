import os
import cv2
import numpy as np
import torch
from ultralytics.utils.metrics import box_iou, mask_iou
import logging
import json

from ultralytics_lmi.yolo.model import Yolo, YoloPose, YoloObb
from dataset_utils.representations import Dataset, Annotation, AnnotationType, Box, Mask
from dataset_utils.ops.dataset_resize import resize_annotated_image, resize_annotations
from dataset_utils.ops.dataset_pad import pad_annotated_image


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
        dict: a dictionary contains 'classes','boxes','masks'
    """
    boxes = []
    masks = []
    label_names = []
    for annot in annotations:
        label_names.append(annot.label_id)
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
        'classes': np.array(label_names)
    }


def write_json(model_path, config_path, image_dir, label_path, out_pred_json, out_image_dir, out_iou_dir, image_size: tuple[int,int] | None, confidence=0.01, iou=0.45, max_det=600):
    """write predictions and labels to a json file

    Args:
        model_path (str): a path to a model weights file
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
    model = Yolo(model_path)
    dataset = Dataset.load(label_path)
    
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
        h_train,w_train = image_size if image_size is not None else (h,w)

        im_resized, annotations_resized = resize_annotated_image(im, file_annot.annotations, w_train, h_train, maintain_aspect_ratio=True)
        im_padded, annotations_padded, _ = pad_annotated_image(im_resized, annotations_resized, w_train, h_train)
        labels = parse_annotations(annotations_padded, h_train, w_train)

        preds,_ = model.predict(im_padded, confidence, iou=iou, max_det=max_det, return_segments=False)
        
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
            n_gt=n_gt,
            n_pred=n_pred,
            iou=ious_out # a shape of n_gt x n_pred
        )
        os.makedirs(out_iou_dir, exist_ok=True)
        out_iou_path = os.path.join(out_iou_dir, file_annot.id + '.json')
        with open(out_iou_path, 'w') as f:
            json.dump(iou_json, f)
        
        # add predictions to dataset
        logger.info(f'Found {len(preds["classes"])} predictions for {fname}')
        preds_padded = []
        for i in range(len(preds['classes'])):
            box = preds['boxes'][i]
            mask = preds['masks'][i] if 'masks' in preds else None
            label_name = preds['classes'][i]
            score = preds['scores'][i].item()
            
            if mask is not None:
                dt = dict(
                    id=str(pred_annot_id), label_id=label_name, type=AnnotationType.MASK, value=Mask(mask), 
                    confidence=score, 
                )
                preds_padded.append(Annotation(**dt))
                pred_annot_id += 1
            else:
                dt = dict(
                    id=str(pred_annot_id), label_id=label_name, type=AnnotationType.BOX, value=Box(*box,angle=0), 
                    confidence=score
                )
                preds_padded.append(Annotation(**dt))
                pred_annot_id += 1

        # remove padding and save image                
        im_unpadded, preds_unpadded, _ = pad_annotated_image(im_padded, preds_padded, im_resized.shape[1], im_resized.shape[0])
        _, annotations_unpadded, _ = pad_annotated_image(im_padded, annotations_padded, im_resized.shape[1], im_resized.shape[0])
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


if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',required=True,help='a path to a model weights file')
    parser.add_argument('--config_path',default=None,help='[optional] a path to a model config file')
    parser.add_argument('--img_dir',required=True,help='a input image directory')
    parser.add_argument('--label_path',required=True,help='a path to a label json file')
    parser.add_argument('--out_pred_json',required=True,help='a full output json file path for predictions and labels')
    parser.add_argument('--out_image_dir',required=True,help='a path to save output images')
    parser.add_argument('--out_iou_dir',required=True,help='a full output folder for saving iou json files')
    parser.add_argument('--image_size',default=None,help='[optional] a target image size for the model, either w,h or single number')
    parser.add_argument('--confidence',default=0.01,type=float,help='[optional] confidence threshold, defaults to 0.01')
    parser.add_argument('--iou',default=0.45,type=float,help='[optional] iou NMS threshold, defaults to 0.45')
    parser.add_argument('--max_det',default=600,type=int,help='[optional] the max number of detections per image, default to 600')
    ap = parser.parse_args()
    
    image_size = None
    if ap.image_size is not None:
        parsed_size = ap.image_size.split(',')
        if len(parsed_size) == 1:
            image_size = (int(parsed_size[0]), int(parsed_size[0]))
        elif len(parsed_size) == 2:
            image_size = (int(parsed_size[1]), int(parsed_size[0]))     #w,h -> (h,w)
        else:
            raise Exception(f'Invalid image size: {ap.image_size}; must be either w,h or a single number')

    write_json(ap.model_path, ap.config_path, ap.img_dir, ap.label_path, ap.out_pred_json, ap.out_image_dir, ap.out_iou_dir, image_size, ap.confidence, ap.iou, ap.max_det)
    