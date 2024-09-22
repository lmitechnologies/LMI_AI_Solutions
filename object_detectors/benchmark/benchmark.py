import numpy as np
from label_utils.csv_utils import load_csv
from label_utils.shapes import Mask, Rect, Keypoint
import torch
import torchvision.ops.boxes as bops

def binaryMaskIOU(mask1, mask2):
    """source : https://stackoverflow.com/questions/66595055/fastest-way-of-computing-binary-mask-iou-with-numpy"""
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou

def get_dataset(csv_file_path, image_dir):
    shapes, _ = load_csv(
            fname=csv_file_path,
            path_img=image_dir
        )
    return shapes

def bbox_analysis(pred_shapes, gt_shapes, confidence_thresholds=[0.4,0.45,0.5, 0.75, 0.8,0.85,0.9, 0.95]):
    analysis_dictionary = {}
    for threshold_confidence in confidence_thresholds:
        filtered_preds = [p for p in pred_shapes if p.confidence >= threshold_confidence]
        analysis_dictionary[threshold_confidence] = []
        for shape in filtered_preds:
            if isinstance(shape, Rect):
                x0,y0 = shape.up_left
                x2,y2 = shape.bottom_right
                confidence = shape.confidence
                category = shape.category
                best_iou = 0
                for gt_shape in gt_shapes:
                    if isinstance(gt_shape, Rect):
                        x0_gt, y0_gt = gt_shape.up_left
                        x2_gt, y2_gt = gt_shape.bottom_right
                        category_gt = gt_shape.category
                        iou = bops.box_iou(torch.tensor([[x0,y0,x2,y2]]), torch.tensor([[x0_gt,y0_gt,x2_gt,y2_gt]]))
                        best_iou = max(best_iou, iou)
                analysis_dictionary[threshold_confidence].append({
                    'category': category,
                    'confidence': confidence,
                    'best_iou': best_iou,
                    'conf_threshold': threshold_confidence
                })
                
                        



