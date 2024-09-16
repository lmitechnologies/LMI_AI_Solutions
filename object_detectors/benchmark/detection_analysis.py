import numpy as np
import torch
from label_utils.csv_utils import load_csv
import torchvision.ops.boxes as box_ops

def read_csv(path_csv:str, path_imgs:str):
    shapes, _ = load_csv(
        fname=path_csv,
        path_img=path_imgs
    )
    return shapes

def bbox_analysis(predicted_shapes, ground_truth_shapes, score_threshold=0.5):
    # using the ground truth shapes find the missed detections
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    analysis_data = {}
    for filename, shapes in ground_truth_shapes.items():
        matched_gt_shapes = set()
        matched_pred_shapes = set()
        analysis_data[filename] = []
        for pred_shape in predicted_shapes:
            x1, y1 = pred_shape.up_left
            x2, y2 = pred_shape.bottom_right
            best_iou = 0
            best_gt_shape = None   
            for gt_shape in shapes:
                x1_gt, y1_gt = gt_shape.up_left
                x2_gt, y2_gt = gt_shape.bottom_right
                iou = box_ops.box_iou(
                    torch.tensor([[x1, y1, x2, y2]]),
                    torch.tensor([[x1_gt, y1_gt, x2_gt, y2_gt]])
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_shape = gt_shape
            if best_iou > 0.5:
                tp += 1
                matched_pred_shapes.add(pred_shape)
                matched_gt_shapes.add(best_gt_shape)
            else:
                fp += 1
            
        for i , shape in enumerate(shapes):
            if shape not in matched_gt_shapes:
                fn += 1
        for pred in predicted_shapes:
            if pred not in matched_pred_shapes:
                tn += 1
    return tp, fp, fn, tn


# def calculate_statistics(analysis_data):
#     """
#     Arguments:
#         analysis_data (dict): {filename: [shape1, shape2, ...]}
#     Returns:
#         dict: {category: {'precision': 0, 'recall': 0, 'f1': 0}}
#     """
#     stats = {}
#     for filename, shapes in analysis_data.items():
#         for shape in shapes:
#             category = shape['category']
#             if category not in stats:
#                 stats[category] = {
#                     'true_positives': 0,
#                     'false_positives': 0,
#                     'false_negatives': 0,
#                 }
#             if shape['iou'] > 0.5:
#                 stats[category]['true_positives'] += 1
#             else:
#                 stats[category]['false_positives'] += 1
#     for category, data in stats.items():
#         data['false_negatives'] = 0
#     return stats


                
                    
                    