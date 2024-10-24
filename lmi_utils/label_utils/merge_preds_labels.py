import os
import logging
import cv2

#LMI packages
from label_utils import csv_utils
from torchvision.ops import box_iou
import torch

# set up logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def merge_preds_labels(preds, labels, imgs_path,path_out, iou_threshold=0.3, output_csv='merged.csv'):
    """
    merge predictions and labels into a csv file
    Arguments:
        preds(str): the path of prediction csv file
        labels(str): the path of label csv file
        output_dir(str): the output directory
        output_csv(str): the output csv file name
    """
    
    # load preds and labels
    preds_shapes, _ = csv_utils.load_csv(preds, path_img=imgs_path)
    labels_shapes, _ = csv_utils.load_csv(labels, path_img=imgs_path)
    # merge unique preds and labels determine by iou_threshold if the iou is less than iou_threshold, 
    # it will be considered as a new shape
    
    # update the label category with <category>-pred
    for im_name, shapes in preds_shapes.items():
        for shape in shapes:
            shape.category = f'{shape.category}-pred'

    for im_name, shapes in labels_shapes.items():
        if im_name not in preds_shapes:
            preds_shapes[im_name] = shapes
    
    # determine the predictions that are not in the labels

    for im_name, shapes in preds_shapes.items():
        if im_name not in labels_shapes:
            labels_shapes[im_name] = shapes
        else:
            for shape in shapes:
                x1,y1 =shape.up_left
                x2,y2 = shape.bottom_right
                pred_category = shape.category
                best_iou = 0
                for label_shape in labels_shapes[im_name]:
                    label_category = label_shape.category
                    if pred_category.replace('-pred', '') != label_category:
                        continue
                    x1_l,y1_l = label_shape.up_left
                    x2_l,y2_l = label_shape.bottom_right
                    # compute the iou between the prediction and the label
                    iou = box_iou(torch.tensor([[x1, y1, x2, y2]]), torch.tensor([[x1_l, y1_l, x2_l, y2_l]]))
                    best_iou = max(iou, best_iou)
                if best_iou <= iou_threshold:
                    logger.info(f'{im_name}: up left {shape.up_left} bottom right {shape.bottom_right} is not in labels')
                    labels_shapes[im_name].append(shape)
    
    # save to csv
    csv_utils.write_to_csv(labels_shapes, os.path.join(path_out,output_csv), overwrite=True)
    
    logger.info(f'merged {preds} and {labels} to {path_out}/{output_csv}')
    
    return preds_shapes

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument('--path_csv',default='labels.csv', help='the path of a csv file that corresponds to labels', required=True)
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to images')
    ap.add_argument('--path_preds', '-p', required=True, help='the path of a csv file that corresponds to predictions')
    ap.add_argument('--path_out', '-o', required=True, help='the path to save the merged csv file')
    ap.add_argument('--iou', type=float, default=0.3, help='the iou threshold to determine the same shape, default=0.3')
    ap.add_argument('--output_csv', default='merged.csv', help='the output csv file name, default="merged.csv"')
    args = ap.parse_args()

    merge_preds_labels(args.path_preds, args.path_csv, args.path_imgs ,args.path_out, args.iou, args.output_csv)