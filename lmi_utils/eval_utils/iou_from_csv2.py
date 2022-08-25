from gc import collect
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid
import sys
import csv
import cv2

#LMI packages
from label_utils import csv_utils,rect,mask,shape


def bbox_iou(bbox1, bbox2):
    """
    calculate the iou between bbox1 and bbox2
    arguments:
        bbox1: an array of size [N,4]
        bbox2: an array of size [M,4]
    return:
        iou of size [N,M]
    """
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)
    
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))
    
    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w
    
    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union


def polygon_iou(polygon_1, polygon_2):
    """
    caluclate the IOU between two plygons
    Arugments:
        polygon_1: [[row1, col1], [row2, col2], ...]
        polygon_2: same as polygon_1
    return:
        IOU
    """
    try:
        poly_1 = Polygon(polygon_1)
        poly_2 = Polygon(polygon_2)
    except Exception as e:
        #usually less than 3 points for creating the polygons
        #print(e)
        return 0

    if not poly_1.is_valid:
        poly_1 = make_valid(poly_1)
    if not poly_2.is_valid:
        poly_2 = make_valid(poly_2)

    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def polygon_ious(polygons_1, polygons_2):
    N,M = len(polygons_1), len(polygons_2)
    ious = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            ious[i][j] = polygon_iou(polygons_1[i],polygons_2[j])
    return ious


def plot_shapes(image, shape_label, class_label, shape_pred, class_pred, is_mask=False):
    # BGR
    BLUE = (255,0,0)
    RED = (0,0,255)
    
    def plot_bboxs(image, bboxs, labels, color:tuple, pos='uleft'):
        for i in range(len(bboxs)):
            x1,y1,x2,y2 = bboxs[i]
            label = labels[i]
            uleft,bright = (x1,y1),(x2,y2)
            cv2.rectangle(image,uleft,bright,color,1)
            if pos=='uleft':
                cv2.putText(image, label, uleft, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            else:
                cv2.putText(image, label, bright, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
    def plot_masks(image, masks, labels, color:tuple, pos='uleft'):
        for i in range(len(masks)):
            pts = masks[i]
            label = labels[i]
            uleft = pts.min(axis=0)
            bright = pts.max(axis=0)
            cv2.polylines(image,pts,True,color,1)
            if pos=='uleft':
                cv2.putText(image, label, uleft, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            else:
                cv2.putText(image, label, bright, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    if not is_mask:
        plot_bboxs(image,shape_label,class_label,color=BLUE)
        plot_bboxs(image,shape_pred,class_pred,color=RED,pos='bright')
    else:
        plot_masks(image,shape_label,class_label,color=BLUE)
        plot_masks(image,shape_pred,class_pred,color=RED,pos='bright')


def get_ious(path_imgs:str,path_out:str,label_dt:dict, pred_dt:dict, class_map:dict, skip_classes=[]):
    """
    calculate the precision and recall based on the threshold of iou and confidence
    arguments:
        label_dt: the map <fname, list of Shapes> from label annotations
        pred_dt: the map <fname, list of Shapes> from prediction
        class_map: the map <class, class id>
    return:
        all_ious: the map <fname, D>, where D is <class, list of ious of that class>. The list might contain nan which is FN.
    """

    def mask_to_np(shapes):
        masks = []
        for shape in shapes:
            cur = np.empty((0,2))
            if not isinstance(shape, mask.Mask):
                continue
            for x,y in zip(shape.X,shape.Y):
                cur = np.concatenate((cur,[[x,y]]),axis=0)
            masks.append(cur)
        masks = np.array(masks,dtype=object)
        return masks
    
    all_ious = {}
    all_not_nan_ious = collections.defaultdict(list)
    fnames = set([f for f in label_dt]+[f for f in pred_dt])
    for fname in fnames:
        is_mask = 0
        I = cv2.imread(os.path.join(path_imgs,fname))
        if not label_dt[fname]:
            print(f'[warning] cannot find corresponding labels for the file: {fname}')
            bbox_label = np.empty((0,4))
            bbox_pred = np.empty((0,4))
            class_label = np.empty((0,))
            class_pred = np.empty((0,))
        elif isinstance(label_dt[fname][0], rect.Rect):
            # bbox: [x1,y1,x2,y2]
            bbox_label = np.array([shape.up_left+shape.bottom_right for shape in label_dt[fname] if isinstance(shape, rect.Rect) ])
            bbox_pred = np.array([shape.up_left+shape.bottom_right for shape in pred_dt[fname] if isinstance(shape, rect.Rect) ])
            class_label = np.array([shape.category for shape in label_dt[fname] if isinstance(shape, rect.Rect) ])
            class_pred = np.array([shape.category for shape in pred_dt[fname] if isinstance(shape, rect.Rect) ])
            conf = np.array([shape.confidence for shape in pred_dt[fname] if isinstance(shape, rect.Rect) ])
        elif isinstance(label_dt[fname][0], mask.Mask):
            is_mask = 1
            # mask: [[x1,y1],[x2,y2] ...]
            bbox_label = mask_to_np(label_dt[fname])
            bbox_pred = mask_to_np(pred_dt[fname])
            class_label = np.array([shape.category for shape in label_dt[fname] if isinstance(shape, mask.Mask) ])
            class_pred = np.array([shape.category for shape in pred_dt[fname] if isinstance(shape, mask.Mask) ])
            conf = np.array([shape.confidence for shape in pred_dt[fname] if isinstance(shape, mask.Mask) ])

        # plot shapes
        if is_mask:
            plot_shapes(I, bbox_label, class_label, bbox_pred, class_pred, is_mask=True)
        else:
            plot_shapes(I, bbox_label, class_label, bbox_pred, class_pred, is_mask=False)
        
        # write image
        outname = os.path.splitext(fname)[0]+'_iou.png'
        if I is not None:
            cv2.imwrite(os.path.join(path_out,outname),I)

        # get iou
        class_to_iou = {}
        all_classes = np.concatenate((class_label, class_pred), axis=0)
        for c in set(all_classes):
            m_label = np.empty((0,))
            if class_label.shape[0]:
                m_label = class_label==c    
                
            m_pred = np.empty((0,))
            if class_pred.shape[0]:
                m_pred = class_pred == c

            if not m_label.sum():
                final_ious_nan = np.zeros((m_pred.sum(),))
            elif not m_pred.sum():
                final_ious_nan = np.zeros((m_label.sum(),))*np.nan
            else:
                if is_mask:
                    ious = polygon_ious(bbox_pred[m_pred], bbox_label[m_label])
                else:
                    ious = bbox_iou(bbox_pred[m_pred,:], bbox_label[m_label,:])
                M,N = ious.shape[:2]
                middle_ious = np.max(ious, axis=1)
                all_not_nan_ious[c].append(middle_ious.mean())
                final_ious_nan = middle_ious.copy()
                # found false negatives, append NaN
                for _ in range(max(0, N-M)):
                    final_ious_nan = np.append(final_ious_nan, np.nan)
            class_to_iou[c] = final_ious_nan
            
        all_ious[fname] = class_to_iou
    return all_ious,all_not_nan_ious


def write_to_csv(all_ious:dict, mean_ious:dict, filename:str):
    """
    write a dictionary of list of shapes into a csv file
    Arguments:
        shape(dict): a dictionary maps the filename to a list of Mask or Rect objects, i.e., <filename, list of Mask or Rect>
        filename(str): the output csv filename
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["nan: false negative","0: false positive"])
        for im_name in all_ious:
            for category in all_ious[im_name]:
                writer.writerow([im_name, category] + all_ious[im_name][category].tolist())
                    
        for c in mean_ious:
            writer.writerow([f'mean iou of {c}: {mean_ious[c]}'])




if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser(description='Get the IOU and save to file. Right now, it does NOT support mixed polygons and bboxs in csv files.')
    parse.add_argument('--path_imgs', required=True, help='the path to the images')
    parse.add_argument('--model_csv', required=True, help='the path to the model prediction csv')
    parse.add_argument('--label_csv', required=True, help='the path to the ground truth csv')
    parse.add_argument('--path_out', required=True, help='the output path')
    parse.add_argument('--skip_classes', default='', help='skip calculating the P/R curves for these comma separated classes')
    args = vars(parse.parse_args())

    path_imgs = args['path_imgs']
    model_csv = args['model_csv']
    label_csv = args['label_csv']
    path_out = args['path_out']
    skip_classes = args['skip_classes']
    if skip_classes=='':
        skip_classes = []
    else:
        skip_classes = skip_classes.split(',')

    if not os.path.isfile(model_csv):
        raise Exception(f'Not found the "preds.csv" in {os.path.dirname(model_csv)}')

    if not os.path.isfile(label_csv):
        raise Exception(f'Not found the "labels.csv" in {os.path.dirname(label_csv)}')
    
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    label_dt,class_map = csv_utils.load_csv(label_csv)
    print(f'found class map: {class_map}')
    pred_dt,_ = csv_utils.load_csv(model_csv, class_map=class_map)

    all_ious, all_not_nan_ious = get_ious(path_imgs,path_out,label_dt,pred_dt,class_map,skip_classes)
    
    # get mean IOU
    mean_ious = {}
    total = 0
    cnt = 0
    for c in all_not_nan_ious:
        l = all_not_nan_ious[c]
        mean_ious[c] = sum(l)/len(l)
        total += sum(l)
        cnt += len(l)    
    mean_ious['all'] = total/cnt
        
    write_to_csv(all_ious, mean_ious, os.path.join(path_out,'ious.csv'))