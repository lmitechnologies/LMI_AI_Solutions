import csv_utils

import os
import numpy as np
import collections
import matplotlib.pyplot as plt


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



def precision_recall(label_dt:dict, pred_dt:dict, class_map:dict, threshold_iou=0.5, threshold_conf=0.1, image_level=False):
    """
    calculate the precision and recall based on the threshold of iou and confidence
    arguments:
        label_dt: the map <fname, list of Shapes> from label annotations
        pred_dt: the map <fname, list of Shapes> from prediction
        class_map: the map <class, class id>
        threshold_iou: iou threshold, default=0.5
        threshold_conf: confidence threshold, default=0.1
    return:
        P: the map <class: class's precision>
        R: the map <class: class's recall>
    """
    
    # get TP (num of tp), FP(num of fp) and GT(num of ground truth)
    TP,FP,GT,FN = collections.defaultdict(int),collections.defaultdict(int),collections.defaultdict(int),collections.defaultdict(int)
    TP_im,FP_im,GT_im,FN_im = collections.defaultdict(int),collections.defaultdict(int),collections.defaultdict(int),collections.defaultdict(int)
    # TODO: deal with found detects but no GT
    fnames = set([f for f in label_dt]+[f for f in pred_dt])
    total_imgs = len(label_dt)
    cnt = 0
    for fname in fnames:
        #bbox: [x1,y1,x2,y2]
        bbox_label = np.array([shape.up_left+shape.bottom_right for shape in label_dt[fname]])
        class_label = np.array([shape.category for shape in label_dt[fname]])
        
        bbox_pred = np.array([shape.up_left+shape.bottom_right for shape in pred_dt[fname]])
        class_pred = np.array([shape.category for shape in pred_dt[fname]])
        conf = np.array([shape.confidence for shape in pred_dt[fname]])

        #found GT but no predictions
        if class_label.shape[0] and not class_pred.shape[0]:
            cnt += 1

        #gather FP, TP and GT
        all_classes = np.concatenate((class_label, class_pred), axis=0)
        for c in set(all_classes):
            m_label = np.empty((0,))
            if class_label.shape[0]:
                m_label = class_label==c    
                
            m_pred = np.empty((0,))
            if class_pred.shape[0]:
                m_pred = np.logical_and(class_pred == c, conf >= threshold_conf)

            #update GT
            GT_im[c] += 1 if m_label.sum() else 0
            GT[c] += m_label.sum()

            # no GT labels
            if not m_label.sum():
                if m_pred.sum():
                    #print(f'fp: in {fname}, no GT but found detection')
                    FP_im[c] += 1
                    FP[c] += m_pred.sum()

            # not found any bbox found
            if not m_pred.sum():
                if m_label.sum():
                    #print(f'FN found in class: {c}, filename: {fname}')
                    FN_im[c] += 1
                    FN[c] += m_label.sum()

            if not m_label.sum() or not m_pred.sum():
                continue

            ious = bbox_iou(bbox_pred[m_pred,:], bbox_label[m_label,:])
            M = np.max(ious, axis=1) >= threshold_iou

            if M.sum():
                TP_im[c] += 1
            else:
                #print(f'fp: in {fname}, lower than threshold')
                FP_im[c] += 1 
            TP[c] += M.sum()
            FP[c] += (~M).sum()
    print(f'found {cnt} images has GT but not found any detections')
    print(f'threshold_iou: {threshold_iou}, threshold_conf: {threshold_conf}')
    #calcualte precision and recall
    epsilon=1e-16
    P,R = {},{}
    Err = {} #image level results
    total_tp, total_fp, total_gt = 0,0,0
    for c in class_map:
        if image_level:
            tp,fp,gt,fn = TP_im[c],FP_im[c],GT_im[c],FN_im[c]
        else:
            tp,fp,gt,fn = TP[c],FP[c],GT[c],FN[c]
        
        P[c] = tp / (tp + fp + epsilon)
        R[c] = tp / (gt + epsilon)
        Err[c] = FN_im[c]/total_imgs
        print(f'class {c}: ', f'error rate: {Err[c]:.4f}, ', f'precision: {P[c]:.4f}, ', f'recall: {R[c]:.4f}')

        total_tp += tp
        total_fp += fp
        total_gt += gt
    P['all'] = total_tp / (total_tp + total_fp + epsilon)
    R['all'] = total_tp / (total_gt + epsilon)
    print('')
    return P,R,Err


def plot_curve(px, dt_y, save_dir='my_curve.png', xlabel='Confidence', ylabel='Metric', y_range=[0, 1.1], step=0.1, threshold_iou=0.5):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    for k in dt_y:
        if k=='all':
            ax.plot(px, dt_y['all'], linewidth=3, color='blue', label=f'all classes @ iou_threshold={threshold_iou}')
        else:
            ax.plot(px, dt_y[k], linewidth=1, label=f'{k}')  # plot(confidence, metric)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(*y_range)
    ax.set_yticks(np.arange(*y_range, step=step))
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_csv', required=True, help='the path to the model prediction csv')
    parse.add_argument('--label_csv', required=True, help='the path to the ground truth csv')
    parse.add_argument('--threshold_iou', type=float, default=0.5, help='the iou threshold, default=0.5')
    parse.add_argument('--image_level', action='store_true', help='calculate the precision and recall on image level')
    parse.add_argument('--out_path', required=True, help='the output path for storing Precision and Recall figures')
    args = vars(parse.parse_args())

    threshold_iou = args['threshold_iou']
    model_csv = args['model_csv']
    label_csv = args['label_csv']
    out_path = args['out_path']
    image_level = args['image_level']

    if not os.path.isfile(model_csv):
        raise Exception(f'Not found the "preds.csv" in {os.path.dirname(model_csv)}')

    if not os.path.isfile(label_csv):
        raise Exception(f'Not found the "labels.csv" in {os.path.dirname(label_csv)}')

    label_dt,class_map = csv_utils.load_csv(label_csv)
    print(f'found class map: {class_map}')
    pred_dt,_ = csv_utils.load_csv(model_csv, class_map=class_map)
    confs = set([shape.confidence for fname in pred_dt for shape in pred_dt[fname]])
    #X = [0.2]
    X = [0] + list(confs) + [1]
    X = list(set(X))
    X.sort()

    Ps,Rs = collections.defaultdict(list),collections.defaultdict(list)
    Errs = collections.defaultdict(list)
    for conf in X:
        P,R,err = precision_recall(label_dt, pred_dt, class_map, threshold_iou, conf, image_level)
        for c in P:
            Ps[c].append(P[c])
        for c in R:
            Rs[c].append(R[c])
        for c in err:
            Errs[c].append(err[c]*100)

    postfix = ''
    if image_level:
        postfix = '_im_level'
    plot_curve(X, Ps, save_dir=os.path.join(out_path,'precision'+postfix+'.png'), ylabel='Precision', threshold_iou=threshold_iou, step=0.05)
    plot_curve(X, Rs, save_dir=os.path.join(out_path,'recall'+postfix+'.png'), ylabel='Recall', threshold_iou=threshold_iou, step=0.05)
    plot_curve(X, Errs, save_dir=os.path.join(out_path,'error_rate_im_level.png'), ylabel='Error Rate (%) on image level', threshold_iou=threshold_iou, y_range=[0,20.1], step=1)
    print(f'Precision and Recall figures are saved in {out_path}')
