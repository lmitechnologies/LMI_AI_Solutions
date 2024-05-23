import os
import argparse
import logging
import json
import numpy as np
import collections
import glob

from label_utils.csv_utils import write_to_csv
from label_utils.rect import Rect
from label_utils.mask import Mask
from label_utils.keypoint import Keypoint
from label_utils.bbox_utils import convert_from_ls

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LABEL_NAME = 'labels.csv'
PRED_NAME = 'preds.csv'



def lst_to_shape(result:dict, fname:str, load_confidence=False):
    """parse the result from label studio result dict, return a Rect/Mask object
    """
    result_type=result['type']
    labels = result['value'][result_type]
    if len(labels) > 1:
        raise Exception('Not support more than one labels in a bbox/polygon')
    if len(labels) == 0:
        logger.warning(f'found empty label in {fname}, skip')
        return
    label = labels[0]
    if result_type=='rectanglelabels':   
        # get bbox
        x,y,w,h,angle = convert_from_ls(result)
        x1,y1,w,h = list(map(int,[x,y,w,h]))
        x2,y2 = x1+w, y1+h
        conf = 1.0
        if load_confidence:
            conf = result['value']['score']
        rect = Rect(im_name=fname, category=label, up_left=[x1,y1], bottom_right=[x2,y2], angle=angle, confidence=conf)
        return rect
    elif result_type=='polygonlabels':
        points=result['value']['points']
        points_np=np.array(points)
        x_coordinates = (points_np[:, 0]/100*result['original_width']).astype(np.int32)
        y_coordinates = (points_np[:, 1]/100*result['original_height']).astype(np.int32)
        conf = 1.0
        if load_confidence:
            conf = result['value']['score']
        mask = Mask(im_name=fname,category=label,x_vals=list(x_coordinates),y_vals=list(y_coordinates),confidence=conf)
        return mask
    elif result_type=='keypointlabels':
        dt = result['value']
        x,y = dt['x']/100*result['original_width'],dt['y']/100*result['original_height']
        conf = 1.0
        if load_confidence:
            conf = dt['score']
        kp = Keypoint(im_name=fname,category=label,x=x,y=y,confidence=conf)
        return kp
    else:
        logger.warning(f'unsupported result type: {result_type}, skip')


def get_annotations_from_json(path_json):
    """read annotation from label studio json file.

    Args:
        path_json (str): the path to a directory of label studio json files

    Returns:
        dict: a map <image name, a list of Rect objects>
    """
    if os.path.splitext(path_json)[1]=='.json':
        json_files=[path_json]
    else:
        json_files=glob.glob(os.path.join(path_json,'*.json'))

    annots = collections.defaultdict(list)
    preds = collections.defaultdict(list)
    for path_json in json_files:
        logger.info(f'Extracting labels from: {path_json}')
        with open(path_json) as f:    
            l = json.load(f)

        cnt_anno = 0
        cnt_image = 0
        cnt_pred = 0
        cnt_wrong = 0
        for dt in l:
            # load file name
            if 'data' not in dt:
                raise Exception('missing "data" in json file. Ensure that the label studio export format is not JSON-MIN.')
            f = dt['data']['image'] # image web path
            fname = os.path.basename(f)
            
            if 'annotations' in dt:
                cnt = 0
                for annot in dt['annotations']:
                    num_labels = len(annot['result'])
                    if num_labels>0:
                        cnt += 1
                    for result in annot['result']:
                        shape = lst_to_shape(result,fname)
                        if shape is not None:
                            annots[fname].append(shape)
                            cnt_anno += 1
                            
                    if 'prediction' in annot and 'result' in annot['prediction']:
                        for result in annot['prediction']['result']:
                            shape = lst_to_shape(result,fname,load_confidence=True)
                            if shape is not None:
                                preds[fname].append(shape)
                                cnt_pred += 1
                if cnt>0:
                    cnt_image += 1
                if cnt==0 and dt['total_annotations']>0:
                    cnt_wrong += 1
                    logger.warning(f'found 0 annotation in {fname}, but lst claims total_annotations = {dt["total_annotations"]}')

            if 'predictions' in dt:
                for pred in dt['predictions']:
                    if isinstance(pred, dict):
                        for result in pred['result']:
                            shape = lst_to_shape(result,fname,load_confidence=True)
                            if shape is not None:
                                preds[fname].append(shape)
                                cnt_pred += 1

        logger.info(f'{cnt_image} out of {len(l)} images have annotations')
        if cnt_wrong>0:
            logger.info(f'{cnt_wrong} images with total_annotations > 0, but found 0 annotation')
        logger.info(f'total {cnt_anno} annotations')
        logger.info(f'total {cnt_pred} predictions')
    return annots, preds


if __name__ == '__main__':
    ap = argparse.ArgumentParser('Convert label studio json file to csv format')
    ap.add_argument('-i', '--path_json', required=True, help='the directory of label-studio json files')
    ap.add_argument('-o', '--path_out', required=True, help='output directory')
    args = ap.parse_args()
    
    annots,preds = get_annotations_from_json(args.path_json)
    
    if os.path.isfile(args.path_out):
        raise Exception('The output path should be a directory')
    
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)    
    write_to_csv(annots, os.path.join(args.path_out, LABEL_NAME))
    write_to_csv(preds, os.path.join(args.path_out, PRED_NAME))
