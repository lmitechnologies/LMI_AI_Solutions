import os
import argparse
import logging
import json
import numpy as np
import collections
import glob

from label_utils.csv_utils import write_to_csv
from label_utils.rect import Rect
from label_utils.bbox_utils import convert_from_ls

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FNAME = 'labels.csv'


def get_annotations_from_json(path_json):
    """read annotation from label studio json file. Right now, it only supports bounding box annotations

    Args:
        path_json (str): the path to a directory of label studio json files

    Returns:
        dict: a map <image name, a list of Rect objects>
    """
    if os.path.splitext(path_json)=='.json':
        json_files=[path_json]
    else:
        json_files=glob.glob(os.path.join(path_json,'*.json'))
    annots = collections.defaultdict(list)
    cnt = 0
    cnt_image = 0
    for path_json in json_files:
        logger.info(f'Extracting labels from: {path_json}')
        with open(path_json) as f:    
            l = json.load(f)
            for dt in l:
                # load file name
                if 'data' not in dt:
                    raise Exception('missing "data" in json file. Ensure that the label studio export format is not JSON-MIN.')
                f = dt['data']['image'] # image web path
                fname = os.path.basename(f)
                
                for annot in dt['annotations']:
                    num_bbox = len(annot['result'])
                    if num_bbox>0:
                        cnt_image += 1
                        logger.info(f'{num_bbox} annotation(s) in {fname}')
                    for result in annot['result']:
                        # get label
                        if len(result['value']['rectanglelabels']) > 1:
                            raise Exception('each bbox should have one label, but found more than one')
                        label = result['value']['rectanglelabels'][0]
                        
                        # get bbox
                        x,y,w,h,angle = convert_from_ls(result)
                        x1,y1,w,h = list(map(int,[x,y,w,h]))
                        x2,y2 = x1+w-1, y1+h-1
                        
                        rect = Rect(im_name=fname, category=label, up_left=[x1,y1], bottom_right=[x2,y2])
                        annots[fname].append(rect)
                        cnt += 1
        logger.info(f'{cnt_image} out of {len(l)} images with annotations')
        logger.info(f'total {cnt} annotations')
    return annots


if __name__ == '__main__':
    ap = argparse.ArgumentParser('Note: currently only loads the bboxes!')
    ap.add_argument('-i', '--path_json', required=True, help='path to the label-studio json file')
    ap.add_argument('-o', '--path_out', required=True, help='either a file name or directory. If it is a directory, the default name "labels.csv" is used.')
    args = ap.parse_args()
    
    annots = get_annotations_from_json(args.path_json)
    
    _,ext = os.path.splitext(args.path_out)
    if ext!='.csv':
        args.path_out = os.path.join(args.path_out, FNAME)
        
    write_to_csv(annots, args.path_out)
    