import os
import argparse
import logging
import json
import cv2
import numpy as np

from label_utils.COCO_dataset import COCO_Dataset, rotate
from label_utils.plot_utils import plot_one_polygon


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def get_annotations_from_json(path_json, path_imgs, path_out, conf):
    id_to_img = {}
    id_to_img_name = {}
    with open(path_json) as f:
        dt = json.load(f)
    images = dt['images']
    for m in images:
        fname = m['file_name']
        path_img = os.path.join(path_imgs, fname)
        if not os.path.isfile(path_img):
            raise Exception(f'Cannot find the file: {path_img}')
        id_to_img[m['id']] = cv2.imread(path_img)
        id_to_img_name[m['id']] = fname
        

    for annot in dt['annotations']:
        bbox = annot['bbox']
        cat_id = int(annot['category_id'])
        img_id = annot['image_id']
        score = float(annot['score'])
        segs = annot['segmentation']
        
        if score<=conf:
            continue
        logger.info(f'score: {score}')
        
        if len(bbox) == 5:
            x,y,w,h,angle = bbox
            angle = float(angle)
        else:
            x,y,w,h = bbox
            angle = 0
        
        # load img
        I = id_to_img[img_id]
            
        # plot 
        x,y,w,h = list(map(int,[x,y,w,h]))
        pts = rotate(x,y,w,h,angle,unit='radian',rot_center='center')
        pts = pts.reshape((-1, 1, 2))
        plot_one_polygon(pts, I, label=f'{cat_id}: {score:.2f}')
        
        for seg in segs:
            pts = np.array(list(map(int, seg)))
            pts = pts.reshape((-1, 1, 2))
            plot_one_polygon(pts, I, label=f'{cat_id}: {score:.2f}')
    
    for id in id_to_img_name:
        path = os.path.join(path_out,id_to_img_name[id])
        logger.info(f'writing to {path}')
        I = id_to_img[id]
        cv2.imwrite(path,I)
        


if __name__ == '__main__':
    ap = argparse.ArgumentParser('Note: currently only loads the bboxes!')
    ap.add_argument('--path_imgs', required=True)
    ap.add_argument('--path_json', required=True)
    ap.add_argument('-o','--path_out', required=True, help='output path')
    ap.add_argument('--confidence', default=0.6, type=float)
    args = ap.parse_args()
    
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)
    
    get_annotations_from_json(args.path_json, args.path_imgs, args.path_out, args.confidence)
    