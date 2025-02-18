import argparse
import os
import cv2
import numpy as np
import logging
import json
import math
from dataset_utils.representations import Dataset, AnnotationType
from dataset_utils.mask_encoder import mask2rle
from label_utils.bbox_utils import rotate
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to images')
    ap.add_argument('--path_json', default='labels.json', help='[optional] the path of a json file that corresponds to path_imgs, default="labels.json" in path_imgs')
    ap.add_argument('--path_out', '-o', required=True, help='the path to resized images')
    ap.add_argument('--angle', default=90, type=int, help='the angle to rotate the images')
    ap.add_argument('--counter-clockwise', action='store_true', help='rotate the images counter-clockwise')
    ap.add_argument('--bg', action='store_true', help='save background images that have no labels')
    args = vars(ap.parse_args())
    return args

def rotate_dataset(dataset, angle, path_imgs,path_out,clockwise=False, save_bg_images=False):
    """
    Rotate annotations clockwise or counterclockwise
    """
    if clockwise is False:
        angle = -angle
    cnt_bg = 0
    for file in dataset.files:
        
        if not file.has_annotations:
            if not save_bg_images:
                continue
            cnt_bg += 1
            logger.info(f'{os.path.basename(file.path)}: has no labels')
        
        if not os.path.isfile(os.path.join(path_imgs, file.path)):
            logger.warning(f'Not found file: {file.path}')
            continue
        
        # read the image:
        img = cv2.imread(os.path.join(path_imgs, file.path))
        if img is None:
            logging.warning(f'cannot read image: {file.path}')
            continue
        
        # rotate the image
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        
        for annot in file.annotations:
            if annot.type == AnnotationType.BOX:
                x_min, y_min, x_max, y_max, theta = annot.value.coords()
                if angle != 0:
                    corners = rotate(x_min, y_min, x_max-x_min, y_max-y_min, angle, 'up_left', 'degree')
                else:
                    corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                
                # rotate the corners
                rotated_corners = cv2.transform(np.array([corners]), rotation_matrix)[0]
                
                # get the new coordinates
                if theta != 0:
                    new_x, new_y, new_w, new_h, new_theta = cv2.minAreaRect(rotated_corners)
                    new_xmin, new_ymin, new_xmax, new_ymax = new_x-new_w/2, new_y-new_h/2, new_x+new_w/2, new_y+new_h/2
                    annot.value.x_min, annot.value.y_min, annot.value.x_max, annot.value.y_max = new_xmin, new_ymin, new_xmax, new_ymax
                    annot.value.angle = new_theta
                else:
                    x_min, y_min = rotated_corners[0]
                    x_max, y_max = rotated_corners[2]
                    annot.value.x_min, annot.value.y_min, annot.value.x_max, annot.value.y_max = x_min, y_min, x_max, y_max
                    
                
            elif annot.type == AnnotationType.POLYGON:
                points = annot.value.to_numpy()
                for i in range(len(points)):
                    points[i] = np.dot(rotation_matrix[:, :2], [points[i][0], points[i][1]]) + rotation_matrix[:, 2]
                annot.value.points = points.tolist()
            
            elif annot.value.type == AnnotationType.KEYPOINT:
                x, y = annot.x, annot.y
                new_x, new_y = np.dot(rotation_matrix[:, :2], [x, y]) + rotation_matrix[:, 2]
                annot.value.x, annot.value.y = new_x, new_y
            
            elif annot.type == AnnotationType.MASK:
                mask = annot.value.mask.to_numpy()
                mask = cv2.warpAffine(mask, rotation_matrix, (width, height))
                annot.value.mask = mask2rle(mask)
            
            else:
                logging.warning(f'unsupported annotation type: {annot.type}')
        
        # update the file
        
        ext = os.path.basename(file.path).split('.')[-1]
        out_file = os.path.basename(file.path).replace(f'.{ext}', f'_rotated_{angle*-1}.{ext}') # -1 so that the angle is positive for clockwise rotation
        cv2.imwrite(os.path.join(path_out, out_file), rotated_img)
        file.height, file.width = rotated_img.shape[:2]
        file.path = out_file
    return dataset

def main(args):
    path_imgs = args['path_imgs']
    path_json = args['path_json']
    path_out = args['path_out']
    angle = args['angle']
    path_json = args['path_json'] if args['path_json']!='labels.json' else os.path.join(path_imgs, args['path_json'])
    counter_clockwise = args['counter_clockwise']
    save_bg = args['bg']
    
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        

    
    if not os.path.exists(path_json):
        raise Exception(f'json file not found: {path_json}')
    
    # read the dataset
    dataset = Dataset.load(path_json)
    
    # rotate the dataset
    dataset = rotate_dataset(dataset, angle, path_imgs, path_out, counter_clockwise, save_bg)
    
    # save the rotated dataset
    dataset.save(os.path.join(path_out, 'labels.json'))

if __name__ == '__main__':
    args = get_args()
    main(args)
        
        
        
