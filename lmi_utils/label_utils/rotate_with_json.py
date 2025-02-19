import argparse
import os
import cv2
import numpy as np
import logging
import json
import math
from dataset_utils.representations import Dataset, AnnotationType
from dataset_utils.mask_encoder import mask2rle
from label_utils.bbox_utils import rotate, get_rotated_bbox
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def rotate_bbox(bbox, rotation_matrix):
    """
    Rotate an axis-aligned bounding box using an affine rotation matrix.
    """
    # Unpack the bounding box coordinates.
    x_min, y_min, x_max, y_max = bbox
    
    # Define the four corners of the bounding box.
    corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])
    
    # Convert corners to homogeneous coordinates (add a column of ones).
    ones = np.ones((corners.shape[0], 1))
    corners_hom = np.hstack([corners, ones])
    
    # Apply the rotation matrix to each corner.
    rotated_corners = np.dot(rotation_matrix, corners_hom.T).T
    
    # Find the new bounding box coordinates.
    x_min_new = np.min(rotated_corners[:, 0])
    y_min_new = np.min(rotated_corners[:, 1])
    x_max_new = np.max(rotated_corners[:, 0])
    y_max_new = np.max(rotated_corners[:, 1])
    
    return [x_min_new, y_min_new, x_max_new, y_max_new]

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
        new_h, new_w = rotated_img.shape[:2]
        
        
        for annot in file.annotations:
            if annot.type == AnnotationType.BOX:
                x_min, y_min, x_max, y_max, theta = annot.value.coords()
                x_min, y_min, x_max, y_max = rotate_bbox([x_min, y_min, x_max, y_max], rotation_matrix)
                annot.value.x_min, annot.value.y_min = x_min, y_min
                annot.value.x_max, annot.value.y_max = x_max, y_max
                
            elif annot.type == AnnotationType.POLYGON:
                points = annot.value.to_numpy()
                for i in range(len(points)):
                    points[i] = np.dot(rotation_matrix[:, :2], [points[i][0], points[i][1]]) + rotation_matrix[:, 2]
                annot.value.points = points.astype(float).tolist()
            
            elif annot.type == AnnotationType.KEYPOINT:
                x, y = annot.value.x, annot.value.y
                new_x, new_y = np.dot(rotation_matrix[:, :2], [x, y]) + rotation_matrix[:, 2]
                annot.value.x, annot.value.y = new_x.astype(float), new_y.astype(float)
            
            elif annot.type == AnnotationType.MASK:
                mask = annot.value.to_numpy(h=height, w=width)
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
        
        
        
