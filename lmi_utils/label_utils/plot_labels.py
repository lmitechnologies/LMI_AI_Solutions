import numpy as np
import random
import cv2
import os
import json
import logging

#LMI packages
# from label_utils.shapes import Rect, Mask, Keypoint, Brush
from dataset_utils.representations import Dataset,AnnotationType
from label_utils.plot_utils import plot_one_box, plot_one_polygon, plot_one_pt, plot_one_brush
from label_utils.bbox_utils import rotate


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_shape(shape, im, color_map):
    img_h, img_w = im.shape[:2]
    if shape.type == AnnotationType.BOX:
        x1,y1, x2, y2, angle = shape.value.coords()
        width = x2 - x1
        height = y2 - y1
        # rotated rectangle
        if angle > 0:
            rotated_rect = rotate(x1, y1, width, height, angle)
        else:
            rotated_rect = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        plot_one_polygon(np.array([rotated_rect]), im, label=shape.label_id, color=color_map[shape.label_id])
    elif shape.type == AnnotationType.POLYGON:
        pts = shape.value.to_numpy().reshape((-1, 1, 2)).astype(int)
        plot_one_polygon(pts, im, label=shape.label_id, color=color_map[shape.label_id])
    elif shape.type == AnnotationType.MASK:
        x,y = shape.value.coords(h=img_h, w=img_w)
        plot_one_brush(x,y,im,label=shape.label_id,color=color_map[shape.label_id])
    elif shape.type == AnnotationType.KEYPOINT:
        x,y,_ = shape.value.coords()
        plot_one_pt([x,y], im, label=shape.label_id, color=color_map[shape.label_id])
    else:
        raise Exception(f'Unknown shape: {type(shape)}')
    return



if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--path_imgs', required=True, help='the path to the input image folder')
    ap.add_argument('-o','--path_out', required=True, help='the path to the output folder')
    ap.add_argument('--path_json', default='labels.json', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.json" in path_imgs')
    args = vars(ap.parse_args())

    path_imgs = args['path_imgs']
    path_json = args['path_json'] if args['path_json']!='labels.json' else os.path.join(path_imgs, args['path_json'])
    output_path = args['path_out']
    assert path_imgs!=output_path, 'output path must be different with input path'
    # fname_to_shape, class_map = load_csv(path_json, path_imgs, class_map)
    
    dataset = Dataset.load(path_json)
    base_prefix = dataset.base_path
    
    # init color map
    color_map = {}
    for cls in dataset.get_label_ids():
        logger.info(f'CLASS: {cls}')
        color_map[cls] = tuple([random.randint(0,255) for _ in range(3)])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for f in dataset.files:
        file_path = f.relative_path(base_prefix)
        im_name = os.path.basename(file_path)
        logger.info(f'processing {im_name}')
        if not os.path.exists(os.path.join(path_imgs, file_path)):
            raise Exception(f'file not found: {file_path}')
        im = cv2.imread(os.path.join(path_imgs, file_path))
        h,w = im.shape[:2]
        for shape in f.annotations:
            plot_shape(shape, im, color_map)
        outname = os.path.join(output_path, im_name)
        cv2.imwrite(outname, im)
