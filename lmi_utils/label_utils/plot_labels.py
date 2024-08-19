import numpy as np
import random
import cv2
import os
import json
import logging

#LMI packages
from label_utils.csv_utils import load_csv
from label_utils.shapes import Rect, Mask, Keypoint
from label_utils.plot_utils import plot_one_box, plot_one_polygon, plot_one_pt
from label_utils.bbox_utils import rotate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_shape(shape, im, color_map):
    if isinstance(shape, Rect):
        if shape.angle > 0:
            x1 , y1 = shape.up_left
            x2 , y2 = shape.bottom_right
            angle = shape.angle
            width = x2 - x1
            height = y2 - y1
            # rotated rectangle
            rotated_rect = rotate(x1, y1, width, height, angle)
            # draw the rotated rectangle
            plot_one_polygon(np.array([rotated_rect]), im, label=shape.category, color=color_map[shape.category])
        else:
            box = shape.up_left + shape.bottom_right
            plot_one_box(box, im, label=shape.category, color=color_map[shape.category])
    elif isinstance(shape, Mask):
        pts = np.array([[x,y] for x,y in zip(shape.X,shape.Y)])
        pts = pts.reshape((-1, 1, 2)).astype(int)
        plot_one_polygon(pts, im, label=shape.category, color=color_map[shape.category])
    elif isinstance(shape, Keypoint):
        pt = shape.x, shape.y
        plot_one_pt(pt, im, label=shape.category, color=color_map[shape.category])
    else:
        raise Exception(f'Unknown shape: {type(shape)}')
    return



if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--path_imgs', required=True, help='the path to the input image folder')
    ap.add_argument('-o','--path_out', required=True, help='the path to the output folder')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--class_map_json', default=None, help='[optinal] the path of a class map json file')
    args = vars(ap.parse_args())

    path_imgs = args['path_imgs']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    output_path = args['path_out']
    if args['class_map_json']:
        with open(args['class_map_json']) as f:
            class_map = json.load(f)
        logger.info(f'loaded class map: {class_map}')
    else:
        class_map = None

    if not os.path.isfile(path_csv):
        raise Exception(f'Not found file: {path_csv}')
    assert path_imgs!=output_path, 'output path must be different with input path'
    fname_to_shape, class_map = load_csv(path_csv, path_imgs, class_map)
    
    # init color map
    color_map = {}
    for cls in sorted(class_map.keys()):
        logger.info(f'CLASS: {cls}')
        color_map[cls] = tuple([random.randint(0,255) for _ in range(3)])

    logger.info(f'found {len(fname_to_shape)} images')
    for im_name in fname_to_shape:
        shapes = fname_to_shape[im_name]
        im = cv2.imread(shapes[0].fullpath)
        if im is None:
            logger.warning(f'cannot read image: {shapes[0].fullpath}')
            continue
        for shape in shapes:
            plot_shape(shape, im, color_map)
            
        os.makedirs(output_path, exist_ok=True)
        outname = os.path.join(output_path, im_name)
        cv2.imwrite(outname, im)
