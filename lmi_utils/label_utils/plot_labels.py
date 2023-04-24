import numpy as np
import random
import cv2
import os
import json

#LMI packages
from label_utils.csv_utils import load_csv
from label_utils import rect, mask
from label_utils.plot_utils import plot_one_box, plot_one_polygon


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', required=True, help='the path to the input image folder')
    ap.add_argument('--path_out', required=True, help='the path to the output folder')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--class_map_json', default=None, help='[optinal] the path of a class map json file')
    args = vars(ap.parse_args())

    path_imgs = args['path_imgs']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    output_path = args['path_out']
    if args['class_map_json']:
        with open(args['class_map_json']) as f:
            class_map = json.load(f)
        print(f'loaded class map: {class_map}')
    else:
        class_map = None

    if not os.path.isfile(path_csv):
        raise Exception(f'Not found file: {path_csv}')

    assert path_imgs!=output_path, 'output path must be different with input path'

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    fname_to_shape, class_map = load_csv(path_csv, path_imgs, class_map)
    min_id = min(class_map.values())
    colors = [(0,0,255),(255,0,0),(0,255,0),(102,51,153),(255,140,0),(105,105,105),(127,25,27),(9,200,100)]
    color_map = {}
    for cls in class_map:
        i = class_map[cls]
        if min_id != 0:
            i -= 1
        if i < len(colors):
            color_map[cls] = colors[i]
        else:
            color_map[cls] = tuple([random.randint(0,255) for _ in range(3)])

    for im_name in fname_to_shape:
        print(f'[FILE] {im_name}')
        shapes = fname_to_shape[im_name]
        im = cv2.imread(shapes[0].fullpath)
        for shape in shapes:
            if isinstance(shape, rect.Rect):
                box = shape.up_left + shape.bottom_right
                plot_one_box(box, im, label=shape.category, color=color_map[shape.category])
            elif isinstance(shape, mask.Mask):
                pts = np.array([[x,y] for x,y in zip(shape.X,shape.Y)])
                pts = pts.reshape((-1, 1, 2))
                plot_one_polygon(pts, im, label=shape.category, color=color_map[shape.category])
        outname = os.path.join(output_path, im_name)
        cv2.imwrite(outname, im)
