import os
import json
import numpy as np
import cv2
import collections

from label_utils.csv_utils import load_csv
from label_utils import rect
from label_utils import mask


OUT_NAME = 'lst.json'


def rect_to_lst(rect_obj, width, height, is_pred):
    x1,y1 = rect_obj.up_left
    x2,y2 = rect_obj.bottom_right
    w,h = x2-x1, y2-y1
    box = {
        'original_width': width,
        'original_height': height,
        'image_rotation': 0,
        'value': {
            'x': x1 / width * 100,
            'y': y1 / height * 100,
            'width': w / width * 100,
            'height': h / height * 100,
            'rotation': rect_obj.angle,
            'rectanglelabels': [rect_obj.category]
        },
        'from_name': 'label',
        'to_name': 'image',
        'type': 'rectanglelabels'
    }
    if is_pred:
        box['value']['score'] = rect_obj.confidence
    return box


def mask_to_lst(mask_obj, width, height, is_pred):
    X,Y = mask_obj.X, mask_obj.Y
    polygon = {
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            'value': {
                'points': [],
                'polygonlabels': [mask_obj.category]
            },
            'from_name': 'polygon',
            'to_name': 'image',
            'type': 'polygonlabels'
        }
    for i in range(0, len(X)):
        x = X[i] / width * 100
        y = Y[i] / height * 100
        polygon['value']['points'].append([x,y])
    if is_pred:
        polygon['value']['score'] = mask_obj.confidence
    return polygon


def init_label_obj(path_img, is_pred):
    # TODO: support BOTH annotations and predictions
    label_obj = {}
    if not is_pred:
        label_obj['annotations'] = [
            {
                'result': []
            }
        ]
    else:
        label_obj['predictions'] = [
            {
                'model_version': 'prediction',
                'result': []
            }
        ]
    label_obj['data'] = {
        'image':path_img
    }
    return label_obj


def write_to_lst(shapes, out_path, images_path, gs_path, width, height, is_pred):
    labels = []
    for fname in shapes:
        if images_path is not None:
            im = cv2.imread(os.path.join(images_path, fname))
            height,width = im.shape[:2]

        label_obj = init_label_obj(os.path.join(gs_path, fname), is_pred)
        target = 'predictions' if is_pred else 'annotations'
        for shape in shapes[fname]:
            if isinstance(shape, rect.Rect):
                box = rect_to_lst(shape, width, height, is_pred)
                label_obj[target][0]['result'].append(box)
            elif isinstance(shape, mask.Mask):
                polygon = mask_to_lst(shape, width, height, is_pred)
                label_obj[target][0]['result'].append(polygon)
            else:
                raise Exception(f'Invalid shape type: {type(shape)}')
        labels.append(label_obj)

    # save to json
    with open(os.path.join(out_path,OUT_NAME), 'w') as f:
        json.dump(labels, f, indent=4)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, help='path to the csv file')
    ap.add_argument('--path_imgs', type=str)
    ap.add_argument('--wh', type=str, help='width and height of the images separated by a comma. Assume that images are the same size')
    ap.add_argument('--path_gs', type=str, required=True, help='path to the google storage bucket directory')
    ap.add_argument('--path_out', '-o', type=str, required=True)
    ap.add_argument('--pred', action='store_true', help='if the csv file is a prediction file')
    args = ap.parse_args()

    if args.path_imgs is None and args.wh is None:
        raise Exception('Provide the path to the images or the width and height of the images')
    if args.wh is not None:
        wh = args.wh.split(',')
        width = int(wh[0])
        height = int(wh[1])

    if os.path.isfile(args.path_out):
        raise Exception('The output path should be a directory')
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)

    shapes = load_csv(args.csv)[0]
    write_to_lst(shapes, args.path_out, args.path_imgs, args.path_gs, width, height, args.pred)
