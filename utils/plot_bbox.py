from ast import arg
import numpy as np
import random
import cv2
import os

from csv_utils import load_csv

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    arguments:
        x(list):      a box likes [x1,y1,x2,y2]
        img(np array):    a opencv image object in BGR format
        color(tuple):  color to draw rectangle, such as (0,255,0)
        label(str):  the class name
        line_thickness(int): the thickness of the line
    return:
        no return
    """

    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_path', required=True, help='the path to the input image folder, where it has labels.csv')
    #ap.add_argument('-c', '--csv_path', required=True, help='the path to the csv file')
    ap.add_argument('-o', '--output_path', required=True, help='the path to the output folder')
    args = vars(ap.parse_args())

    input_path = args['input_path']
    csv_path = os.path.join(input_path, 'labels.csv')
    output_path = args['output_path']

    if not os.path.isfile(csv_path):
        raise Exception(f'the csv file does not exist: {csv_path}')

    assert input_path!=output_path, 'output path must be different with input path'

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    fname_to_shape, class_map = load_csv(csv_path, input_path)
    for im_name in fname_to_shape:
        print(f'[FILE] {im_name}')
        shapes = fname_to_shape[im_name]
        im = cv2.imread(shapes[0].fullpath)
        for shape in shapes:
            box = shape.up_left + shape.bottom_right
            plot_one_box(box, im, label=shape.category)
        outname = os.path.join(output_path, im_name)
        cv2.imwrite(outname, im)
