#built-in packages
import os
import logging
import cv2
import numpy as np

#LMI packages
from label_utils import csv_utils
from label_utils.shapes import Rect, Mask, Keypoint
from system_utils.path_utils import get_relative_paths


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def flip_imgs_with_csv(path_imgs, path_csv, flip, path_out, save_bg_images, recursive):
    """
    resize images and its annotations with a csv file
    if the aspect ratio changes, it will generate warnings.
    Arguments:
        path_imgs(str): the image folder
        path_csv(str): the path of csv annotation file
        flip(list): a list of two elements, [flip_x, flip_y], where flip_x and flip_y are boolean values
    Return:
        shapes(dict): the map <original image name, a list of shape objects>, where shape objects are annotations
    """
    
    shapes,_ = csv_utils.load_csv(path_csv, path_img=path_imgs)
    cnt_bg = 0
    files = get_relative_paths(path_imgs, recursive)
    for f in files:
        im_name = os.path.basename(f)
        im = cv2.imread(os.path.join(path_imgs, f))
        h,w = im.shape[:2]
        
        # found bg image
        if im_name not in shapes:
            if not save_bg_images:
                continue
            cnt_bg += 1
            logger.info(f'{im_name}: wh of [{w},{h}] has no labels')
        else:
            logger.info(f'{im_name}: wh of [{w},{h}]')
        
        # flip image
        flipx, flipy = flip
        im2 = im.copy()
        if flipx:
            im2 = np.flip(im2, 1)
        if flipy:
            im2 = np.flip(im2, 0)
            
        tx = 'x' if flipx else ''
        ty = 'y' if flipy else ''
        out_name = os.path.splitext(im_name)[0] + f'_flip_{tx}{ty}' + '.png'
        logger.info(f'write to {out_name}')
        cv2.imwrite(os.path.join(path_out,out_name), im2)
        
        for i in range(len(shapes[im_name])):
            shapes[im_name][i].im_name = out_name
            if isinstance(shapes[im_name][i], Rect):
                x1,y1 = shapes[im_name][i].up_left
                x2,y2 = shapes[im_name][i].bottom_right
                if flipx:
                    x1 = w - x1
                    x2 = w - x2
                if flipy:
                    y1 = h - y1
                    y2 = h - y2
                shapes[im_name][i].up_left = [min(x1,x2), min(y1,y2)]
                shapes[im_name][i].bottom_right = [max(x1,x2), max(y1,y2)]
            elif isinstance(shapes[im_name][i], Mask):
                xs = shapes[im_name][i].X
                ys = shapes[im_name][i].Y
                if flipx:
                    xs = [w-x for x in xs]
                if flipy:
                    ys = [h-y for y in ys]
                shapes[im_name][i].X = xs
                shapes[im_name][i].Y = ys
            elif isinstance(shapes[im_name][i], Keypoint):
                if flipx:
                    shapes[im_name][i].x = w-shapes[im_name][i].x
                if flipy:
                    shapes[im_name][i].y = h-shapes[im_name][i].y
    if cnt_bg:
        logger.info(f'found {cnt_bg} images with no labels. These images will be used as background training data in YOLO')
    return shapes



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to images')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--path_out', '-o', required=True, help='the path to resized images')
    ap.add_argument('--flip_lr',action='store_true', help='flip images horizontally')
    ap.add_argument('--flip_ud',action='store_true', help='flip images vertically')
    ap.add_argument('--bg', action='store_true', help='save background images that have no labels')
    ap.add_argument('--append', action='store_true', help='append to the existing output csv file')
    ap.add_argument('--recursive', action='store_true', help='search images recursively')
    args = vars(ap.parse_args())
    
    path_imgs = args['path_imgs']
    path_out = args['path_out']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    
    #check if annotation exists
    if not os.path.isfile(path_csv):
        raise Exception(f'cannot find file: {path_csv}')
    
    # create output path
    assert path_imgs!=path_out, 'input and output path must be different'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    flip = [args['flip_lr'], args['flip_ud']]
    #resize images with annotation csv file
    shapes = flip_imgs_with_csv(path_imgs, path_csv, flip, path_out, args['bg'], args['recursive'])

    #write csv file
    csv_utils.write_to_csv(shapes, os.path.join(path_out,'labels.csv'), overwrite=not args['append'])
