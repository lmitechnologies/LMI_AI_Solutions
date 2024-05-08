#built-in packages
import os
import glob
import shutil
import logging
import numpy as np

#3rd party packages
import cv2

#LMI packages
from label_utils import mask, rect, csv_utils
from image_utils.path_utils import get_relative_paths


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def rot90(x,y,h,w):
    """rotate point (x,y) 90 degree counter-clockwise

    Args:
        x (float): x coordinate of a point
        y (float): y coordinate of a point
        h (int): the height of the image
        w (int): the width of the image

    Returns:
        list: a rotated point of (x,y)
    """
    return [y,w-x]


def rot90_shape(shape,h,w):
    if isinstance(shape, rect.Rect):
        x0,y0,x2,y2 = shape.up_left+shape.bottom_right
        x0,x2 = np.clip([x0,x2],a_min=0,a_max=w)
        y0,y2 = np.clip([y0,y2],a_min=0,a_max=h)
        x1,y1 = x2,y0
        x3,y3 = x0,y2
        shape.up_left = rot90(x1,y1,h,w)
        shape.bottom_right = rot90(x3,y3,h,w)
    elif isinstance(shape, mask.Mask):
        new_x,new_y = [],[]
        for x,y in zip(shape.X,shape.Y):
            nx,ny = rot90(x,y,h,w)
            new_x.append(nx)
            new_y.append(ny)
        shape.X = new_x
        shape.Y = new_y
    else:
        raise Exception("Found unsupported classes. Supported classes are mask and rect")


def rot90_imgs_with_csv(path_imgs, path_csv, recursive):
    """
    rotate images and its annotations counterclock 90 degrees.
    Arguments:
        path_imgs(str): the image folder
        path_csv(str): the path of csv annotation file
    Return:
        name_to_im(dict): the map <output image name, im>
        shapes(dict): the map <original image name, a list of shape objects>, where shape objects are annotations
    """
    paths = get_relative_paths(path_imgs,recursive)
    shapes,_ = csv_utils.load_csv(path_csv, path_img=path_imgs)
    name_to_im = {}
    for p in paths:
        im_name = os.path.basename(p)
        if im_name not in shapes:
            continue

        p = os.path.join(path_imgs,p)
        im = cv2.imread(p)
        h,w = im.shape[:2]
        out_name = os.path.splitext(im_name)[0] + f'_rot90' + '.png'
        
        im2 = np.rot90(im)
        name_to_im[out_name] = im2

        for i in range(len(shapes[im_name])):
            rot90_shape(shapes[im_name][i],h,w)
            shapes[im_name][i].im_name = out_name

    return name_to_im, shapes



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to images')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--path_out', '-o', required=True, help='the path to resized images')
    ap.add_argument('--append', action='store_true', help='append to the existing output csv file')
    ap.add_argument('--recursive', action='store_true', help='search images recursively')
    args = vars(ap.parse_args())


    path_imgs = args['path_imgs']
    path_out = args['path_out']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    recursive = args['recursive']
    
    #check if annotation exists
    if not os.path.isfile(path_csv):
        raise Exception(f'cannot find file: {path_csv}')

    #resize images with annotation csv file
    name_to_im,shapes = rot90_imgs_with_csv(path_imgs, path_csv, recursive)

    # create output path
    assert path_imgs!=path_out, 'input and output path must be different'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    #write images and csv file
    for im_name in name_to_im:
        logger.info(f'writting to {im_name}')
        cv2.imwrite(os.path.join(path_out,im_name), name_to_im[im_name])
    csv_utils.write_to_csv(shapes, os.path.join(path_out,'labels.csv'), overwrite=not args['append'])
