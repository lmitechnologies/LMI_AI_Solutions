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


logging.basicConfig()
logger = logging.getLogger()
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
        shape.up_left = rot90(shape.up_left[0],shape.up_left[1],h,w)
        shape.bottom_right = rot90(shape.bottom_right[0],shape.bottom_right[1],h,w)
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


def rot90_imgs_with_csv(path_imgs, path_csv):
    """
    rotate images and its annotations counterclock 90 degrees.
    Arguments:
        path_imgs(str): the image folder
        path_csv(str): the path of csv annotation file
    Return:
        name_to_im(dict): the map <output image name, im>
        shapes(dict): the map <original image name, a list of shape objects>, where shape objects are annotations
    """
    file_list = glob.glob(os.path.join(path_imgs, '*.png'))
    shapes,_ = csv_utils.load_csv(path_csv, path_img=path_imgs)
    name_to_im = {}
    for file in file_list:
        im = cv2.imread(file)
        h,w = im.shape[:2]
        im_name = os.path.basename(file)
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
    ap.add_argument('--path_imgs', required=True, help='the path to images')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    # ap.add_argument('--degree', type=int, default=90, help='the rotation degree, default=90')
    ap.add_argument('--path_out', required=True, help='the path to resized images')
    args = vars(ap.parse_args())


    path_imgs = args['path_imgs']
    path_out = args['path_out']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    
    #check if annotation exists
    if not os.path.isfile(path_csv):
        raise Exception(f'cannot find file: {path_csv}')

    #resize images with annotation csv file
    name_to_im,shapes = rot90_imgs_with_csv(path_imgs, path_csv)

    # create output path
    assert path_imgs!=path_out, 'input and output path must be different'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    #write images and csv file
    for im_name in name_to_im:
        logger.info(f'writting to {im_name}')
        cv2.imwrite(os.path.join(path_out,im_name), name_to_im[im_name])
    csv_utils.write_to_csv(shapes, os.path.join(path_out,'labels.csv'))
