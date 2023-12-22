#built-in packages
import os
import glob
import shutil
import logging

#3rd party packages
import cv2

#LMI packages
from label_utils import mask, rect, csv_utils


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def resize_imgs_with_csv(path_imgs, path_csv, output_imsize, path_out, save_bg_images):
    """
    resize images and its annotations with a csv file
    if the aspect ratio changes, it will generate warnings.
    Arguments:
        path_imgs(str): the image folder
        path_csv(str): the path of csv annotation file
        output_imsize(list): a list of output image size [w,h]
    Return:
        name_to_im(dict): the map <output image name, im>
        shapes(dict): the map <original image name, a list of shape objects>, where shape objects are annotations
    """
    
    shapes,_ = csv_utils.load_csv(path_csv, path_img=path_imgs)
    name_to_im = {}
    cnt_bg = 0
    W,H = output_imsize
    ratio_out = W/H
    for path, _, files in os.walk(path_imgs):
        for im_name in files:
            im = cv2.imread(os.path.join(path, im_name))
            if im is None:
                logger.warning(f'The file: {im_name} might be not a image or corrupted, skip!')
                continue
            
            h,w = im.shape[:2]
            # found bg image
            if im_name not in shapes:
                if not save_bg_images:
                    continue
                cnt_bg += 1
                logger.info(f'Input file: {im_name} with size of [{w},{h}] has no labels')
            else:
                logger.info(f'Input file: {im_name} with size of [{w},{h}]')
            
            ratio_in = w/h
            if ratio_in != ratio_out:
                logger.warning(f'file: {im_name}, asepect ratio changed from {ratio_in} to {ratio_out}')
            
            rx,ry = W/w, H/h
            im2 = cv2.resize(im, dsize=tuple(output_imsize))
            
            out_name = os.path.splitext(im_name)[0] + f'_resized_{W}x{H}' + '.png'
            logger.info(f'write to {out_name}')
            cv2.imwrite(os.path.join(path_out,out_name), im2)
            
            name_to_im[out_name] = im2
            for i in range(len(shapes[im_name])):
                if isinstance(shapes[im_name][i], rect.Rect):
                    x,y = shapes[im_name][i].up_left
                    shapes[im_name][i].up_left = [int(x*rx), int(y*ry)]
                    x,y = shapes[im_name][i].bottom_right
                    shapes[im_name][i].bottom_right = [int(x*rx), int(y*ry)]
                    shapes[im_name][i].im_name = out_name
                elif isinstance(shapes[im_name][i], mask.Mask):
                    shapes[im_name][i].X = [int(v*rx) for v in shapes[im_name][i].X]
                    shapes[im_name][i].Y = [int(v*ry) for v in shapes[im_name][i].Y]
                    shapes[im_name][i].im_name = out_name
                else:
                    raise Exception("Found unsupported classes. Supported classes are mask and rect")
    if cnt_bg:
        logger.info(f'found {cnt_bg} images with no labels. These images will be used as background training data in YOLO')
    return name_to_im, shapes



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to images')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--out_imsz', required=True, help='the output image size [w,h], w and h are separated by a comma')
    ap.add_argument('--path_out', '-o', required=True, help='the path to resized images')
    ap.add_argument('--bg_images', action='store_true', help='save images with no labels')
    args = vars(ap.parse_args())

    output_imsize = list(map(int,args['out_imsz'].split(',')))
    assert len(output_imsize)==2, 'the output image size must be two ints'
    logger.info(f'output image size: {output_imsize}')
    
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

    #resize images with annotation csv file
    name_to_im,shapes = resize_imgs_with_csv(path_imgs, path_csv, output_imsize, path_out, args['bg_images'])

    #write csv file
    csv_utils.write_to_csv(shapes, os.path.join(path_out,'labels.csv'))
