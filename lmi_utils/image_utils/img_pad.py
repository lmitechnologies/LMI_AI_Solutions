import cv2
import os
import argparse
import glob
import logging

from gadget_utils.pipeline_utils import fit_array_to_size
from image_utils.classifier_utils import get_relative_paths


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def fit_image_to_size(input_path, output_path, out_wh, recursive):
    """
    pad/crop the image to the size [W,H] and modify its annotations accordingly
    arguments:
        input_path(str): the input image path
        out_wh(list): the width and height of the output image
    """
    if not os.path.isdir(input_path):
        raise Exception(f'input image folder does not exist: {input_path}')
    
    W,H = out_wh
    img_paths = get_relative_paths(input_path, recursive)
    for path in img_paths:
        im = cv2.imread(os.path.join(input_path,path))
        h,w = im.shape[:2]
        im_name = os.path.basename(path)

        #pad image and save it
        im_out, pad_L, pad_R, pad_T, pad_B = fit_array_to_size(im,W,H)
        if pad_L<0 or pad_R<0 or pad_T<0 or pad_B<0:
            logger.warning(f'{im_name} with the size of {w}x{h} is larger than the output size {W}x{H}. Pixels are copped.')

        #create output fname
        out_name = os.path.splitext(im_name)[0] + f'_pad_{W}x{H}.png'
        outp = os.path.join(output_path, os.path.dirname(path))
        if not os.path.isdir(outp):
            os.makedirs(outp)
        output_file=os.path.join(outp, out_name)
        cv2.imwrite(output_file,im_out)
        # logger.info(f'Output file: {out_name}')
    


if __name__=="__main__":
    ap=argparse.ArgumentParser(description='Pad or crop images to output size.')
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to the images')
    ap.add_argument('--path_out', '-o', required=True, help='the output path')
    ap.add_argument('--wh', required=True, help='the output image size [w,h], w and h are separated by a comma')
    ap.add_argument('--recursive', action='store_true', help='process images recursively')
    args=vars(ap.parse_args())

    path_imgs = args['path_imgs']
    out_path = args['path_out']
    out_wh = list(map(int,args['wh'].split(',')))
    recursive = args['recursive']

    logger.info(f'output image size: {out_wh}')
    assert len(out_wh)==2, 'the output image size must be two ints'

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    fit_image_to_size(path_imgs, out_path, out_wh, recursive)
    logger.info('Done pading or cropping images.')
