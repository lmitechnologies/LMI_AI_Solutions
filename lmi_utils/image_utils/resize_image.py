#built-in packagesprint
import os
import glob
import logging

#3rd party packages
import cv2

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def resize_images(path_imgs, output_imsize, path_out):
    """
    resize images, while keep the aspect ratio.
    if the aspect ratio changes, it will generate an error.
    Arguments:
        path_imgs(str): the image folder
        output_imsize(list): a list of output image size [w,h]
        path_out(str): the output folder
    Return:

    """
    file_list = glob.glob(os.path.join(path_imgs, '*.png'))
    W,H = output_imsize
    ratio_out = W/H
    for file in file_list:
        im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        im_name = os.path.basename(file)
        h,w = im.shape[:2]
        logger.info(f'input file: {im_name} with size of [{w},{h}]')
        
        ratio_in = w/h
        if ratio_in != ratio_out:
            logger.warning(f'file: {im_name}, asepect ratio changed from {ratio_in} to {ratio_out}')
        
        out_name = im_name
        
        im2 = cv2.resize(im, dsize=tuple(output_imsize))

        logger.info(f'writting to {out_name}\n')
        cv2.imwrite(os.path.join(path_out,out_name), im2)
    return



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', required=True, help='the path to images')
    ap.add_argument('--out_imsz', required=True, help='the output image size [w,h], w and h are separated by a comma')
    ap.add_argument('--path_out', required=True, help='the path to resized images')
    args = vars(ap.parse_args())

    output_imsize = list(map(int,args['out_imsz'].split(',')))
    assert len(output_imsize)==2, 'the output image size must be two ints'
    logger.info(f'output image size: {output_imsize}')
    
    path_imgs = args['path_imgs']
    path_out = args['path_out']

    # create output path
    assert path_imgs!=path_out, 'input and output path must be different'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    #resize images with annotation csv file
    resize_images(path_imgs, output_imsize, path_out)
