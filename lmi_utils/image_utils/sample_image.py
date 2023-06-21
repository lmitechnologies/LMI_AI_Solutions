#built-in packagesprint
import os
import glob
import logging

#3rd party packages
import shutil
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def sample_images(path_imgs, path_out, num_samples, is_random=False):
    """
    sample images from source to dest folder
    Arguments:
        path_imgs(str): the image folder
        path_out(str): the output folder
        num_samples(int): the number of images to be sampled
        is_random(bool): true if draw samples randomly
    Return:

    """
    file_list = glob.glob(os.path.join(path_imgs, '*.png'))
    if is_random:
        random.shuffle(file_list)
    
    for file in file_list[:num_samples]:
        im_name = os.path.basename(file)
        logger.info(f'select file: {im_name}')
        
        dest_path = os.path.join(path_out, im_name)
        shutil.copy2(file, dest_path)
        
    logger.info(f'writting {num_samples} images to {path_out}\n')
    return



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to images')
    ap.add_argument('--path_out', '-o', required=True, help='the path to sampled images')
    ap.add_argument('--num_samples', '-n', required=True, type=int, help='the number of sample images')
    ap.add_argument('--random', action='store_true', help='randomly sample images')
    args = ap.parse_args()

    # create output path
    assert args.path_imgs!=args.path_out, 'input and output path must be different'
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)

    sample_images(args.path_imgs, args.path_out, args.num_samples, args.random)
