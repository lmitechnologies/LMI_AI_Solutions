import cv2
import os
import argparse
import glob
import numpy as np

BLACK=(0,0,0)

def flip_images(input_path:str, output_path:str, flip:str):
    """
    flip images either vertically or horizontally
    arguments:
        input_path(str): the input image path
        output_path(str): the the output path of the image
        flip(str): the flip direction
    """
    if not os.path.isdir(input_path):
        raise Exception(f'input image folder does not exist: {input_path}')
    
    img_paths = glob.glob(os.path.join(input_path, '*.png'))
    for path in img_paths:
        im = cv2.imread(path)
        h,w = im.shape[:2]
        im_name = os.path.basename(path)
        print(f'Input file: {im_name} with size of [{w},{h}]')

        #flip image
        im_out = np.flip(im, axis=1 if flip=='lr' else 0)

        #create output fname
        out_name = os.path.splitext(im_name)[0] + f'_flip{flip}' + '.png'
        output_file=os.path.join(output_path, out_name)
        print(f'Output file: {output_file}') 
        cv2.imwrite(output_file,im_out)
        print()
    


if __name__=="__main__":
    ap=argparse.ArgumentParser(description='flip images')
    ap.add_argument('--path_imgs', required=True, help='the path to the images')
    ap.add_argument('--path_out', required=True, help='the output path')
    ap.add_argument('--flip', required=True, choices=['ud','lr'], help='flip image horizontally or vertically')
    args=vars(ap.parse_args())

    output_path=args['path_out']
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    flip_images(args['path_imgs'], output_path, args['flip'])