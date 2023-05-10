import cv2
import os
import argparse
import glob
import numpy as np
import warnings



def split_vstack_image(im, num_split:int):
    """split the images horizontally into segments, vstack these segments
    Args:
        im (np.ndarray): the input image
        num_split (int): the num of segments after the split
    """
    h,w = im.shape[:2]
    if w%num_split:
        warnings.warn(f'the image width of {w} is not divisible by {num_split}')
    w_seg = w//num_split
    im_segs = []
    for j in range(num_split):
        s,e = j*w_seg,(j+1)*w_seg
        im_segs.append(im[:,s:e,:])
    im_out = np.vstack(im_segs)
    return im_out

def split_hstack_image(im, num_split:int):
    """split the images vertically into segments, hstack these segments
    Args:
        im (np.ndarray): the input image
        num_split (int): the num of segments after the split
    """
    h,w = im.shape[:2]
    if h%num_split:
        warnings.warn(f'the image width {h} is not divisible by {num_split}')
    h_seg = h//num_split
    im_segs = []
    for j in range(num_split):
        s,e = j*h_seg,(j+1)*h_seg
        im_segs.append(im[s:e,:,:])
    im_out = np.hstack(im_segs)
    return im_out

def split_vstack_images(input_path:str, output_path:str, num_split:int):
    """
    split the images horizontally into segments, vstack these segments
    arguments:
        input_path(str): the input image path
        output_path(str): the the output path of the image
        num_split(int): the num of segments after the split
    """
    if not os.path.isdir(input_path):
        raise Exception(f'input image folder does not exist: {input_path}')
    
    img_paths = glob.glob(os.path.join(input_path, '*.png'))
    for path in img_paths:
        im = cv2.imread(path)
        h,w = im.shape[:2]
        im_name = os.path.basename(path)
        print(f'Input file: {im_name} with size of [{w},{h}]')

        # split image
        im_out = split_vstack_image(im, num_split)
        nh,nw = im_out.shape[:2]
        print(f'The output image size: [{nw},{nh}]')

        #create output fname
        out_name = os.path.splitext(im_name)[0] + f'_split-{num_split}-vstack' + '.png'
        output_file=os.path.join(output_path, out_name)
        print(f'Output file: {output_file}') 
        cv2.imwrite(output_file,im_out)
        print()
    


if __name__=="__main__":
    ap=argparse.ArgumentParser(description='split images into segments and vstack these segments')
    ap.add_argument('--path_imgs', required=True, help='the path to the images')
    ap.add_argument('--path_out', required=True, help='the output path')
    ap.add_argument('--num_split', required=True, type=int, help='the num of evenly splits horizontally')
    args=vars(ap.parse_args())

    output_path=args['path_out']
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    split_vstack_images(args['path_imgs'], output_path, args['num_split'])