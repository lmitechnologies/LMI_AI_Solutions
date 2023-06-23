import cv2
import os
import argparse
import glob

from gadget_utils.pipeline_utils import fit_array_to_size


BLACK=(0,0,0)

def fit_image_to_size(input_path, output_path, output_imsize):
    """
    pad/crop the image to the size [W,H] and modify its annotations accordingly
    arguments:
        input_path(str): the input image path
        output_imsize(list): the width and height of the output image
    """
    if not os.path.isdir(input_path):
        raise Exception(f'input image folder does not exist: {input_path}')
    
    W,H = output_imsize
    img_paths = glob.glob(os.path.join(input_path, '*.png'))
    for path in img_paths:
        im = cv2.imread(path)
        h,w = im.shape[:2]
        im_name = os.path.basename(path)
        print(f'[INFO] Input file: {im_name} with size of [{w},{h}]')

        #pad image and save it
        im_out,_,_,_,_ = fit_array_to_size(im,W,H)

        #create output fname
        out_name = os.path.splitext(im_name)[0] + f'_padded_{W}x{H}' + '.png'
        output_file=os.path.join(output_path, out_name)
        print(f'[INFO] Output file: {output_file}') 
        cv2.imwrite(output_file,im_out)
        print()
    


if __name__=="__main__":
    ap=argparse.ArgumentParser(description='Pad or crop images to output size.')
    ap.add_argument('--path_imgs', required=True, help='the path to the images')
    ap.add_argument('--path_out', required=True, help='the output path')
    ap.add_argument('--out_imsz', required=True, help='the output image size [w,h], w and h are separated by a comma')
    args=vars(ap.parse_args())

    path_imgs = args['path_imgs']
    output_path=args['path_out']
    output_imsize = list(map(int,args['out_imsz'].split(',')))

    print(f'output image size: {output_imsize}')
    assert len(output_imsize)==2, 'the output image size must be two ints'

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    fit_image_to_size(path_imgs, output_path, output_imsize)
