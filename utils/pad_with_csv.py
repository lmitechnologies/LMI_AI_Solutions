import cv2
import os
import argparse
import numpy as np
import shutil
import glob
from csv_utils import load_csv, write_to_csv

BLACK=(0,0,0)

def pad_image_with_csv(input_path,csv_path,output_path,W,H):
    if not os.path.isdir(input_path):
        raise Exception(f'input image folder does not exist: {input_path}')
    fname_to_shape, class_map = load_csv(csv_path, input_path)

    output_shapes = {}
    for im_name in fname_to_shape:
        print(f'Input file: {im_name}')
        
        input_file = os.path.join(input_path, im_name)
        im = cv2.imread(input_file)

        #pad image and shapes
        im_out, pad_l, pad_t = pad_array(im,W,H)
        shapes = fname_to_shape[im_name]
        print('before: ',[(shape.up_left, shape.bottom_right) for shape in shapes])
        shapes = pad_to_shapes(shapes, pad_l, pad_t)
        print('after: ',[(shape.up_left, shape.bottom_right) for shape in shapes])

        fname=os.path.splitext(im_name)[0]
        fname=fname+'_'+str(W)+'x'+str(H)+'.png'
        output_file=os.path.join(output_path,fname)
        print(f'Output file: {output_file}') 
        cv2.imwrite(output_file,im_out)

        #modify the image name 
        for shape in shapes:
            shape.im_name = fname
        output_shapes[fname] = shapes
    output_csv = os.path.join(output_path, "labels.csv")
    write_to_csv(output_shapes, output_csv)
    
    
def pad_to_shapes(shapes, pad_l, pad_t):
    """
    description:
        add the left and top paddings to the shapes
    arguments:
        shapes(list): a list of Shape objects (Rect or Mask)
        pad_l(int): the left paddings
        pad_t(int): the top paddings 
    """
    for shape in shapes:
        shape.up_left[0] += pad_l
        shape.up_left[1] += pad_t
        shape.bottom_right[0] += pad_l
        shape.bottom_right[1] += pad_t
    return shapes

def pad_array(im,W,H):
    """
    description:
        pad the image to the size [W,H] with BLACK pixels
        NOTE: the size [W,H] must be greater than the image size
    arguments:
        im(numpy array): the numpy array of a image
        W(int): the target width
        H(int): the target height
    return:
        im(numpy array): the padded image 
    """
    h_im,w_im=im.shape[:2]
    assert H>=h_im and W>=w_im, f"the target size: {H,W} must be greater equal to the image size: {h_im,w_im}"

    # pad width
    pad_L=(W-w_im)//2
    pad_R=W-w_im-pad_L
    im=cv2.copyMakeBorder(im,0,0,pad_L,pad_R,cv2.BORDER_CONSTANT,value=BLACK)

    # pad height
    pad_T=(H-h_im)//2
    pad_B=H-h_im-pad_T
    im=cv2.copyMakeBorder(im,pad_T,pad_B,0,0,cv2.BORDER_CONSTANT,value=BLACK)

    return im, pad_L, pad_T
    

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input_image_path',required=True)
    ap.add_argument('-c','--csv_file', required=True)
    ap.add_argument('-o','--output_path',required=True)
    ap.add_argument('--W',type=int,required=True)
    ap.add_argument('--H',type=int,required=True)
    args=vars(ap.parse_args())

    input_path=args['input_image_path']
    csv_path = args['csv_file']
    output_path=args['output_path']
    W=args['W']
    H=args['H']

    assert input_path!=output_path, 'input and output must be different'
    if os.path.exists(output_path):
        print(f'deleting the old stuff in output path: {output_path}')
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    pad_image_with_csv(input_path,csv_path,output_path,W,H)
