import cv2
import os
import argparse
import numpy as np
import glob

BLACK=(0,0,0)

def pad_image(input_path,output_path,W,H):
    if os.path.isdir(input_path):
        input_files=glob.glob(os.path.join(input_path,'*.png'))
    else:
        input_files=[input_path]        

    for cnt,input_file in enumerate(input_files):
        print(f'Input file: {input_file}')
        im=cv2.imread(input_file)
        im_out=pad_array(im,W,H)
        fname=os.path.split(input_file)[1]
        fname=os.path.splitext(fname)[0]
        fname=fname+'_'+str(W)+'x'+str(H)+'.png'
        output_file=os.path.join(output_path,fname)
        print(f'Output file: {output_file}') 
        cv2.imwrite(output_file,im_out)
    

def pad_array(im,W,H):
    h_im,w_im=im.shape[:2]
    # pad width or crop from center
    if W>w_im:
        pad_L=(W-w_im)//2
        pad_R=W-w_im-pad_L
        im=cv2.copyMakeBorder(im,0,0,pad_L,pad_R,cv2.BORDER_CONSTANT,value=BLACK)
    elif W<w_im:
        rem_L=(w_im-W)//2
        to_R=w_im-W-rem_L
        im=im[:,rem_L:to_R]

    # pad height or crop from center
    if H>h_im:
        pad_T=(H-h_im)//2
        pad_B=H-h_im-pad_T
        im=cv2.copyMakeBorder(im,pad_T,pad_B,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    elif H<h_im:
        rem_T=(h_im-H)//2
        to_B=h_im-H-rem_T
        im=im[rem_T:to_B,:]

    return im
    

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_image_path',required=True)
    ap.add_argument('--output_path',required=True)
    ap.add_argument('--W',type=int,required=True)
    ap.add_argument('--H',type=int,required=True)
    args=vars(ap.parse_args())
    input_path=args['input_image_path']
    output_path=args['output_path']
    W=args['W']
    H=args['H']

    output_dir=os.path.split(output_path)[0]
    if os.path.exists(output_dir):
        pass
    else:
        os.makedirs(output_dir)
    
    pad_image(input_path,output_path,W,H)
