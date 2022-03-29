import cv2
import os
import argparse
import glob

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
        im_out, pad_l, pad_t = fit_array_to_size(im,W,H)

        #create output fname
        l_name = os.path.splitext(im_name)[0].split('_')
        fname = '_'.join(l_name[:-1])
        suffix = l_name[-1]
        if suffix.find(str(h)) != -1 and suffix.find(str(w)) != -1:
            suffix = suffix.replace(str(h),str(H),1)
            suffix = suffix.replace(str(w),str(W),1)
        else:
            suffix = suffix+'_'+str(W)+'x'+str(H)
        fname += '_'+suffix+'.png'
        output_file=os.path.join(output_path,fname)
        print(f'[INFO] Output file: {output_file}') 
        cv2.imwrite(output_file,im_out)
        print()



def fit_array_to_size(im,W,H):
    """
    description:
        pad/crop the image to the size [W,H] with BLACK pixels
        NOTE: the size [W,H] must be greater than the image size
    arguments:
        im(numpy array): the numpy array of a image
        W(int): the target width
        H(int): the target height
    return:
        im(numpy array): the padded image 
    """
    h_im,w_im=im.shape[:2]

    # pad or chop width
    if W >= w_im:
        pad_L=(W-w_im)//2
        pad_R=W-w_im-pad_L
        im=cv2.copyMakeBorder(im,0,0,pad_L,pad_R,cv2.BORDER_CONSTANT,value=BLACK)
    else:
        pad_L = (w_im-W)//2
        pad_R = w_im-W-pad_L
        im = im[:,pad_L:-pad_R,:]
        pad_L *= -1
        pad_R *= -1

    # pad or chop height
    if H >= h_im:
        pad_T=(H-h_im)//2
        pad_B=H-h_im-pad_T
        im=cv2.copyMakeBorder(im,pad_T,pad_B,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    else:
        pad_T = (h_im-H)//2
        pad_B = h_im-H-pad_T
        im = im[pad_T:-pad_B,:,:]
        pad_T *= -1
        pad_B *= -1

    return im, pad_L, pad_T
    


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
