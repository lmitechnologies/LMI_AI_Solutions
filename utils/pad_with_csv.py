import cv2
import os
import argparse
import numpy as np
import glob
from csv_utils import load_csv, write_to_csv

BLACK=(0,0,0)

def pad_image_with_csv(input_path, csv_path, output_path, output_imsize):
    """
    pad/crop the image to the size [W,H] and modify its annotations accordingly
    arguments:
        input_path(str): the input image path
        csv_path(str): the path to the csv annotation file
        output_imsize(list): the width and height of the output image
    """
    if not os.path.isdir(input_path):
        raise Exception(f'input image folder does not exist: {input_path}')
    fname_to_shape, _ = load_csv(csv_path, input_path)

    W,H = output_imsize
    cnt = 0
    cnt_warnings = 0
    output_shapes = {}
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

        #pad shapes
        if im_name not in fname_to_shape:
            print('No shapes found, skipping...\n')
            cnt += 1
            continue
        shapes = fname_to_shape[im_name]
        #print('[INFO] before: ',[s.up_left+s.bottom_right for s in shapes])
        shapes = fit_shapes_to_size(shapes, pad_l, pad_t)
        shapes,is_warning = chop_shapes(shapes, W, H)
        #print('[INFO] after: ',[s.up_left+s.bottom_right for s in shapes])
        if is_warning:
            cnt_warnings += 1
        print()

        #modify the image name 
        for shape in shapes:
            shape.im_name = fname
        output_shapes[fname] = shapes
    print(f'[INFO] found {cnt} images with no bbox')
    print(f'[INFO] found {cnt_warnings} images with bbox that is either removed entirely, or chopped to fit the new size')
    output_csv = os.path.join(output_path, "labels.csv")
    write_to_csv(output_shapes, output_csv)
    

def chop_shapes(shapes, W, H):
    """
    description:
        clip the shapes so that they are fit in the target size [W,H]
    """
    to_del = []
    is_warning = False
    shapes = np.array(shapes)
    for i,shape in enumerate(shapes):
        x1,y1 = shape.up_left
        x2,y2 = shape.bottom_right
        if x1>=W or y1>=H:
            print(f'[*WARNING] bbox [{x1},{y1},{x2},{y2}] is outside of the size [{W},{H}]')
            is_warning = True
            to_del.append(i)
        else:
            nx = min(x2,W)
            ny = min(y2,H)
            shapes[i].bottom_right = [nx,ny]
            if nx==W or ny==H:
                print(f'[*WARNING] bbox [{x1},{y1},{x2},{y2}] is chopped to fit in the size [{W}, {H}]')
                is_warning = True
    new_shapes = np.delete(shapes,to_del,axis=0)
    return new_shapes.tolist(), is_warning
    

def fit_shapes_to_size(shapes, pad_l, pad_t):
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


def fit_array_to_size(im,W,H):
    """
    description:
        pad/chop the image to the size [W,H] with BLACK pixels
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
    ap=argparse.ArgumentParser(description='Pad or crop images with csv to output size.')
    ap.add_argument('--path_imgs', required=True, help='the path to the images')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--path_out', required=True, help='the output path')
    ap.add_argument('--out_imsz', required=True, help='the output image size [w,h], w and h are separated by a comma')
    args=vars(ap.parse_args())

    path_imgs = args['path_imgs']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    if not os.path.isfile(path_csv):
        raise Exception(f'Not found file: {path_csv}')
    output_path=args['path_out']
    output_imsize = list(map(int,args['out_imsz'].split(',')))

    print(f'output image size: {output_imsize}')
    assert len(output_imsize)==2, 'the output image size must be two ints'
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    pad_image_with_csv(path_imgs, path_csv, output_path, output_imsize)
