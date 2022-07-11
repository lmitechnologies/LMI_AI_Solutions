#built-in packages
import os
import glob
import shutil

#3rd party packages
import cv2

#import LMI modules
import mask
import rect
import csv_utils


def resize_imgs_with_csv(path_imgs, path_csv, output_imsize):
    """
    resize images and its annotations, while keep the aspect ratio.
    if the aspect ratio changes, it will generate an error.
    Arguments:
        path_imgs(str): the image folder
        path_csv(str): the path of csv annotation file
        output_imsize(list): a list of output image size [w,h]
    Return:
        name_to_im(dict): the map <output image name, im>
        shapes(dict): the map <original image name, a list of shape objects>, where shape objects are annotations
    """
    file_list = glob.glob(os.path.join(path_imgs, '*.png'))
    shapes,_ = csv_utils.load_csv(path_csv, path_img=path_imgs)
    name_to_im = {}
    W,H = output_imsize
    ratio_out = W/H
    for file in file_list:
        im = cv2.imread(file)
        im_name = os.path.basename(file)
        h,w = im.shape[:2]
        
        ratio_in = w/h
        assert ratio_in==ratio_out,f'asepect ratio changed from {ratio_in} to {ratio_out}'
        
        out_name = os.path.splitext(im_name)[0] + f'_resized_{W}x{H}' + '.png'
        
        ratio = W/w
        im2 = cv2.resize(im, dsize=output_imsize)
        name_to_im[out_name] = im2

        for i in range(len(shapes[im_name])):
            if isinstance(shapes[im_name][i], rect.Rect):
                shapes[im_name][i].up_left = [int(v*ratio) for v in shapes[im_name][i].up_left]
                shapes[im_name][i].bottom_right = [int(v*ratio) for v in shapes[im_name][i].bottom_right]
                shapes[im_name][i].im_name = out_name
            elif isinstance(shapes[im_name][i], mask.Mask):
                shapes[im_name][i].X = [int(v*ratio) for v in shapes[im_name][i].X]
                shapes[im_name][i].Y = [int(v*ratio) for v in shapes[im_name][i].Y]
                shapes[im_name][i].im_name = out_name
            else:
                raise Exception("Found unsupported classes. Supported classes are mask and rect")
    return name_to_im, shapes



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', required=True, help='the path to images')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--out_imsz', required=True, help='the output image size [w,h], w and h are separated by a comma')
    ap.add_argument('--path_out', required=True, help='the path to resized images')
    args = vars(ap.parse_args())

    output_imsize = list(map(int,args['out_imsz'].split(',')))
    assert len(output_imsize)==2, 'the output image size must be two ints'
    print(f'output image size: {output_imsize}')
    
    path_imgs = args['path_imgs']
    path_out = args['path_out']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    
    #check if annotation exists
    if not os.path.isfile(path_csv):
        raise Exception(f'cannot find file: {path_csv}')

    #resize images with annotation csv file
    name_to_im,shapes = resize_imgs_with_csv(path_imgs, path_csv, output_imsize)

    # create output path
    assert path_imgs!=path_out, 'input and output path must be different'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    #write images and csv file
    for im_name in name_to_im:
        print(f'writting to {im_name}')
        cv2.imwrite(os.path.join(path_out,im_name), name_to_im[im_name])
    csv_utils.write_to_csv(shapes, os.path.join(path_out,'labels.csv'))
