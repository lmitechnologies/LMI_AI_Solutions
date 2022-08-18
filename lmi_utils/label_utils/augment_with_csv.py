#built-in packages
import collections
import os
import glob
import random
import copy

#3rd party packages
import cv2

#LMI packages
from label_utils import mask, rect, csv_utils


def augment_imgs_with_csv(path_imgs:str, path_csv:str, path_out:str, pixel_mul:float, size_mul:int):
    """
    augment images and its annotations by multiplying a constant to each pixel
    Arguments:
        path_imgs(str): the image folder
        path_csv(str): the path of csv annotation file
        pixel_mul(float): a float number indicates the max pixel multiplier
        size_mul(int): a int number indicates the data size multiplier
    Return:
        new_shapes(dict): the map <original image name, a list of shape objects>, where shape objects are annotations
    """
    file_list = glob.glob(os.path.join(path_imgs, '*.png'))
    shapes,_ = csv_utils.load_csv(path_csv, path_img=path_imgs)
    new_shapes = collections.defaultdict(list)
    
    for sz in range(1,1+size_mul):
        for file in file_list:
            im = cv2.imread(file)
            im_name = os.path.basename(file)
            
            p_mul = random.uniform(1,max(1,pixel_mul))
            out_name = os.path.splitext(im_name)[0] + f'_size_mul_{sz}_px_mul_{pixel_mul}' + '.png'
            
            print(f'writting to {out_name}')
            cv2.imwrite(os.path.join(path_out,out_name), im*p_mul)

            for i in range(len(shapes[im_name])):
                if isinstance(shapes[im_name][i], rect.Rect):
                    #shapes[im_name][i].up_left = [int(v) for v in shapes[im_name][i].up_left]
                    #shapes[im_name][i].bottom_right = [int(v) for v in shapes[im_name][i].bottom_right]
                    temp = copy.deepcopy(shapes[im_name][i])
                    temp.im_name = out_name
                    new_shapes[out_name].append(temp)
                elif isinstance(shapes[im_name][i], mask.Mask):
                    #shapes[im_name][i].X = [int(v*ratio) for v in shapes[im_name][i].X]
                    #shapes[im_name][i].Y = [int(v*ratio) for v in shapes[im_name][i].Y]
                    temp = copy.deepcopy(shapes[im_name][i])
                    temp.im_name = out_name
                    new_shapes[out_name].append(temp)
                else:
                    raise Exception("Found unsupported classes. Supported classes are mask and rect")
    return new_shapes



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', required=True, help='the path to images')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--pixel_multiplier', default=2, type=float, help='the max multiplier to each pixel in the image, default = 2')
    ap.add_argument('--data_size_multipler', default=5, type=int, help='the output sample size = data_size_multipler*length_of_input_data_size, default = 5')
    ap.add_argument('--path_out', required=True, help='the path to augmented images')
    args = vars(ap.parse_args())

    pixel_mul = args['pixel_multiplier']    
    size_mul = args['data_size_multipler']
    path_imgs = args['path_imgs']
    path_out = args['path_out']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    
    #check if annotation exists
    if not os.path.isfile(path_csv):
        raise Exception(f'cannot find file: {path_csv}')

    # create output path
    assert path_imgs!=path_out, 'input and output path must be different'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
        
    # augment images
    shapes = augment_imgs_with_csv(path_imgs,path_csv,path_out,pixel_mul,size_mul)
        
    #write images and csv file  
    csv_utils.write_to_csv(shapes, os.path.join(path_out,'labels.csv'))
