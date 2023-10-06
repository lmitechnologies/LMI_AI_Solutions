from logging import warning
import cv2
import os
import argparse
import numpy as np
import glob

#LMI packages
from label_utils import rect, mask
from label_utils.csv_utils import load_csv, write_to_csv
from gadget_utils.pipeline_utils import fit_array_to_size


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
        im_name = os.path.basename(path)
        if im_name not in fname_to_shape:
            cnt += 1
            continue
        
        im = cv2.imread(path)
        if im is None:
            warning(f'The file: {im_name} might be corrupted, skip!')
            continue
        h,w = im.shape[:2]
        print(f'[INFO] Input file: {im_name} with size of [{w},{h}]')
        
        #pad image
        im_out,pad_l,_,pad_t,_ = fit_array_to_size(im,W,H)

        #create output fname and save it
        out_name = os.path.splitext(im_name)[0] + f'_padded_{W}x{H}' + '.png'
        output_file=os.path.join(output_path, out_name)
        print(f'[INFO] Output file: {output_file}')
        cv2.imwrite(output_file,im_out)

        #pad shapes
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
            shape.im_name = out_name
        output_shapes[out_name] = shapes
    print(f'[INFO] found {cnt} images with no labels')
    print(f'[INFO] found {cnt_warnings} images with labels that is either removed entirely, or chopped to fit the new size')
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
        is_del = 0
        if isinstance(shape, rect.Rect):
            box = np.array(shape.up_left+shape.bottom_right)
            new_box = np.clip(box, a_min=0, a_max=[W,H,W,H])
            shapes[i].up_left = new_box[:2].tolist()
            shapes[i].bottom_right = new_box[2:].tolist()
            if np.all(new_box==0) or new_box[0]==new_box[2] or new_box[1]==new_box[3]:
                is_del = 1
                warning(f'bbox {box} is outside of the size [{W},{H}]')
            elif (np.any(new_box==W) and np.all(box!=W)) or (np.any(new_box==H) and np.all(box!=H)) \
                    or (np.any(new_box==0) and np.all(box!=0)):
                warning(f'bbox {box} is chopped to fit the size [{W}, {H}]')
                is_warning = True
        elif isinstance(shape, mask.Mask):
            X,Y = np.array(shape.X), np.array(shape.Y)
            new_X = np.clip(X, a_min=0, a_max=W)
            new_Y = np.clip(Y, a_min=0, a_max=H)
            shapes[i].X,shapes[i].Y = new_X.tolist(),new_Y.tolist()
            if np.all(new_X==W) or np.all(new_Y==H) or np.all(new_X==0) or np.all(new_Y==0):
                is_del = 1
                warning(f'polygon {[(x,y) for x,y in zip(new_X,new_Y)]} is outside of the size [{W},{H}]')
            elif (np.any(new_X==W) and np.all(X!=W)) or (np.any(new_Y==H) and np.all(Y!=H)) \
                or (np.any(new_X==0) and np.all(X!=0)) or (np.any(new_Y==0) and np.all(Y!=0)):
                warning(f'polygon {[(x,y) for x,y in zip(new_X,new_Y)]} is chopped to fit the size [{W}, {H}]')
                is_warning = True
        if is_del:
            is_warning = True
            to_del.append(i)
            
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
        if isinstance(shape, rect.Rect):
            shape.up_left[0] += pad_l
            shape.up_left[1] += pad_t
            shape.bottom_right[0] += pad_l
            shape.bottom_right[1] += pad_t
        elif isinstance(shape, mask.Mask):
            shape.X = [v+pad_l for v in shape.X]
            shape.Y = [v+pad_t for v in shape.Y]
    return shapes
    


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
