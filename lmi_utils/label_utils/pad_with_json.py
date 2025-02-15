import cv2
import os
import argparse
import numpy as np
import logging

#LMI packages
from dataset_utils.representations import Dataset, File, AnnotationType, Box, Mask, Polygon, Point2d
from gadget_utils.pipeline_utils import fit_array_to_size


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pad_image_with_json(input_path, json_path, output_path, output_imsize, save_bg_images):
    """
    pad/crop the image to the size [W,H] and modify its annotations accordingly
    arguments:
        input_path(str): the input image path
        csv_path(str): the path to the csv annotation file
        output_imsize(list): the width and height of the output image
    """
    if not os.path.isdir(input_path):
        raise Exception(f'input image folder does not exist: {input_path}')

    W,H = output_imsize
    cnt_bg = 0
    cnt_warnings = 0
    output_shapes = {}
    
    dataset = Dataset.load(json_path)
    base_prefix = dataset.base_path
    
    for f in dataset.files:
        file_path = f.relative_path(base_prefix)
        p = os.path.join(input_path, file_path)
        im_name = os.path.basename(file_path)
        im = cv2.imread(p)
        h,w = im.shape[:2]
        
        # found bg image
        if not f.has_annotations:
            if not save_bg_images:
                continue
            cnt_bg += 1
            logger.info(f'{im_name}: wh of [{w},{h}] has no labels')
        else:
            logger.info(f'{im_name}: wh of [{w},{h}]')
        
        # pad image
        im_out,pad_l,_,pad_t,_ = fit_array_to_size(im,W,H)

        #create output fname and save it
        out_name = os.path.splitext(im_name)[0] + f'_pad_{W}x{H}' + '.png'
        output_file=os.path.join(output_path, out_name)
        logger.info(f'write to: {output_file}')
        cv2.imwrite(output_file,im_out)

        #pad shapes
        f.annotations = fit_shapes_to_size(f.annotations,pad_l,pad_t, pad_h=H, pad_w=W, orig_h=h, orig_w=w)
            
        delete_ids,is_warning = clip_shapes(f.annotations, W, H)
        f.annotations = [shape for shape in f.annotations if shape.id not in delete_ids]

        # update the dataset annotaions
        
        
        if is_warning:
            cnt_warnings += 1

        height = im_out.shape[0]
        width = im_out.shape[1]
        f.update_file(File(path=output_file, width=width, height=height, id=f.id))
    if cnt_bg:
        logger.info(f'found {cnt_bg} images with no labels. These images will be used as background training data for YOLO.')
    if cnt_warnings:
        logger.warning(f'found {cnt_warnings} images with labels that is either removed entirely, or chopped to fit the new size')
    output_json = os.path.join(output_path, "labels.json")
    dataset.save(output_json)
    

def clip_shapes(shapes, W, H):
    """
    description:
        clip the shapes so that they are fit in the target size [W,H]
    """
    is_warning = False
    shapes = np.array(shapes)
    delete_ids = []
    for i,shape in enumerate(shapes):
        is_del = 0
        if shape.type == AnnotationType.BOX:
            
            box = shape.value.to_numpy()[:-1]
            new_box = np.clip(box, a_min=0, a_max=[W,H,W,H])
            
            if np.all(new_box==0) or new_box[0]==new_box[2] or new_box[1]==new_box[3]:
                is_del = 1
                delete_ids.append(shape.id)
                logger.warning(f'bbox {box} is outside of the size [{W},{H}]')
            elif (np.any(new_box==W) and np.all(box!=W)) or (np.any(new_box==H) and np.all(box!=H)) \
                    or (np.any(new_box==0) and np.all(box!=0)):
                logger.warning(f'bbox {box} is chopped to fit the size [{W}, {H}]')
                is_warning = True
                
                shape.value = Box(*new_box)
        
        elif shape.type == AnnotationType.MASK or shape.type == AnnotationType.POLYGON:
            X,Y = shape.value.coords(w=W,h=H)
            new_X = np.clip(X, a_min=0, a_max=W)
            new_Y = np.clip(Y, a_min=0, a_max=H)
            if np.all(new_X==W) or np.all(new_Y==H) or np.all(new_X==0) or np.all(new_Y==0):
                is_del = 1
                delete_ids.append(shape.id)
                # logger.warning(f'polygon {[(x,y) for x,y in zip(new_X,new_Y)]} is outside of the size [{W},{H}]')
                logger.warning(f'polygon {shape.id} is outside of the size [{W},{H}]')

            elif (np.any(new_X==W) and np.all(X!=W)) or (np.any(new_Y==H) and np.all(Y!=H)) \
                or (np.any(new_X==0) and np.all(X!=0)) or (np.any(new_Y==0) and np.all(Y!=0)):
                logger.warning(f'polygon {shape.id} is chopped to fit the size [{W}, {H}]')
                is_warning = True
                if shape.type == AnnotationType.POLYGON:
                    shape.value = Polygon(points=np.array(list(zip(new_X,new_Y))).astype(int).tolist())
                else:
                    img = np.zeros((H,W),dtype=np.uint8)
                    cv2.fillPoly(img, [np.array(list(zip(new_X,new_Y))).astype(int)], 1)
                    shape.value = Mask(mask=img)
                
        elif shape.type == AnnotationType.KEYPOINT:
            x,y = shape.value.coords()
            if x<0 or x>W or y<0 or y>H:
                is_del = 1
                logger.warning(f'keypoint ({x},{y}) is outside of the size [{W},{H}]')
                delete_ids.append(shape.id)
            else:
                shape.value = Point2d(x=x, y=y)
                
        if is_del:
            is_warning = True
            
    return delete_ids, is_warning
    

def fit_shapes_to_size(shapes, pad_l, pad_t, pad_h, pad_w,orig_h,orig_w):
    """
    description:
        add the left and top paddings to the shapes, modify in-place
    arguments:
        shapes(list): a list of Shape objects (Rect or Mask)
        pad_l(int): the left paddings
        pad_t(int): the top paddings 
    """
    
    for annot in shapes:
        annot.value = annot.value.pad(pad_h=pad_h, pad_w=pad_w, pl=pad_l, pt=pad_t, h=orig_h, w=orig_w)   
    return shapes 
    


if __name__=="__main__":
    ap = argparse.ArgumentParser(description='Pad or crop images with csv to output size.')
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to the images')
    ap.add_argument('--path_csv', default='labels.json', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.json" in path_imgs')
    ap.add_argument('--path_out','-o', required=True, help='the output path')
    ap.add_argument('--wh', required=True, help='the output image size [w,h], w and h are separated by a comma')
    ap.add_argument('--bg', action='store_true', help='save background images with no labels')
    ap.add_argument('--append', action='store_true', help='append to the existing output csv file')
    ap.add_argument('--recursive', action='store_true', help='search images recursively')
    args = vars(ap.parse_args())

    path_imgs = args['path_imgs']
    path_csv = args['path_csv'] if args['path_csv']!='labels.json' else os.path.join(path_imgs, args['path_csv'])
    if not os.path.isfile(path_csv):
        raise Exception(f'Not found file: {path_csv}. Please create an empty csv file, if there are no labels.')
    output_path=args['path_out']
    output_imsize = list(map(int,args['wh'].split(',')))

    logger.info(f'output image size: {output_imsize}')
    assert len(output_imsize)==2, 'the output image size must be two ints'
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    pad_image_with_json(path_imgs, path_csv, output_path, output_imsize, args['bg'])
    
    
