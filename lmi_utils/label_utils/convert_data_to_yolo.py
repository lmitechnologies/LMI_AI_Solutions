"""
convert the data (images with a csv annotation file) to yolo file format
"""

#built-in packages
import cv2
import shutil
import glob
import json
import os
import yaml

#LMI packages
from label_utils.csv_utils import load_csv
from label_utils.mask import Mask
from label_utils.rect import Rect
from label_utils.bbox_utils import rotate


def convert_to_txt(fname_to_shapes, class_to_id, target_classes:list, is_seg=False, is_convert=False, obb=False):
    """
    convert the map <fname, list of shape objects> to YOLO format
    Arguments:
        fname_to_shapes(dict): the map <fname, list of shape objects>
        class_to_id(dict): the map <class_name, class_ID>
        is_seg: whether to load as segmentation labels
        is_convert: whether to perform conversion: the bbox-to-mask if is_seg is true, or mask-to-bbox if is_seg is false
    Return:
        fname_to_rows: the map <file name, list of row>, where each row is [class_ID, x, y, w, h]
    """
    fname_to_rows = {}
    for fname in fname_to_shapes:
        shapes = fname_to_shapes[fname]
        rows = []
        for shape in shapes:
            #get class ID
            if shape.category not in target_classes:
                continue
            class_id = class_to_id[shape.category]
            #get the image H,W
            I = cv2.imread(shape.fullpath)
            H,W = I.shape[:2]
            if isinstance(shape, Rect):
                #get bbox w,h
                x0,y0 = shape.up_left
                x2,y2 = shape.bottom_right
                if not is_seg and not obb:
                    w = x2 - x0
                    h = y2 - y0
                    #get bbox center
                    cx,cy = (x0+x2)/2, (y0+y2)/2
                    # normalize to [0-1]
                    row = [class_id, cx/W, cy/H, w/W, h/H]
                    rows.append(row)
                elif obb:
                    # bbox-to-obb
                    angle = shape.angle
                    w = x2 - x0
                    h = y2 - y0
                    rotated_coords = rotate(x0, y0, w, h, angle, rot_center='up_left', unit='degree')
                    xyxy = [class_id]
                    for pt in rotated_coords:
                        xyxy += [pt[0]/W, pt[1]/H]
                    row = xyxy
                    rows.append(row)
                elif is_convert:
                    # bbox-to-mask
                    x1,y1 = x2,y0
                    x3,y3 = x0,y2
                    xyxy = [x0/W, y0/H, x1/W, y1/H, x2/W, y2/H, x3/W, y3/H]
                    row = [class_id]+xyxy
                    rows.append(row)
            elif isinstance(shape, Mask):
                if is_seg:
                    xyxy = []
                    for x,y in zip(shape.X,shape.Y):
                        xyxy += [x/W,y/H]
                    row = [class_id]+xyxy
                    rows.append(row)
                elif is_convert:
                    # mask-to-bbox
                    x0,y0 = min(shape.X),min(shape.Y)
                    x2,y2 = max(shape.X),max(shape.Y)
                    w = x2 - x0
                    h = y2 - y0
                    cx,cy = (x0+x2)/2, (y0+y2)/2
                    row = [class_id, cx/W, cy/H, w/W, h/H]
                    rows.append(row)
        txt_name = fname.replace('.png','.txt').replace('.jpg','.txt')
        fname_to_rows[txt_name] = rows
    return fname_to_rows


def write_txts(fname_to_rows, path_txts):
    """
    write to the yolo format txts
    Arugments:
        fname_to_rows(dict): a map <filename, a list of rows>, where each row is [class_ID, x, y, w, h]
        path_txts: the output folder contains txt files
    """
    os.makedirs(path_txts, exist_ok=True)
    for fname in fname_to_rows:
        txt_file = os.path.join(path_txts, fname)
        print('writting to {}'.format(fname))
        with open(txt_file, 'w') as f:
            for shape in fname_to_rows[fname]:
                class_id = shape[0]
                xyxy = shape[1:]
                row2 = f'{class_id} '
                for pt in xyxy:
                    row2 += f'{pt:.4f} '
                row2 += '\n'
                f.write(row2)
    print(f'[INFO] wrote {len(fname_to_rows)} txt files to {path_txts}')
    

def copy_images_in_folder(path_img, path_out, fnames=None):
    """
    copy the images from one folder to another
    Arguments:
        path_img(str): the path of original image folder
        path_out(str): the path of output folder
    """
    os.makedirs(path_out, exist_ok=True)
    if not fnames:
        l = glob.glob(os.path.join(path_img, '*.png')) + glob.glob(os.path.join(path_img, '*.jpg'))
    else:
        l = [f"{path_img}/{fname}" for fname in fnames]
    for f in l:
        shutil.copy(f, path_out)


if __name__ =='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', '-i', required=True, help='the path of a image folder')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--class_map_json', help='[optinal] the class map json file')
    ap.add_argument('--path_out', '-o', required=True, help='the output path')
    ap.add_argument('--target_classes',default='all', help='[optional] the comma separated target classes, default=all')
    ap.add_argument('--seg', action='store_true', help='load labels in segmentation format')
    ap.add_argument('--convert', action='store_true', help='convert label formats: bbox-to-mask if "--seg" is enabled, otherwise mask-to-bbox')
    ap.add_argument('--bg_images', action='store_true', help='save images with no labels, where yolo models treat them as background')
    ap.add_argument('--obb', action='store_true', help='support for oriented bounding box support')
    args = vars(ap.parse_args())

    path_imgs = args['path_imgs']
    class_map_file = args['class_map_json']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    target_classes = args['target_classes'].split(',')
    is_seg = args['seg']
    is_convert = args['convert']

    if not os.path.isfile(path_csv):
        raise Exception(f'Not found file: {path_csv}')

    class_map = None
    if class_map_file:
        with open(class_map_file) as f:
            class_map = json.load(f)
        fname_to_shapes,class_to_id = load_csv(path_csv, path_imgs, class_map)
    else:
        fname_to_shapes,class_to_id = load_csv(path_csv, path_imgs, zero_index=True)
    
    if len(target_classes)==1 and target_classes[0]=='all':
        target_classes = [cls for cls in class_to_id]
    for cls in target_classes:
        if cls not in class_to_id:
            raise Exception(f'Not found target class: {cls}')
    print(f'target_classes: {target_classes}')
    
    # modify the map class to id
    keys = class_to_id.keys()
    del_keys = set(keys)-set(target_classes)
    for key in del_keys:
        del class_to_id[key]
    # re-assign id 
    id = 0
    for cls in class_to_id:
        class_to_id[cls] = id
        id += 1
    
    fname_to_rows = convert_to_txt(fname_to_shapes, class_to_id, target_classes, is_seg, is_convert, args['obb'])

    #generate output yolo dataset
    if not os.path.isdir(args['path_out']):
        os.makedirs(args['path_out'])

    #write class map file
    fname = os.path.join(args['path_out'], 'class_map.json')
    with open(fname, 'w') as outfile:
        json.dump(class_to_id, outfile)
        
    # write class map yolo yaml
    with open(os.path.join(args['path_out'], 'dataset.yaml'), 'w') as f:
        dt = {
            'path': '/app/data',
            'train': 'images',
            'val': 'images',
            'test': None,
            'names':{v:k for k,v in class_to_id.items()},
        }
        yaml.dump(dt, f, sort_keys=False)

    #write labels/annotations
    path_txts = os.path.join(args['path_out'], 'labels')
    write_txts(fname_to_rows, path_txts)

    #write images
    path_img_out = os.path.join(args['path_out'], 'images')
    if args['bg_images']:
        print('save background images')
        copy_images_in_folder(path_imgs, path_img_out)
    else:
        print('skip background images')
        copy_images_in_folder(path_imgs, path_img_out, fname_to_shapes.keys())
