import os
import json
from dataset_utils.representations import Dataset
import logging
import yaml
import argparse
import random
import glob
import cv2
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_train_json', default='labels.json', help='[optional] the path of a json file for train')
    ap.add_argument('--path_val_json', default='labels.json', help='[optional] the path of a json file for val')
    ap.add_argument('--path_out', '-o', required=True, help='the output path for dataset')
    ap.add_argument('--path_train_imgs', '-ti', required=True, help='the output path for train images')
    ap.add_argument('--path_val_imgs', '-vi', required=False, help='the output path for val images')
    ap.add_argument('--path_dataset_yaml', '-cy', required=False, help='[optional] the path of a yolo dataset yaml file')
    ap.add_argument('--target_classes',default='all', help='[optional] the comma separated target classes, default=all')
    ap.add_argument('--obb', action='store_true', help='support for oriented bounding box support')
    ap.add_argument('--seg', action='store_true', help='convert label formats: mask-to-bbox if "--convert" is enabled, otherwise bbox-to-mask')
    ap.add_argument('--convert', action='store_true', help='convert label formats: bbox-to-mask if "--seg" is enabled, otherwise mask-to-bbox')
    ap.add_argument('--bg', action='store_true', help='save images with no labels, where yolo models treat them as background')
    ap.add_argument('--merge_box', action='store_true', help='merge multiple instances of same class boxes into one. Brush labels only!')
    args = vars(ap.parse_args())
    return args

def write_txts(fname_to_rows, path_txts, fnames,file_id_map):
    """
    write to the yolo format txts
    Arugments:
        fname_to_rows(dict): a map <filename, a list of rows>, where each row is [class_ID, x, y, w, h]
        path_txts: the output folder contains txt files
    """
    os.makedirs(path_txts, exist_ok=True)
    
    for fname in fname_to_rows:
        if fnames is not None and fname not in fnames:
            continue
        name = os.path.basename(fname)
        out_name = name
        ext = name.split('.')[-1]
        file_id = file_id_map[fname]
        if f'id{file_id}_' not in out_name:
            out_name = f'id{file_id}_{out_name}'
        txt_file = os.path.join(path_txts, out_name)
        txt_file = txt_file.replace(f'.{ext}', '.txt')
    
        with open(txt_file, 'w') as f:
            for shape in fname_to_rows[fname]:
                class_id = shape[0]
                xyxy = shape[1:]
                row2 = f'{class_id} '
                for pt in xyxy:
                    row2 += f'{pt:.4f} '
                row2 += '\n'
                f.write(row2)
    logger.debug(f' wrote {len(fnames) if fnames is not None else len(fname_to_rows)} txt files to {path_txts}')

def update_file_dimensions(dataset, path_imgs):
    """
    update the file dimensions
    """
    for file in dataset.files:
        if os.path.isfile(os.path.join(path_imgs, file.path)) is False:
            raise Exception(f'File not found: {file.path}')
        img = cv2.imread(os.path.join(path_imgs, file.path))
        if img is None:
            raise Exception(f'cannot read image: {file.path}')
        h,w = img.shape[:2]
        file.height = h
        file.width = w
    return dataset

def copy_images_in_folder(path_img, path_out, fnames, file_id_map):
    """
    copy the images from one folder to another
    Arguments:
        path_img(str): the path of original image folder
        path_out(str): the path of output folder
    """
    os.makedirs(path_out, exist_ok=True)
    if fnames is None:
        raise Exception('fnames cannot be None')
    for fname in fnames:
        logger.debug(f'copying {fname}')
        out_name = os.path.basename(fname)
        file_id = file_id_map[fname]
        if f'id{file_id}_' not in out_name:
            out_name = f'id{file_id}_{out_name}'
        shutil.copy(os.path.join(path_img, fname), os.path.join(path_out, out_name))
    
    logger.debug(f'copied {len(fnames)} images to {path_out}')

def convert_to_yolo(args):
    path_train_imgs = args['path_train_imgs']
    path_val_imgs = args['path_val_imgs'] if args.get('path_val_imgs') else path_train_imgs
    path_train_json = args['path_train_json'] if args['path_train_json']!='labels.json' else os.path.join(path_train_imgs, args['path_train_json'])
    path_val_json = args['path_val_json'] if args['path_val_json']!='labels.json' else os.path.join(path_val_imgs, args['path_val_json'])
    path_out = args['path_out']
    merge_box = args.get('merge_box', False)
    bbox_to_mask = True if args.get('convert', False) and args.get('seg', False) else False
    mask_to_od = True if args.get('convert', False) and not args.get('seg', False) else False
    target_classes = args['target_classes'].split(',')
    use_obb = args.get('obb', False)
    path_class_yaml = args.get('path_dataset_yaml', None)
    
    # check if the dataset path exists
    if not os.path.exists(path_train_imgs):
        raise Exception('The training image path does not exist')
    if not os.path.exists(path_val_imgs) :
        raise Exception('The validation image path does not exist')
    if not os.path.exists(path_train_json):
        raise Exception('The json file does not exist')
    if not os.path.exists(path_val_json):
        raise Exception('The json file does not exist')
    if path_class_yaml is not None and not os.path.exists(path_class_yaml):
        raise Exception('The yaml file for the class map does not exist')
    
    # check if using training data for validation based on path
    use_train_for_val = False
    if path_train_imgs == path_val_imgs and path_train_json == path_val_json:
        use_train_for_val = True
        logger.warning('The training and validation image paths are the same, will use train images for validation')
        
    # load the class map
    class_map = None
    if path_class_yaml is not None:
        with open(path_class_yaml, 'r') as f:
            class_map = yaml.load(f, Loader=yaml.SafeLoader)['names']
            logger.info(f'class map: {class_map}')
         
    # load the json file
    train_dataset = Dataset.load(path_train_json)
    train_dataset = update_file_dimensions(train_dataset, path_train_imgs)
    train_file_id_map = {f.path:f.id for f in train_dataset.files}
    
    train_yolo_dataset = train_dataset.to_yolo(
        merge_boxes=merge_box,
        to_segmentation=bbox_to_mask,
        to_object_detection=mask_to_od,
        target_classes=target_classes,
        use_obb=use_obb,
        class_map=class_map
    )
    if use_train_for_val is False:
        val_dataset = Dataset.load(path_val_json)
        val_dataset = update_file_dimensions(val_dataset, path_val_imgs)
        val_file_id_map = {f.path:f.id for f in val_dataset.files}
        val_yolo_dataset = val_dataset.to_yolo(
            merge_boxes=merge_box,
            to_segmentation=bbox_to_mask,
            to_object_detection=mask_to_od,
            target_classes=target_classes,
            use_obb=use_obb
        )
    else:
        val_yolo_dataset = train_yolo_dataset
        val_file_id_map = train_file_id_map
    
    
    # path for labels files
    path_txts_train = os.path.join(path_out, 'labels/train')
    path_txts_val = os.path.join(path_out, 'labels/val')
    path_out_imgs_train = os.path.join(path_out, 'images/train')
    path_out_imgs_val = os.path.join(path_out, 'images/val')
    
    
    train_files = list(train_yolo_dataset['image_labels'].keys())
    val_files = list(val_yolo_dataset['image_labels'].keys())
    
    
    logger.debug(f'train files: {len(train_files)}')
    logger.debug(f'val files: {len(val_files)}')
    

    write_txts(train_yolo_dataset['image_labels'], path_txts=path_txts_train, fnames=train_files, file_id_map=train_file_id_map)
    if not os.path.exists(path_out_imgs_train):
        os.makedirs(path_out_imgs_train)
    if len(val_files)>0 and use_train_for_val is False:
        write_txts(val_yolo_dataset['image_labels'], path_txts=path_txts_val,fnames=val_files, file_id_map=val_file_id_map)
        if not os.path.exists(path_out_imgs_val):
            os.makedirs(path_out_imgs_val)
    

    
        # write class map yolo yaml
    with open(os.path.join(args['path_out'], 'dataset.yaml'), 'w') as f:
        dt = {
            'path': path_out,
            'train': 'images/train',
            'val': 'images/train' if len(val_files)==0 or use_train_for_val is True else 'images/val',
            'test': None,
        }
        if train_yolo_dataset['n_kpts']:
            dt['kpt_shape'] = [train_yolo_dataset['n_kpts'],2]
        dt['names'] = {int(v):k for k,v in train_yolo_dataset['class_map'].items() }
        yaml.dump(dt, f, sort_keys=False)
    
    fname = os.path.join(args['path_out'], 'class_map.json')
    
    with open(fname, 'w') as outfile:
        json.dump({k:int(v)
            for k,v in train_yolo_dataset['class_map'].items()}, outfile)
    
    train_fnames = []
    val_fnames = []
    
    if not args.get('bg'):
        train_fnames = [k for k in train_files if len(train_yolo_dataset['image_labels'][k])>0]
        if len(val_files)>0 and use_train_for_val is False:
            val_fnames = [k for k in val_files if len(val_yolo_dataset['image_labels'][k])>0]
    
    else:
        train_fnames = [k for k in train_files]
        if len(val_files)>0 and use_train_for_val is False:
            val_fnames = [k for k in val_files]
    
    copy_images_in_folder(path_img=path_train_imgs, path_out=path_out_imgs_train, fnames=train_fnames, file_id_map=train_file_id_map)
    
    if len(val_fnames)>0 and use_train_for_val is False:
        copy_images_in_folder(path_img=path_val_imgs, path_out=path_out_imgs_val, fnames=val_fnames, file_id_map=val_file_id_map)
    

def main():
    args = get_args()
    convert_to_yolo(args)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    