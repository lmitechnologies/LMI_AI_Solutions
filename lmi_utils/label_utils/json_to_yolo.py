import os
import json
from dataset_utils.representations import Dataset
from label_utils.csv_to_yolo import copy_images_in_folder
import logging
import yaml
import argparse
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def args():
    ap = argparse.ArgumentParser()
    # ap.add_argument('--path_imgs', '-i', required=True, help='the path of a image folder')
    ap.add_argument('--path_train_json', default='labels.json', help='[optional] the path of a json file for train')
    ap.add_argument('--path_val_json', default='labels.json', help='[optional] the path of a json file for val')
    ap.add_argument('--path_out', '-o', required=True, help='the output path for dataset')
    ap.add_argument('--path_train_imgs', '-ti', required=True, help='the output path for train images')
    ap.add_argument('--path_val_imgs', '-vi', required=True, help='the output path for val images')
    # ap.add_argument('--split_ratio', type=float, default=0.0, help='the ratio to split the dataset, default=0.0, all images will go to train' )
    ap.add_argument('--target_classes',default='all', help='[optional] the comma separated target classes, default=all')
    ap.add_argument('--obb', action='store_true', help='support for oriented bounding box support')
    ap.add_argument('--seg', action='store_true', help='convert label formats: mask-to-bbox if "--convert" is enabled, otherwise bbox-to-mask')
    ap.add_argument('--convert', action='store_true', help='convert label formats: bbox-to-mask if "--seg" is enabled, otherwise mask-to-bbox')
    ap.add_argument('--bg', action='store_true', help='save images with no labels, where yolo models treat them as background')
    ap.add_argument('--merge_box', action='store_true', help='merge multiple instances of same class boxes into one. Brush labels only!')
    args = vars(ap.parse_args())
    return args

def write_txts(fname_to_rows, path_txts, fnames=None):
    """
    write to the yolo format txts
    Arugments:
        fname_to_rows(dict): a map <filename, a list of rows>, where each row is [class_ID, x, y, w, h]
        path_txts: the output folder contains txt files
    """
    os.makedirs(path_txts, exist_ok=True)
    
    for fname in fname_to_rows:
        if fnames is not None and os.path.basename(fname) not in fnames:
            continue
        txt_file = os.path.join(path_txts, fname)
        ext = fname.split('.')[-1]
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
    logger.info(f' wrote {len(fnames) if fnames is not None else len(fname_to_rows)} txt files to {path_txts}')
    

def convert_to_yolo(args):
    path_train_json = args['path_train_json'] if args['path_train_json']!='labels.json' else os.path.join(args.get('path_imgs'), args['path_train_json'])
    path_val_json = args['path_val_json'] if args['path_val_json']!='labels.json' else os.path.join(args.get('path_imgs'), args['path_val_json'])
    path_out = args['path_out']
    # path_imgs = args['path_imgs']
    path_train_imgs = args['path_train_imgs']
    path_val_imgs = args['path_val_imgs']
    merge_box = args.get('merge_box', False)
    bbox_to_mask = True if args.get('convert', False) and args.get('seg', False) else False
    mask_to_od = True if args.get('convert', False) and not args.get('seg', False) else False
    target_classes = args['target_classes'].split(',')
    use_obb = args.get('obb', False)
    
    # check if the dataset path exists
    if not os.path.exists(path_train_imgs):
        raise Exception('The training image path does not exist')
    if not os.path.exists(path_val_imgs):
        raise Exception('The validation image path does not exist')
    if not os.path.exists(path_train_json):
        raise Exception('The json file does not exist')
    if not os.path.exists(path_val_json):
        raise Exception('The json file does not exist')
    
    # check if using training data for validation based on path
    if path_train_imgs == path_val_imgs and path_train_json == path_val_json:
        logger.warning('The training and validation image paths are the same, will use train images for validation')
    
         
    # load the json file
    
    train_dataset = Dataset.load(path_train_json)
    val_dataset = Dataset.load(path_val_json)
    
    train_yolo_dataset = train_dataset.to_yolo(
        merge_boxes=merge_box,
        to_segmentation=bbox_to_mask,
        to_object_detection=mask_to_od,
        target_classes=target_classes,
        use_obb=use_obb
    )
    val_yolo_dataset = val_dataset.to_yolo(
        merge_boxes=merge_box,
        to_segmentation=bbox_to_mask,
        to_object_detection=mask_to_od,
        target_classes=target_classes,
        use_obb=use_obb
    )
    
    
    # print(yolo_dataset)
    
    # path for labels files
    path_txts_train = os.path.join(path_out, 'labels/train')
    path_txts_val = os.path.join(path_out, 'labels/val')
    
    
    # files = list(yolo_dataset['image_labels'].keys())
    # logger.info(f'# of Files: {len(files)}')
    # random.shuffle(files)
    # logger.info(f'# of Files: {len(files)}')
    # shuffle the files
    train_files = list(train_yolo_dataset['image_labels'].keys())
    val_files = list(val_yolo_dataset['image_labels'].keys())
    
    # if args.get('split_ratio', 0.0)>0.0:
        
    #     n_train = int(len(files) * args.get('split_ratio'))
    #     train_files = list(files)[:n_train]
    #     val_files = list(files)[n_train:]
    #     if len(val_files)>len(train_files):
    #         train_files, val_files = val_files, train_files
    #     if len(val_files)==0:
    #         logger.warning('no validation files')
    
    
    logger.info(f'train files: {len(train_files)}')
    logger.info(f'val files: {len(val_files)}')
    

    write_txts(train_yolo_dataset['image_labels'], path_txts=path_txts_train, fnames=train_files)
    write_txts(val_yolo_dataset['image_labels'], path_txts=path_txts_val,fnames=val_files if len(val_files)>0 else None)
    
    # move the images to the output folder
    
    path_out_imgs_train = os.path.join(path_out, 'images/train')
    path_out_imgs_val = os.path.join(path_out, 'images/val')
    
    # move the images to the output folder
    for p in [path_out_imgs_train, path_out_imgs_val]:
        if not os.path.exists(p):
            os.makedirs(p)
    
        # write class map yolo yaml
    with open(os.path.join(args['path_out'], 'dataset.yaml'), 'w') as f:
        dt = {
            'path': path_out,
            'train': 'images/train',
            'val': 'train' if len(val_files)==0 else 'images/val',
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
    
    # if args.get('bg', False):
    #     fnames = [os.path.basename(k) for k in yolo_dataset['image_labels'].keys()]
    # else:
    #     fnames = [os.path.basename(k) for k in yolo_dataset['image_labels'].keys() if len(yolo_dataset['image_labels'][k])>0]
    
    train_fnames = []
    val_fnames = []
    
    if not args.get('bg'):
        train_fnames = [os.path.basename(k) for k in train_files if len(train_yolo_dataset['image_labels'][k])>0]
        if len(val_files)>0:
            val_fnames = [os.path.basename(k) for k in val_files if len(val_yolo_dataset['image_labels'][k])>0]
    
    else:
        train_fnames = [os.path.basename(k) for k in train_files]
        if len(val_files)>0:
            val_fnames = [os.path.basename(k) for k in val_files]
    
    copy_images_in_folder(path_img=path_train_imgs, path_out=path_out_imgs_train, fnames=train_fnames)
    
    if len(val_fnames)>0:
        copy_images_in_folder(path_img=path_val_imgs, path_out=path_out_imgs_val, fnames=val_fnames)
    

def main(args):
    convert_to_yolo(args)

if __name__ == '__main__':
    args = args()
    main(args)
    
    
    
    
    
    
    
    