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
    ap.add_argument('--path_imgs', '-i', required=True, help='the path of a image folder')
    ap.add_argument('--path_json', default='labels.json', help='[optional] the path of a json file that corresponds to path_imgs, default="labels.json" in path_imgs')
    ap.add_argument('--path_out', '-o', required=True, help='the output path for dataset')
    ap.add_argument('--split_ratio', type=float, default=0.0, help='the ratio to split the dataset, default=0.0, all images will go to train' )
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
    logger.info(f' wrote {len(fnames)} txt files to {path_txts}')
    

def convert_to_yolo(args):
    path_file = args['path_json'] if args['path_json']!='labels.json' else os.path.join(args.get('path_imgs'), args['path_json'])
    path_out = args['path_out']
    path_imgs = args['path_imgs']
    merge_box = args.get('merge_box', False)
    bbox_to_mask = True if args.get('convert', False) and args.get('seg', False) else False
    mask_to_od = True if args.get('convert', False) and not args.get('seg', False) else False
    target_classes = args['target_classes'].split(',')
    use_obb = args.get('obb', False)
    
    # check if the dataset path exists
    if not os.path.exists(path_imgs):
        raise Exception('The image path does not exist')
    if not os.path.exists(path_file):
        raise Exception('The json file does not exist')
    
    # load the json file
    
    dataset = Dataset.load(path_file)
    
    yolo_dataset = dataset.to_yolo(
        merge_boxes=merge_box,
        to_segmentation=bbox_to_mask,
        to_object_detection=mask_to_od,
        target_classes=target_classes,
        use_obb=use_obb
    )
    
    # print(yolo_dataset)
    
    # path for labels files
    path_txts = os.path.join(path_out, 'labels')
    
    
    files = list(yolo_dataset['image_labels'].keys())
    logger.info(f'# of Files: {len(files)}')
    random.shuffle(files)
    logger.info(f'# of Files: {len(files)}')
    # shuffle the files
    train_files = files
    val_files = []
    
    if args.get('split_ratio', 0.0)>0.0:
        
        n_train = int(len(files) * args.get('split_ratio'))
        train_files = list(files)[:n_train]
        val_files = list(files)[n_train:]
    
    
    logger.info(f'train files: {len(train_files)}')
    logger.info(f'val files: {len(val_files)}')
    

    write_txts(yolo_dataset['image_labels'], path_txts=path_txts, fnames=train_files)
    write_txts(yolo_dataset['image_labels'], path_txts=path_txts,fnames=val_files if len(val_files)>0 else None)
    
    # move the images to the output folder
    
    path_out_imgs_train = os.path.join(path_out, 'train')
    path_out_imgs_val = os.path.join(path_out, 'val')
    
    # move the images to the output folder
    for p in [path_out_imgs_train, path_out_imgs_val]:
        if not os.path.exists(p):
            os.makedirs(p)
    
        # write class map yolo yaml
    with open(os.path.join(args['path_out'], 'dataset.yaml'), 'w') as f:
        dt = {
            'path': path_out,
            'train': 'train',
            'val': 'train' if len(val_files)==0 else 'val',
            'test': None,
        }
        if yolo_dataset['n_kpts']:
            dt['kpt_shape'] = [yolo_dataset['n_kpts'],2]
        dt['names'] = {int(v):k for k,v in yolo_dataset['class_map'].items() }
        yaml.dump(dt, f, sort_keys=False)
    
    fname = os.path.join(args['path_out'], 'class_map.json')
    
    with open(fname, 'w') as outfile:
        json.dump({k:int(v)
            for k,v in yolo_dataset['class_map'].items()}, outfile)
    
    # if args.get('bg', False):
    #     fnames = [os.path.basename(k) for k in yolo_dataset['image_labels'].keys()]
    # else:
    #     fnames = [os.path.basename(k) for k in yolo_dataset['image_labels'].keys() if len(yolo_dataset['image_labels'][k])>0]
    
    train_fnames = []
    val_fnames = []
    
    if not args.get('bg'):
        train_fnames = [os.path.basename(k) for k in train_files if len(yolo_dataset['image_labels'][k])>0]
        if len(val_files)>0:
            val_fnames = [os.path.basename(k) for k in val_files if len(yolo_dataset['image_labels'][k])>0]
    
    else:
        train_fnames = [os.path.basename(k) for k in train_files]
        if len(val_files)>0:
            val_fnames = [os.path.basename(k) for k in val_files]
    
    copy_images_in_folder(path_img=path_imgs, path_out=path_out_imgs_train, fnames=train_fnames)
    
    if len(val_fnames)>0:
        copy_images_in_folder(path_img=path_imgs, path_out=path_out_imgs_val, fnames=val_fnames)
    

def main(args):
    convert_to_yolo(args)

if __name__ == '__main__':
    args = args()
    main(args)
    
    
    
    
    
    
    
    