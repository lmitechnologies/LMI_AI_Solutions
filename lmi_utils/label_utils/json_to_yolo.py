import os
import json
from dataset_utils.representations import Dataset
from label_utils.csv_to_yolo import write_txts, copy_images_in_folder
import logging
import yaml
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', '-i', required=True, help='the path of a image folder')
    ap.add_argument('--path_file', default='labels.json', help='[optional] the path of a json file that corresponds to path_imgs, default="labels.json" in path_imgs')
    ap.add_argument('--path_out', '-o', required=True, help='the output path')
    ap.add_argument('--target_classes',default='all', help='[optional] the comma separated target classes, default=all')
    ap.add_argument('--obb', action='store_true', help='support for oriented bounding box support')
    ap.add_argument('--seg', action='store_true', help='convert label formats: mask-to-bbox if "--convert" is enabled, otherwise bbox-to-mask')
    ap.add_argument('--convert', action='store_true', help='convert label formats: bbox-to-mask if "--seg" is enabled, otherwise mask-to-bbox')
    ap.add_argument('--bg', action='store_true', help='save images with no labels, where yolo models treat them as background')
    ap.add_argument('--merge_box', action='store_true', help='merge multiple instances of same class boxes into one. Brush labels only!')
    args = vars(ap.parse_args())
    return args

def convert_to_yolo(args):
    path_file = args['path_file'] if args['path_file']!='labels.json' else os.path.join(args.get('path_imgs'), args['path_file'])
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
    
    text_dict = {}
    
    for k,v in yolo_dataset['image_labels'].items():
        f_ext = os.path.basename(k).split('.')[-1]
        if args.get('bg', False):
            text_dict[k.replace(f'.{f_ext}', '')] = v
        else:
            if len(v)>0:
                text_dict[k.replace(f'.{f_ext}', '')] = v
            
    write_txts(text_dict, path_txts)
    
    # move the images to the output folder
    
    path_out_imgs = os.path.join(path_out, 'images')
    
    # move the images to the output folder
    if not os.path.exists(path_out_imgs):
        os.makedirs(path_out_imgs)
    
        # write class map yolo yaml
    with open(os.path.join(args['path_out'], 'dataset.yaml'), 'w') as f:
        dt = {
            'path': '/app/data',
            'train': 'images',
            'val': 'images',
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
    
    if args.get('bg', False):
        fnames = [os.path.basename(k) for k in yolo_dataset['image_labels'].keys()]
    else:
        fnames = [os.path.basename(k) for k in yolo_dataset['image_labels'].keys() if len(yolo_dataset['image_labels'][k])>0]
    
    copy_images_in_folder(path_img=path_imgs, path_out=path_out_imgs, fnames=fnames)

def main(args):
    convert_to_yolo(args)

if __name__ == '__main__':
    args = args()
    main(args)
    
    
    
    
    
    
    
    