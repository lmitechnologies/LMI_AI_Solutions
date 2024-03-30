#built-in packages
import os
import glob
import shutil
import logging
import random
import json

#3rd party packages
import cv2

#LMI packages
from label_utils.bbox_utils import lst_to_yolo


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lst_obb_to_yolo_obb(path_json, class_map_json):
    with open(class_map_json, 'r') as f:
        class_map = json.load(f)

    with open(path_json, 'r') as f:
        data = json.load(f)
    
    # read the annotations
    fname_to_labels = {}
    for dt in data:
        fname = dt['data']['image'].split("=")[-1]
        fname = os.path.basename(fname)

        if fname not in fname_to_labels:
            fname_to_labels[fname] = []

        for annotation in dt['annotations']:
            for result in annotation['result']:
                if len(result['value']['rectanglelabels']) > 1:
                    logger.warning(
                        f"more than 1 class found for result"
                    )
                class_id = class_map[result['value']["rectanglelabels"][0]]
                row = [
                    class_id
                ]
                row += lst_to_yolo(original_height=result['original_height'], original_width=result['original_width'], x=result['value']['x'], y=result['value']['y'], width=result['value']['width'], height=result['value']['height'], rotation=result['value']['rotation'])
                fname_to_labels[fname].append(row)
    return fname_to_labels

def create_train_val_datasets(images_dir, output_dir, fname_to_rows):
    # move all the images to a folder
    images = glob.glob(
        os.path.join(images_dir, '*.png')
    )
    images_folder_train = os.path.join(output_dir, 'train/images')
    images_folder_val = os.path.join(output_dir, 'val/images')

    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    for image in train_images:
        shutil.copy(image, images_folder_train)

    for image in val_images:
        shutil.copy(image, images_folder_val)

    # now create the text files
        
    create_text_files(output_folder_path=output_dir, fname_to_rows=fname_to_rows,images=list(map(os.path.basename, train_images)), dataset_sub_folder='train/labels')
    create_text_files(output_folder_path=output_dir, fname_to_rows=fname_to_rows,images=list(map(os.path.basename, val_images)), dataset_sub_folder='val/labels')
    
    return None

def create_text_files(output_folder_path, fname_to_rows, images,dataset_sub_folder):
    labels_path = os.path.join(output_folder_path, dataset_sub_folder)
    for fname, rows in fname_to_rows.items():
        if fname not in images:
            continue
        txt_f_name = fname.split('.')[0]
        txt_f_name = f"{txt_f_name}.txt"
        txt_file_path = os.path.join(labels_path, txt_f_name)
        with open(txt_file_path, 'w+') as lbl_file:
            for row in rows:
                row_str = " ".join(map(str, row))
                row_str += '\n'
                lbl_file.write(row_str)
        lbl_file.close()
    return None
            
def create_yolo_dataset_folders(output_folder: str) -> None:
    #create the folder if its not there
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.makedirs(os.path.join(output_folder,'train/images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder,'val/images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder,'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(output_folder,'val/labels'), exist_ok=True)
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True, help='the path of a image folder')
    ap.add_argument('--annotations_json', required=True, help='the path to the annotations json file in label studio format')
    ap.add_argument('--class_map_json', help='class map json file')
    ap.add_argument('--output_path', required=True, help='the output path')
    args = vars(ap.parse_args())


    # create the folders for the yolo dataset
    create_yolo_dataset_folders(output_folder=args['output_path'])
    
    # now create the file name to yolo labels dictionary
    fname_rows_map = lst_obb_to_yolo_obb(path_json=os.path.join(args['images_dir'], args['annotations_json']), class_map_json=os.path.join(args['images_dir'], args['class_map_json']))

    create_train_val_datasets(args['images_dir'], args['output_path'], fname_to_rows=fname_rows_map)



if __name__=='__main__':
    main()