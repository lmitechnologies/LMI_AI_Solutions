#built-in packages
import os
import glob
import shutil
import logging
import json

#3rd party packages
import cv2

#LMI packages
from label_utils.bbox_utils import convert_ls_obb_to_yolo, get_lst_bbox_to_xywh


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
                if "yolo_obb" not in result['value']:
                    bbox = result['value']
                    pixel_x, pixel_y, pixel_width, pixel_height = get_lst_bbox_to_xywh(bbox['x'],bbox['y'], bbox['width'],bbox['height'],result['original_width'], result['original_height'])
                    # now update the lst annotation dictionary with pixel coordinates
                    result['value']['x'] = pixel_x
                    result['value']['y'] = pixel_y
                    result['value']['width'] = pixel_width
                    result['value']['height'] = pixel_height
                    result['value']['yolo_obb'] = convert_ls_obb_to_yolo(result)
                row = [
                    class_id
                ]
                row += result['value']['yolo_obb']
                fname_to_labels[fname].append(row)
    return fname_to_labels

def create_text_files(output_folder_path, fname_to_rows):
    labels_path = os.path.join(output_folder_path, 'labels')
    for fname, rows in fname_to_rows.items():
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

def copy_images_to_yolo_images(images_dir, output_dir):
    # move all the images to a folder
    images = glob.glob(
        os.path.join(images_dir, '*.png')
    )
    images_folder = os.path.join(output_dir, 'images')
    for image in images:
        shutil.copy(image, images_folder)
    return None

            
def create_yolo_dataset_folders(output_folder: str) -> None:
    #create the folder if its not there
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # create image folder
    os.makedirs(os.path.join(output_folder,'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder,'labels'), exist_ok=True)
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

    # generate the text label files
    create_text_files(output_folder_path=args['output_path'], fname_to_rows=fname_rows_map)

    # copy all images 
    copy_images_to_yolo_images(args['images_dir'], args['output_path'])



if __name__=='__main__':
    main()