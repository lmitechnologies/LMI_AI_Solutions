#built-in packages
import os
import glob
import shutil
import logging
import numpy as np
import json

#3rd party packages
import cv2

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def annotate(image, regions):
    new_image = image.copy()
    print("new_image.shape", new_image.shape)
    for region in regions:
        shape = region['shape_attributes']['name']
        if shape == 'polygon':
            a = list(zip(region['shape_attributes']['all_points_x'],region['shape_attributes']['all_points_y']))
            polygon = np.array(a)
            new_image = cv2.polylines(new_image, [polygon], True, (0,255,0), thickness=1)
        if shape == 'rect':
            new_image = cv2.rectangle(new_image, 
                                    (region['shape_attributes']['x'], region['shape_attributes']['y']), 
                                    (region['shape_attributes']['x']+region['shape_attributes']['width'], region['shape_attributes']['y']+region['shape_attributes']['height']),
                                    (0,0,255), 
                                    thickness=1)
    return new_image

def scretch_to_edges(imgs_path, input_json, object_labels, output_json, distance_x, distance_y, annotating):
    path_input_json = f"{imgs_path}/{input_json}"
    if not os.path.isfile(path_input_json):
        raise Exception(f'cannot find: {path_input_json}')

    if annotating:
        annot_imgs_path = f"{os.path.dirname(imgs_path)}/{os.path.basename(imgs_path)}_scretched_annot"
        if os.path.isdir(annot_imgs_path):
            shutil.rmtree(annot_imgs_path)
        os.makedirs(annot_imgs_path)


    with open(path_input_json) as f:
        j = json.load(f)
    
    new_j = {}
    for k, v in j.items():
        filename = v['filename']
        filepath = f"{imgs_path}/{filename}"
        regions = v['regions']
        if not regions:
            print("not regions", filename)
            continue
        if not os.path.isfile(filepath):
            logger.warning(F"cannot find file - {filepath}")

        image = cv2.imread(filepath)
        h,w = image.shape[:2]
        for region in regions:
            label = region['region_attributes']['Name']
            if label not in object_labels:
                print("skip label", label, "as not in", object_labels)
                continue            
            shape = region['shape_attributes']['name']
            if shape == 'polygon':
                xs = region['shape_attributes']['all_points_x']
                xs = [0 if abs(x-0)<distance_x else x for x in xs]
                xs = [w if abs(w-x)<distance_x else x for x in xs]

                ys = region['shape_attributes']['all_points_y']
                ys = [0 if abs(y-0)<distance_y else y for y in ys]
                ys = [h if abs(h-y)<distance_y else y for y in ys]

                region['shape_attributes']['all_points_x'] = xs
                region['shape_attributes']['all_points_y'] = ys
            elif shape == 'rect':
                x,y,width,height = region['shape_attributes']['x'],region['shape_attributes']['y'],region['shape_attributes']['width'],region['shape_attributes']['height']
                if abs(x-0)<distance_x:
                    width += x-0
                    x = 0
                if abs(w-(x+width))<distance_x:
                    width += w-(x+width)
                if abs(y-0)<distance_y:
                    height += y-0
                    y = 0
                if abs(h-(y+height))<distance_y:
                    height += h-(y+height)
                region['shape_attributes']['x'],region['shape_attributes']['y'],region['shape_attributes']['width'],region['shape_attributes']['height'] = \
                                    x,y,width,height
            else:
                pass
        new_j[k] = v

        if annotating:
            annot_image = annotate(image, regions)
            cv2.imwrite(f"{annot_imgs_path}/{filename}", annot_image)

    with open(f"{imgs_path}/{output_json}", "w") as f:
            json.dump(new_j, f, indent=4)


if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgs_path', required=True, help='the path to images')
    ap.add_argument('--input_json', default='labels.json', help='[optinal] the input json file name in imgs_path')
    ap.add_argument('--output_json', default='labels_scretched.json', help='the output json file that contains the scretched labels, corresponds to imgs_path')
    ap.add_argument('--object_labels',required=True,help='Comma separated list of labels to scretch.')

    ap.add_argument('--distance_x',default='12',help='the maximum distance to the left or right edges of a vertex that can be set to edge.')
    ap.add_argument('--distance_y',default='0',help='the maximum distance to the top or bottom edges of a vertex that can be set to edge.')
    
    ap.add_argument('--annotating', action='store_true', dest='annotating', help='annotate to visualize the scretched labels')
    ap.set_defaults(annotating=False)
    args = vars(ap.parse_args())

    imgs_path = args['imgs_path']
    input_json = args['input_json']
    output_json = args['output_json']
    object_labels = args['object_labels'].replace(" ",'').split(",")

    distance_x = int(args['distance_x'])
    distance_y = int(args['distance_y'])
    annotating = bool(args['annotating'])
    scretch_to_edges(imgs_path, input_json, object_labels, output_json, distance_x, distance_y, annotating)
