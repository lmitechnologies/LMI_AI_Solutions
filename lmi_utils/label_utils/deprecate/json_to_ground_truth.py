#%% extract bbox regions from JSON file
import json
import argparse
import os
import cv2
import shutil
import numpy as np


def json_to_ground_truth(data_path, output_path, input_json='labels.json'):
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    with open(f"{data_path}/{input_json}") as f:
        j = json.load(f)
        
    for _,v in j.items():
        filename = v['filename']
        regions = v['regions']
        if not regions:
            continue
        name,ext = os.path.splitext(filename)
        out_name = f"{name}_mask{ext}"
        image = cv2.imread(f"{data_path}/{filename}")
        ground_truth = np.zeros(image.shape, dtype=np.uint8)
        for region in regions:
            shape_attributes = region['shape_attributes']
            xs = shape_attributes['all_points_x']
            ys = shape_attributes['all_points_y']
            points = np.array(list(zip(xs,ys)), dtype=np.int32)
            cv2.fillPoly(ground_truth, pts=[points], color=(255, 255, 255))
        cv2.imwrite(f"{output_path}/{out_name}", cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-d','--data_path',required=True)
    ap.add_argument('--output_path',required=True)
    ap.add_argument('--label_fname',default='labels.json')
    args=vars(ap.parse_args())

    data_path=args['data_path']
    output_fname=args['output_fname']
    label_fname=args['label_fname']

    json_to_ground_truth(data_path,output_fname,label_fname=label_fname)