#%%
from iou.iou_from_csv import csv_to_dictionary
import json
import os
import argparse
import numpy as np

#%% read csv
def csv_to_json(input_csv,output_json):

    obj_classes=np.genfromtxt(input_csv,delimiter=';',dtype=str)[:,1]
    obj_classes=list(np.unique(obj_classes))

    label_dict=csv_to_dictionary(input_csv,obj_classes)
    output_dict={}
    list_of_images=[]
    for label in label_dict:
        if label['image_file'] in list_of_images:
            pass
        else:
            list_of_images.append(label['image_file'])
            output_dict.update({label['image_file']:{'filename':label['image_file'],'size':0,'regions':[],'file_attributes':{}}})
        if label['shape'] == 'polygon':
            xvals=[int(val) for val in label['x_values']]
            yvals=[int(val) for val in label['y_values']]
            output_dict[label['image_file']]['regions'].append({'shape_attributes':{\
                'name':'polygon',\
                'all_points_x':xvals,\
                'all_points_y':yvals},\
                'region_attributes':{'Name':label['obj_class']}} )
        elif label['shape'] == 'rect':
            x=label['upper_left'][0]
            y=label['upper_left'][1]
            width=label['lower_right'][0]-x
            height=label['lower_right'][1]-y
            output_dict[label['image_file']]['regions'].append({'shape_attributes':{\
                'name':'rect',\
                'x':int(x),\
                'y':int(y),\
                'width':int(width),\
                'height':int(height)},\
                'region_attributes':{'Name':label['obj_class']}} )
        else:
            raise Exception(f"[ERROR] Unrecognized shape: {label['shape']}")
    
    with open(output_json, 'w') as result_fp:
        json.dump(output_dict, result_fp)

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_csv',required=True)
    ap.add_argument('--output_json',required=True)
    args=vars(ap.parse_args())

    input_csv=args['input_csv']
    output_json=args['output_json']
    
    csv_to_json(input_csv,output_json)