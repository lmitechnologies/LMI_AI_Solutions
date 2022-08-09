#%% extract bbox regions from JSON file
import json
import pandas as pd
import argparse
import csv
import os
import cv2
import ast
import glob
import numpy as np
from image_utils.img_resize import resize 


def extract_ROI_from_JSON(data_folder_path,output_csv_file_name,label_name='Name',target_classes=[],render=False,mask_to_bbox=False):
    """
    Description:
        parses JSON and writes values to csv
    Arguments:
        data_folder_path (string) - location of json file
        output_csv_file_name (string) - name of the created csv file
        label_name (string) - key name of the label value in region_attributes section of json
        render (bool) - if images should be displayed
        is_mask (bool) - if shape is a polygon
        mask_to_bbox (bool) - if it should write bounding box from mask region
    """

    json_files=glob.glob(os.path.join(data_folder_path,'*.json'))
    output_csv_file_path=os.path.join(data_folder_path,output_csv_file_name)
    write_append_option='w'

    for json_file in json_files:
        # parse the labeling data
        with open(json_file,'r') as read_file:
            object_labels=json.load(read_file)
        
        # keys for each region
        keys=list(object_labels.keys())

        # create a new csv file for first JSONfile, append the same for all subsequent json files
        with open(output_csv_file_path,write_append_option,newline='') as csvfile:
            # extract top left corner and bbox dimensions
            labelWriter=csv.writer(csvfile,delimiter=';')
            for i,key in enumerate(keys):         
                regions=object_labels[key]['regions']
                fname=object_labels[key]['filename']
                print('[INFO] filename=',fname)
                filepath=os.path.join(data_folder_path,fname)
                if os.path.exists(filepath):
                    for j,_ in enumerate(regions):
                        label=regions[j]['region_attributes'][label_name]
                        if label not in target_classes:
                            continue
                        # general mask label file
                        # if is_mask:
                        # test for polygon region
                        if regions[j]['shape_attributes']['name']=='polygon':
                            xj=regions[j]['shape_attributes']['all_points_x']
                            yj=regions[j]['shape_attributes']['all_points_y']
                            if mask_to_bbox:
                                print('[INFO] Writing bounding box from mask region: ',label)
                                xj=np.array(xj,dtype=np.int32)
                                yj=np.array(yj,dtype=np.int32)
                                x_ul=xj.min()
                                y_ul=yj.min()
                                x_lr=xj.max()
                                y_lr=yj.max()
                                labelWriter.writerow([fname,label,'1.0','rect','upper left',x_ul,y_ul])
                                labelWriter.writerow([fname,label,'1.0','rect','lower right',x_lr,y_lr])
                            else:         
                                labelWriter.writerow([fname,label,'1.0','polygon','x values']+xj)
                                labelWriter.writerow([fname,label,'1.0','polygon','y values']+yj)
                        # test for simple bounding box region
                        elif regions[j]['shape_attributes']['name']=='rect':
                            xj=regions[j]['shape_attributes']['x']
                            yj=regions[j]['shape_attributes']['y']
                            wj=regions[j]['shape_attributes']['width']
                            hj=regions[j]['shape_attributes']['height']
                            x_ul=xj
                            y_ul=yj
                            x_lr=xj+wj
                            y_lr=yj+hj
                            labelWriter.writerow([fname,label,'1.0','rect','upper left',x_ul,y_ul])
                            labelWriter.writerow([fname,label,'1.0','rect','lower right',x_lr,y_lr])
                        elif regions[j]['shape_attributes']['name']=='point':
                            cx=regions[j]['shape_attributes']['cx']
                            cy=regions[j]['shape_attributes']['cy']
                            labelWriter.writerow([fname,label,'1.0','point','cx',cx])
                            labelWriter.writerow([fname,label,'1.0','point','cy',cy])
                        else:
                            raise Exception('Unsupported label type.  polygon and rect supported.')

                        # else:
                        #     xj=regions[j]['shape_attributes']['x']
                        #     yj=regions[j]['shape_attributes']['y']
                        #     wj=regions[j]['shape_attributes']['width']
                        #     hj=regions[j]['shape_attributes']['height']

                        #     x_ul=xj
                        #     y_ul=yj
                        #     x_lr=xj+wj
                        #     y_lr=yj+hj
                        #     labelWriter.writerow([fname,label,'1.0','rect','upper left',x_ul,y_ul])
                        #     labelWriter.writerow([fname,label,'1.0','rect','lower right',x_lr,y_lr])
                        
                        if render:
                            # display image
                            THICKNESS=2
                            RADIUS=20
                            image=cv2.imread(filepath)
                            if regions[j]['shape_attributes']['name']=='polygon' and (not mask_to_bbox):
                                overlay=image.copy()
                                xj=np.array(xj,dtype=np.int32)
                                yj=np.array(yj,dtype=np.int32)
                                x_ul=xj.min()
                                y_ul=yj.min()
                                x_lr=xj.max()
                                y_lr=yj.max()
                                pts=np.stack((xj,yj),axis=1)
                                pts=np.expand_dims(pts,axis=0)
                                overlay=cv2.fillPoly(overlay,pts,(255,0,0))
                            elif regions[j]['shape_attributes']['name']=='rect' or (regions[j]['shape_attributes']['name']=='polygon' and mask_to_bbox):
                                image=cv2.rectangle(image,(x_ul,y_ul),(x_lr,y_lr),(255,0,0),thickness=THICKNESS)
                                cv2.putText(image,label,(x_ul,y_ul-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),THICKNESS,cv2.LINE_AA)
                            elif regions[j]['shape_attributes']['name']=='point':
                                image=cv2.circle(image,(cx,cy),RADIUS,(0,255,0),-1)
                                cv2.putText(image,label,(cx-RADIUS-2*THICKNESS,cy-RADIUS-2*THICKNESS),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),THICKNESS,cv2.LINE_AA)
                            image=resize(image,width=512)    
                            cv2.imshow('image',image)
                            cv2.waitKey(500)
                else:
                    print(f'[INFO] File does not exist in data path: {filepath}')

            # append csv file for each additional json file
            write_append_option='a'



if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-d','--data_path',required=True)
    ap.add_argument('--output_fname',required=True)
    ap.add_argument('-r','--render',default="False")
    ap.add_argument('--mask_to_bbox',default="False")
    ap.add_argument('--target_classes',default='',help='comma separated target class names')
    ap.add_argument('--label_name',default='Name')
    args=vars(ap.parse_args())

    data_path=args['data_path']
    output_filename=args['output_fname']
    label_name=args['label_name']
    target_classes=args['target_classes']
    if target_classes=='':
        target_classes = []
    else:
        target_classes = target_classes.split(',')
    print(f'target_classes: {target_classes}')

    # is_mask=ast.literal_eval(args['is_mask'])
    # if type(is_mask) is not type(True):
    #     raise Exception('is_mask should be "True" or "False".') 

    render=ast.literal_eval(args['render'])
    if type(render) is not type(True):
        raise Exception('render should be "True" or "False".') 
    
    mask_to_bbox=ast.literal_eval(args['mask_to_bbox'])
    if type(render) is not type(True):
        raise Exception('mask_to_bbox should be "True" or "False".')

    
    extract_ROI_from_JSON(data_path,output_filename,label_name=label_name,target_classes=target_classes,render=render,mask_to_bbox=mask_to_bbox)