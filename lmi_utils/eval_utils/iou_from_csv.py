#%%
import re
import numpy as np
import cv2
import os
import argparse
import csv


def csv_to_dictionary(csv_file: str,object_classes: str):
    '''
    DESCRIPTION: 
        Converts a .csv file with image paths, labels, and ROIs to a python dictionary.
    ARGUMENTS:
        csv_file: input csv file path
        object classes: list of object classes
    RETURNS:
        list of dictionaries for each object in the input csv file
    
    '''
    
    supported_shapes=['rect','polygon','point']
    list_of_dics=[]
    rows=open(csv_file).read().strip().split('\n')
    ul_found=False 
    lr_found=False 
    x_found=False 
    y_found=False
    cx_found=False
    cy_found=False

    # Step through each row and create a new dictionary when the row includes a target object
    for row in rows:
        print(row)
        if row[-1]==';':
            row=row[0:-1]
        row=row.split(';')

         # search the cells for target object classes
        obj=list(set(object_classes) & set(row))
        if not obj or len(obj)>1:
            # raise Exception(f'csv format error. Bad object definition: {obj}')
            continue
        else:
            this_obj=obj[0]

                
        # search cells for image file
        image_file=[x for x in row if re.search('.png',x)] 
        if len(image_file) != 1:
            raise Exception('csv format error. Image file not present.')
        else:
            this_file=image_file[0]

        # search for supported shapes
        shape=list(set(supported_shapes) & set(row))
        if not shape or len(shape)>1:
            raise Exception('csv format error. Unsupported shape.')
        else:
            this_shape=shape[0]
        
        if this_shape=='rect':
            uli=[i for i,s in enumerate(row) if 'upper left' in s]
            if len(uli)==1:
                x_ul=int(row[uli[0]+1])
                y_ul=int(row[uli[0]+2])
                ul=(x_ul,y_ul)
                ul_found=True
            lri=[i for i,s in enumerate(row) if 'lower right' in s]
            if len(lri)==1:
                x_lr=int(row[lri[0]+1])
                y_lr=int(row[lri[0]+2])
                lr=(x_lr,y_lr)
                lr_found=True
            if ul_found and lr_found:
                list_of_dics.append({"image_file":this_file,"obj_class":this_obj,"shape":this_shape,"upper_left":ul,"lower_right":lr})
                ul_found=False 
                lr_found=False 

        if this_shape=='polygon':
            xind=[i for i,s in enumerate(row) if 'x values' in s]
            if len(xind)==1:          
                this_x=np.array(row[xind[0]+1:],dtype=np.uint32)
                x_found=True
            yind=[i for i,s in enumerate(row) if 'y values' in s] 
            if len(yind)==1:
                this_y=np.array(row[yind[0]+1:],dtype=np.uint32)
                y_found=True
            if x_found and y_found:
                list_of_dics.append({"image_file":this_file,"obj_class":this_obj,"shape":this_shape,"x_values":this_x,"y_values":this_y})
                x_found=False 
                y_found=False
        
        if this_shape=='point':
            cxi=[i for i,s in enumerate(row) if 'cx' in s]
            if len(cxi)==1:
                cx=int(row[cxi[0]+1])
                cx_found=True
            cyi=[i for i,s in enumerate(row) if 'cy' in s]
            if len(cyi)==1:
                cy=int(row[cyi[0]+1])
                cy_found=True
            if cx_found and cy_found:
                list_of_dics.append({"image_file":this_file,"obj_class":this_obj,"shape":this_shape,"cx":cx,"cy":cy})
                cx_found=False 
                cy_found=False 
        
    return list_of_dics

def find_class_index(target_class,list_of_dicts):
    out=[]
    for i,lod in enumerate(list_of_dicts):
        if lod['obj_class']==target_class:
            out.append(i)
    return out
        
def main(model_path:str,manual_path:str,data_dir:str,labels:str,output_dir:str,render:bool):
    '''
    DESCRIPTION: 
        Computes IOU from .csv label files
        Annotates object bounding boxes and segments
        Saves IOU to .csv file
    ARGUMENTS:
        model_path: predicted model performance csv file
        manual_path: manual label csv file
        data_dir: path to images used for inference
        labels: label definitions (comma seperated string or None)
        output_dir: path to folder for output file and annotated images
        render: option to plot images
    RETURNS:
        None
    
    '''
    
    if labels is None:
        object_classes_model=np.genfromtxt(model_path,delimiter=';',dtype=str)[:,1]
        object_classes_manual=np.genfromtxt(manual_path,delimiter=';',dtype=str)[:,1]
        obj_classes=np.hstack((object_classes_model,object_classes_manual))
        obj_classes=list(np.unique(obj_classes))
    else:
        try:
            obj_classes=labels.split(",")
        except:
            print(f'Incorrect labels definition: {labels}')

    manual_data=csv_to_dictionary(manual_path,obj_classes)
    model_data=csv_to_dictionary(model_path,obj_classes)
    #%% get filenames
    model_file_set=set([item['image_file'] for item in model_data]) 
    manual_file_set=set([item['image_file'] for item in manual_data])
    file_intersection=list(model_file_set & manual_file_set)
    #%% search by mask ID
    iou=[]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outfile=os.path.join(output_dir,'iou.csv')
    #%% Generate IOU results
    with open(outfile,'w',newline='') as csvfile:
        rowWriter=csv.writer(csvfile,delimiter=',')
        rowWriter.writerow(['model label',model_path])
        rowWriter.writerow(['manual label',manual_path])
        #  
        for file_x in file_intersection:
            current_model=[item for item in model_data if item['image_file']==file_x]
            current_manual=[item for item in manual_data if item['image_file']==file_x]
            image=cv2.imread(os.path.join(data_dir,current_model[0]['image_file']))
            for obj_class in obj_classes:
                i_manual=find_class_index(obj_class,current_manual)
                i_model=find_class_index(obj_class,current_model)
                if (not i_manual) and (not i_model):
                    continue
                else:
                    found_polygons=False
                    found_boxes=False
                    manual_mask=np.zeros(image.shape[0:2],dtype=np.uint8)
                    for i in i_manual:
                        if current_manual[i]['shape']=='polygon':
                            found_polygons=True
                            x_manual=current_manual[i]['x_values']
                            y_manual=current_manual[i]['y_values']
                            pts=np.stack((x_manual,y_manual),axis=1)
                            pts_manual=np.expand_dims(pts,axis=0).astype(np.int32)
                            cv2.polylines(image,pts_manual,True,(255,0,0),1)
                            cv2.fillPoly(manual_mask,pts_manual,255)
                        elif current_manual[i]['shape']=='rect':
                            found_boxes=True
                            uleft=current_manual[i]['upper_left']
                            bright=current_manual[i]['lower_right']
                            cv2.rectangle(image,uleft,bright,(255,0,0),1)
                            cv2.rectangle(manual_mask,uleft,bright,255,-1)
                        else:
                            raise Exception('Unknown mask shape.')

                    model_mask=np.zeros(image.shape[0:2],dtype=np.uint8)
                    for i in i_model:
                        if found_polygons and current_model[i]['shape']=='polygon':
                            x_model=current_model[i]['x_values']
                            y_model=current_model[i]['y_values']
                            pts=np.stack((x_model,y_model),axis=1)
                            pts_model=np.expand_dims(pts,axis=0).astype(np.int32)
                            cv2.polylines(image,pts_model,True,(0,0,255),1)
                            cv2.fillPoly(model_mask,pts_model,255)
                        elif found_boxes and current_model[i]['shape']=='rect':
                            uleft=current_model[i]['upper_left']
                            bright=current_model[i]['lower_right']
                            cv2.rectangle(image,uleft,bright,(0,0,255),1)
                            cv2.rectangle(model_mask,uleft,bright,255,-1)
                        else:
                            print(f"[INFO] skipping model, {current_model[i]['shape']} not present in labeled data.")
                    
                    image_rsz=image.copy()
                    if render:
                        cv2.imshow('Validation Window',image_rsz)
                        cv2.waitKey(100)
                    img_file=os.path.splitext(file_x)[0]+'_iou.png'
                    cv2.imwrite(os.path.join(output_dir,img_file),image_rsz)
                    
                    #compute IOU
                    union=np.sum(cv2.bitwise_or(model_mask,manual_mask))
                    intersection=np.sum(cv2.bitwise_and(model_mask,manual_mask))
                    iou_i=intersection/union*100
                    iou.append(iou_i)
                    print('[INFO] Class %s : IOU = %.2f percent'%(obj_class,float(iou_i)))
                    rowWriter.writerow([file_x,obj_class,iou_i])
        if render:
            cv2.destroyWindow('Validation Window')
        
        if iou:
            iou_np=np.array(iou)
            iou_np=np.nan_to_num(iou_np)
            iou_sort=np.sort(iou_np)
            iou_filt=iou_sort[1:-1]
            iou_min=np.min(iou_filt)
            iou_max=np.max(iou_filt)
            iou_mean=np.mean(iou_filt)
            print('[INFO] Mean IOU = %.2f percent' % iou_mean)
            print('[INFO] Max IOU = %.2f percent' % iou_max)
            print('[INFO] Min IOU = %.2f percent' % iou_min)
        else:
            print('[INFO] Target label not found in the input files.')
    
if __name__== "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--manual_csv',required=True,help='manual label .csv file')
    ap.add_argument('--model_csv',required=True,help='model output .csv file')
    ap.add_argument('--data_dir',required=True,help='path of data directory')
    ap.add_argument('--labels',default=None,help='comma seperated list of labels.')
    ap.add_argument('--output_dir',required=True,help='output dir')
    ap.add_argument('--render',dest="render", action='store_true')
    ap.set_defaults(render=False)

    args = vars(ap.parse_args())
    render = args['render']

    main(args['model_csv'],args['manual_csv'],args['data_dir'],args['labels'],args['output_dir'],render)


    




