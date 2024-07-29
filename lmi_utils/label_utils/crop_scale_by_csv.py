#%% load modules
import csv
from eval_utils.iou_from_csv import csv_to_dictionary
from eval_utils.iou_from_csv import find_class_index
from label_utils.crop_scale_labeled_image import crop_scale_labeled_image
from label_utils.bbox_utils import rotate
import os
import cv2
import argparse
import numpy as np

NAN_INT=-999999

#%% File paths
def crop_scale(input_data_dir,input_csv_path,output_data_dir,output_csv_path,all_labels,boundingbox_label,object_labels,scl_w,scl_h,p2w,p2h,is_plot): 
    # def_width=False
    # try:
    #     scl_w=int(scl_w)
    #     p2h=int(p2h)
    #     p2w=int(p2w)
    # except:
    #     print('[INFO] Skip rescaling.')
    #     def_width=True
    
    object_classes=object_labels
    label_dicts=csv_to_dictionary(input_csv_path,all_labels)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    outfile=output_csv_path
    # get unique data files
    files_set=set([item['image_file'] for item in label_dicts])
    image_files=list(files_set)
    image_files.sort()
    with open(outfile,'w',newline='') as csvfile:
        rowWriter=csv.writer(csvfile,delimiter=';')
        for image_file in image_files:
            print('[INFO] converting file: ',image_file)
            current_labels=[item for item in label_dicts if item['image_file']==image_file]
            old_image=cv2.imread(os.path.join(input_data_dir,current_labels[0]['image_file']),-1)
            if boundingbox_label is None:
                ivec_bbox=[0]
            else:
                ivec_bbox=find_class_index(boundingbox_label,current_labels)
            for i_bbox in ivec_bbox:
                if boundingbox_label is None:
                    ul=(0,0)
                    (Himg,Wimg)=old_image.shape[0:2]
                    lr=(Wimg,Himg)
                elif current_labels[i_bbox]['shape']=='rect':
                    print(current_labels[i_bbox])
                    ul=current_labels[i_bbox]['upper_left']
                    lr=current_labels[i_bbox]['lower_right']
                    angle=current_labels[i_bbox]['angle']
                    if angle != 0:
                        # rotate the rectangle
                        corners = rotate(ul[0],ul[1],lr[0]-ul[0],lr[1]-ul[1],angle)
                        ul = corners.min(axis=0)
                        lr = corners.max(axis=0)
                else:
                    raise Exception('Bounding box should be a rect.')
                bbox=(ul,lr)
                old_objects=[]
                object_labels=[]
                label_types=[]
                for obj_class in object_classes:
                    # step through remaining object classes
                    object_regions=find_class_index(obj_class,current_labels)
                    # step through split regions, appending all object regions     
                    for i in object_regions:
                        if current_labels[i]['shape']=='polygon':
                            x_vec=current_labels[i]['x_values']
                            y_vec=current_labels[i]['y_values']
                            pts=np.stack((x_vec,y_vec),axis=1)
                            label_types.append('polygon')
                        elif current_labels[i]['shape']=='rect':
                            ul=current_labels[i]['upper_left']
                            lr=current_labels[i]['lower_right']
                            angle=current_labels[i]['angle']
                            if angle != 0:
                                # rotate the rectangle
                                corners = rotate(ul[0],ul[1],lr[0]-ul[0],lr[1]-ul[1],angle)
                                ul = corners.min(axis=0)
                                lr = corners.max(axis=0)
                            x_vec=np.asarray([ul[0],lr[0],lr[0],ul[0]])
                            y_vec=np.asarray([ul[1],ul[1],lr[1],lr[1]])
                            pts=np.stack((x_vec,y_vec, angle),axis=1)
                            label_types.append('rect')
                        else:
                            raise Exception('Unknown object shape.')
                        old_objects.append(pts)
                        object_labels.append(obj_class)
                # extract new image and new objects
                new_image,new_objects=crop_scale_labeled_image(old_image,bbox,old_objects,new_width=scl_w,new_height=scl_h,p2h=p2h,p2w=p2w)
                hout,wout=new_image.shape[:2]         
                fname=os.path.splitext(image_file)[0]+'_crop_h'+str(hout)+'w'+str(wout)+'.png'
                fpath=os.path.join(output_data_dir,fname)
                if os.path.exists(fpath):
                    fname=os.path.splitext(image_file)[0]+'_crop_h'+str(hout)+'w'+str(wout)+'_v2.png'
                    fpath=os.path.join(output_data_dir,fname)

                cv2.imwrite(fpath,new_image)

                # step through object classes again, writing new objects
                for obj_class in object_classes:
                    # object_regions=object_labels.index(obj_class,current_labels)
                    object_regions=[i for i,lab in enumerate(object_labels) if lab==obj_class]
                    for j in object_regions:
                        label_type=label_types[j]
                        label=object_labels[j]
                        xj=list(new_objects[j][:,0])
                        yj=list(new_objects[j][:,1])
                        anglej=list(new_objects[j][:,2])
                        if not (np.any(np.asarray(xj)<NAN_INT) or np.any(np.asarray(yj)<NAN_INT)):
                            if label_type=='polygon':
                                rowWriter.writerow([fname,label,'1.0','polygon','x values']+xj)
                                rowWriter.writerow([fname,label,'1.0','polygon','y values']+yj)
                            elif label_type=='rect':
                                ul=[xj[0],yj[0]]
                                lr=[xj[2],yj[2]]
                                rowWriter.writerow([fname,label,'1.0','rect','upper left']+ul)
                                rowWriter.writerow([fname,label,'1.0','rect','lower right']+lr)
                                rowWriter.writerow([fname,label,'1.0','rect','angle']+anglej)
                            else:
                                raise Exception('Unknown object shape.')
                            # plot 
                            pts=new_objects[j].astype(np.int32)
                            # remove angle from pts
                            if label_type=='rect':
                                pts=pts[:,:2]
                            cv2.polylines(new_image,[pts],True,(255,0,0),1)
                            text_ind=pts.min(axis=0)
                            cv2.putText(new_image,label,(text_ind[0],text_ind[1]-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1,cv2.LINE_AA)
                            if is_plot:    
                                cv2.imshow('Validation Image',new_image)
                                cv2.waitKey(100)
                fpath=os.path.splitext(fpath)[0]+'_annot.png'
                cv2.imwrite(fpath,new_image)
                

if __name__ == '__main__':
    '''
    Args:
        input_data_path: directory of all images
        input_csv_path:  csv file that includes labels
        output_data_path: new images
        output_csv_path: new csv file that has rescaled labels
        label_defs: string that defines all labeled regions
        bb_label: label name for outer bounding box
        object labels: labeled regions of the internal objects
        scl_w, p2h: new image dimensions
    '''

    ap=argparse.ArgumentParser()
    ap.add_argument('--input_data_path',required=True,help='Input data directory.')
    ap.add_argument('--input_csv_path',required=True,help='Input label.csv path')
    ap.add_argument('--output_data_path',required=True,help='Output data directory')
    ap.add_argument('--output_csv_path',required=True,help='Output label.csv path')
    ap.add_argument('--label_defs',required=True,help='Comma separated list of all label categories.')
    ap.add_argument('--bb_label',default=None,help='Bounding box label for cropping. (If it exists.)')
    ap.add_argument('--object_labels',default=None,help='Comma separated list for all labels to keep.')
    # ap.add_argument('--split_bbw',type=int,default=1)
    ap.add_argument('--scale_width',type=int,default=None,help='Width of new image.  Scaling will preserve aspect ratio if scale_height is not defined.')
    ap.add_argument('--scale_height',type=int,default=None,help='Height of new image.  Scaling will preserve aspect ratio if scale_width is not defined.')
    ap.add_argument('--pad2width',type=int,default=None,help='Pad to width')
    ap.add_argument('--pad2height',type=int,default=None,help='Pad to height')
    ap.add_argument('--plot', action='store_true', help='plot the crop region')


    args=vars(ap.parse_args())

    idp=args['input_data_path']
    icsv=args['input_csv_path']
    odp=args['output_data_path']
    ocsv=args['output_csv_path']
    ldef=args['label_defs']
    bblab=args['bb_label']
    objects=args['object_labels']
    scl_w=args['scale_width']
    scl_h=args['scale_height']
    p2w=args['pad2width']
    p2h=args['pad2height']
    is_plot=args['plot']
    
    ldef=ldef.split(",")
    # bblab=bblab.split(",")
    if objects is not None:
        mlab=objects.split(",")
    else:
        mlab=[]

    crop_scale(idp,icsv,odp,ocsv,ldef,bblab,mlab,scl_w,scl_h,p2w,p2h,is_plot)
