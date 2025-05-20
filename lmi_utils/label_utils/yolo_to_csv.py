import cv2
import numpy as np
import csv
import yaml
import pathlib

from label_utils.bbox_utils import get_rotated_bbox


def xywhn2xyxy(xc, yc, w, h, im_w, im_h):
    """convert normallized xywh to xyxy format
    """
    xc = xc * im_w
    yc = yc * im_h
    w = w * im_w
    h = h * im_h
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return x1, y1, x2, y2


def main(path_txt_files,path_imgs,path_csv,yaml_obj,model_type):
    paths = pathlib.Path(path_txt_files).glob('*.txt')
    path_imgs = pathlib.Path(path_imgs)
    names = yaml_obj['names']
    if model_type=='keypoint':
        kpt_shape = yaml_obj['kpt_shape']
        print(f'keypoint shape: {kpt_shape}')
    
    with open(path_csv,'w',newline='') as csv_file:
        labelWriter=csv.writer(csv_file,delimiter=';')
        for path_i in paths:
            # find the corresponding image
            key = path_i.stem
            img_file = None
            for img_file_i in path_imgs.glob(f'{key}.*'):
                img_file = img_file_i
                break
            if img_file is None:
                raise Exception(f'{key} not found in {path_imgs}')
            im = cv2.imread(str(img_file))
            out_h,out_w = im.shape[:2]
            
            with open(path_i,'r') as txt_file:
                for line in txt_file:
                    words = line.strip().split()
                    class_index=int(words[0])
                    if model_type=='mask':
                        xy=words[1:]
                        x_norm=xy[::2]
                        y_norm=xy[1::2]
                        x_float = [float(item)*out_w for item in x_norm]
                        y_float = [float(item)*out_h for item in y_norm]
                        x = np.array(x_float).astype(int)
                        y = np.array(y_float).astype(int)
                        labelWriter.writerow([img_file.name,names[class_index],'1.0','polygon','x values']+x.tolist())
                        labelWriter.writerow([img_file.name,names[class_index],'1.0','polygon','y values']+y.tolist())
                    elif model_type=='keypoint':
                        xc,yc,w,h = [float(x) for x in words[1:5]]
                        x1,y1,x2,y2 = xywhn2xyxy(xc,yc,w,h,out_w,out_h)
                        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                        labelWriter.writerow([img_file.name,names[class_index],'1.0','rect','upper left',x1,y1,'angle',0])
                        labelWriter.writerow([img_file.name,names[class_index],'1.0','rect','lower right',x2,y2,'angle',0])
                        
                        kpts = np.array([float(x) for x in words[5:]])
                        kpt_dim = kpt_shape[1] 
                        kpts = kpts.reshape(-1, kpt_dim)
                        kpts[:, 0] = kpts[:, 0] * out_w
                        kpts[:, 1] = kpts[:, 1] * out_h
                        kpts = kpts.astype(int)
                        for i in range(kpts.shape[0]):
                            labelWriter.writerow([img_file.name,'kp','1.0','keypoint','x value', kpts[i, 0]])
                            labelWriter.writerow([img_file.name,'kp','1.0','keypoint','y value', kpts[i, 1]])
                    elif model_type=='obb':
                        xy = words[1:]
                        x_norm = xy[::2]
                        y_norm = xy[1::2]
                        x_float = [float(x)*out_w for x in x_norm]
                        y_float = [float(y)*out_h for y in y_norm]
                        xy = np.array([[x,y] for x,y in zip(x_float,y_float)]).astype(int)
                        x1,y1,w,h,a = get_rotated_bbox(xy)
                        x2,y2 = x1+w,y1+h
                        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                        labelWriter.writerow([img_file.name,names[class_index],'1.0','rect','upper left',x1,y1,'angle',a])
                        labelWriter.writerow([img_file.name,names[class_index],'1.0','rect','lower right',x2,y2,'angle',a])
                    else:
                        # bbox format
                        # class_index, xc, yc, w, h
                        xc,yc,w,h = [float(x) for x in words[1:5]]
                        x1,y1,x2,y2 = xywhn2xyxy(xc,yc,w,h,out_w,out_h)
                        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                        labelWriter.writerow([img_file.name,names[class_index],'1.0','rect','upper left',x1,y1,'angle',0])
                        labelWriter.writerow([img_file.name,names[class_index],'1.0','rect','lower right',x2,y2,'angle',0])
                        
                    
                        
if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_txt_files', '-i', required=True, help='the path to yolo txt files')
    ap.add_argument('--path_imgs', '-p', required=True, help='the path to images')
    ap.add_argument('--path_yaml','-y', required=True, help='the path to a dataset yaml file')
    ap.add_argument('--path_csv', '-o', default='labels.csv', help='[optinal] the path of a output csv file, default="labels.csv"')
    ap.add_argument('--mask',action='store_true',help="set if mask.")
    ap.add_argument('--pose',action='store_true',help="set if keypoint.")
    ap.add_argument('--obb',action='store_true',help="set if oriented bbox.")
    args = ap.parse_args()
    
    yaml_obj = yaml.safe_load(open(args.path_yaml, 'r', encoding='UTF-8'))
    # names = obj['names']
    model_type='bbox'
    if args.mask:
        model_type='mask'
    if args.pose:
        model_type='keypoint'
    if args.obb:
        model_type='obb'
    
    main(args.path_txt_files,args.path_imgs,args.path_csv,yaml_obj,model_type)