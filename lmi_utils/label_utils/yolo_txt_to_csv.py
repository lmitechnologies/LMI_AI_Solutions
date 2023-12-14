import glob
import os
import numpy as np
import csv

def main(path_txt_files,path_csv,out_h,out_w,path_class_map=None,is_mask=True):
    paths=glob.glob(os.path.join(path_txt_files,'*.txt'))
    with open(path_csv,'w',newline='') as csv_file:
        labelWriter=csv.writer(csv_file,delimiter=';')
        for path_i in paths:
            img_file=os.path.split(path_i)[1].replace('.txt','.png')
            with open(path_i,'r') as txt_file:
                for line in txt_file:
                    words = line.strip().split()
                    class_type=words[0]
                    if is_mask:
                        xy=words[1:]
                        x_norm=xy[::2]
                        y_norm=xy[1::2]
                        x_float = [float(item) for item in x_norm]
                        y_float = [float(item) for item in y_norm]
                        x=(np.array(x_float)*out_w).astype(np.uint16)
                        y=(np.array(y_float)*out_h).astype(np.uint16)
                        labelWriter.writerow([img_file,'class_'+str(class_type),'1.0','polygon','x values']+list(x))
                        labelWriter.writerow([img_file,'class_'+str(class_type),'1.0','polygon','y values']+list(y))
                        
if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_txt_files', '-i', required=True, help='the path to images')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--out_h', type=int, required=True, help='output image height')
    ap.add_argument('--out_w', type=int, required=True, help='output image width')
    ap.add_argument('--path_class_map',default=None, help='path to class_map.json')
    ap.add_argument('--mask',action='store_true',help="set if mask.")
    args = vars(ap.parse_args())
    
    path_txt_files=args['path_txt_files']
    path_csv=args['path_csv']
    out_h=args['out_h']
    out_w=args['out_w']
    path_class_map=args['path_class_map']
    is_mask=args['mask']
    
    main(path_txt_files,path_csv,out_h,out_w,path_class_map=None,is_mask=True)