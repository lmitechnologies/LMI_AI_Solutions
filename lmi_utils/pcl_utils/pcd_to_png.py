import glob
import os
import argparse
import numpy as np
from pcl_utils.point_cloud import PointCloud


def convert_pcd_to_png(path_in,path_out):

    files=glob.glob(os.path.join(path_in,'*.pcd'))

    pointcloud=PointCloud()
    hmax=[]
    hmin=[]
    for file in files:
        pointcloud.read_points(file,zmin=None,zmax=None,clip_mode=False)
        pointcloud.convert_points_to_image(color_mapping='gray',contrast_enhancement=True,zmin_color=None,zmax_color=None)
        file_out=os.path.split(file)[1]
        print(f'[INFO] File: {file_out}, hmin: {pointcloud.zmin}, hmax: {pointcloud.zmax}')
        hmax.append(pointcloud.zmax)
        hmin.append(pointcloud.zmin)
        file_out=os.path.splitext(file_out)[0]
        file_out=os.path.join(path_out,file_out+'.png')
        pointcloud.save_img(file_out)

    hmax=np.asarray(hmax)
    hmin=np.asarray(hmin)
    print('------')
    print(f'[INFO] Hmin={hmin.min()}, Hmax={hmax.max()}')
    print('------')


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input_path',default='./')
    ap.add_argument('-o','--output_path',default='./')
    args=vars(ap.parse_args())
    path_in=args['input_path']
    path_out=args['output_path']

    convert_pcd_to_png(path_in,path_out)