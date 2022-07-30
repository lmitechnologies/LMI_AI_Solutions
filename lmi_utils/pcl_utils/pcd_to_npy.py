import glob
import os
import argparse
import numpy as np
from pcl_utils.point_cloud import PointCloud
import open3d


def convert_pcd_to_npy(path_in,path_out):

    files=glob.glob(os.path.join(path_in,'*.pcd'))

    pointcloud=PointCloud()
    hmax=[]
    hmin=[]
    for file in files:
        pointcloud.read_points(file,zmin=None,zmax=None,clip_mode=False)
        file_out=os.path.split(file)[1].replace('.pcd','.npy')
        print(f'[INFO] .npy output file: {file_out}')
        file_out=os.path.join(path_out,file_out)
        pointcloud.save_as_npy(file_out)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input_path',default='./')
    ap.add_argument('-o','--output_path',default='./')
    args=vars(ap.parse_args())
    path_in=args['input_path']
    path_out=args['output_path']

    convert_pcd_to_npy(path_in,path_out)