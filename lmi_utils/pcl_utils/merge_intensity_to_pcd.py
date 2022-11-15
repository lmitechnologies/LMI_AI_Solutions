import os
import glob
import argparse
import logging
import numpy as np
import open3d


def merge_intensity_to_depth(path_pcd_intensity:str, path_pcd_depth:str):
    """merge intensity from a pcd_intensity file with another pcd_depth file
    Args:
        path_pcd_intensity (str): path to the intensity pcd file
        path_pcd_depth (str): path to the depth pcd file
    """
    data = open3d.io.read_point_cloud(path_pcd_intensity)
    pcd_intensity = np.asarray(data.points)
    xy_2_intensity = {(x,y):val for x,y,val in pcd_intensity}

    data = open3d.io.read_point_cloud(path_pcd_depth)
    pcd_depth = np.asarray(data.points)
    grayscale = []
    for x,y,d in pcd_depth:
        xy = (x,y)
        if xy in xy_2_intensity:
            grayscale += [xy_2_intensity[xy]]
        else:
            logging.error(f'warning: cannot find {xy} in corresponding pcd_intensity file: {os.path.basename(path_pcd_intensity)}')
            
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcd_depth)
    rgb = np.stack([grayscale]*3,axis=-1)
    pcd.colors = open3d.utility.Vector3dVector(rgb.astype(np.float)/255.0)
    return pcd


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input',required=True,help='the input folder, where it has "*_intensity.pcd" files')
    ap.add_argument('-o','--output',required=True,help='the output folder')
    args=ap.parse_args()
    
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    intensity_list = glob.glob(os.path.join(args.input,'*_intensity.pcd'))
    depth_list = [p.replace('_intensity.pcd','.pcd') for p in intensity_list]
    
    for p_intensity,p_depth in zip(intensity_list,depth_list):
        pcd = merge_intensity_to_depth(p_intensity,p_depth)
        fname = os.path.basename(p_depth).replace('.pcd','_with_rgb.pcd')
        logging.info(fname)
        open3d.io.write_point_cloud(os.path.join(args.output,fname), pcd)
        