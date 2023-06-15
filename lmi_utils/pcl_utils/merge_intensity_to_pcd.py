import os
import glob
import argparse
import logging
import numpy as np
import open3d
import cv2


def merge_intensity_to_depth(path_pcd_intensity:str, path_pcd_depth:str, colormap='hot'):
    """merge intensity from a pcd_intensity file with another pcd_depth file
    Args:
        path_pcd_intensity (str): path to the intensity pcd file
        path_pcd_depth (str): path to the depth pcd file
    """
    data = open3d.io.read_point_cloud(path_pcd_depth)
    pcd_depth = np.asarray(data.points)
    
    data = open3d.io.read_point_cloud(path_pcd_intensity)
    pcd_intensity = np.asarray(data.points)
    
    if colormap in ['rainbow','hot','turbo']:
        grayscale = pcd_intensity[:,2].astype(np.uint8)
        if colormap == 'rainbow':
            rgb = cv2.applyColorMap(grayscale, colormap=cv2.COLORMAP_RAINBOW)
        elif colormap == 'hot':
            rgb = cv2.applyColorMap(grayscale, colormap=cv2.COLORMAP_HOT)
        elif colormap == 'turbo':
            rgb = cv2.applyColorMap(grayscale, colormap=cv2.COLORMAP_TURBO)
        xy_2_intensity = {}
        for i in range(pcd_intensity.shape[0]):
            x,y,_ = pcd_intensity[i,:]
            c = np.squeeze(rgb[i,:,:])
            xy_2_intensity[(x,y)] = c
    elif colormap=='grayscale':
        xy_2_intensity = {(x,y):[val]*3 for x,y,val in pcd_intensity}
    else:
        raise Exception(f'unrecognize colormap: {colormap}')

    # map color to depth x,y
    colors = []
    for x,y,d in pcd_depth:
        xy = (x,y)
        if xy in xy_2_intensity:
            colors += [xy_2_intensity[xy]]
        else:
            logging.error(f'warning: cannot find {xy} in corresponding pcd_intensity file: {os.path.basename(path_pcd_intensity)}')
            
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcd_depth)
    rgb = np.stack(colors,axis=0)
    pcd.colors = open3d.utility.Vector3dVector(rgb.astype(float)/255.0)
    return pcd


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input',required=True,help='the input folder, where it have depth pcd and intensity pcd files')
    ap.add_argument('-o','--output',required=True,help='the output folder')
    ap.add_argument('-c','--colormap',default='grayscale',help='the output colormap, "hot", "rainbow", "turbo", default is "grayscale"')
    args=ap.parse_args()
    
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    intensity_list = glob.glob(os.path.join(args.input,'*_intensity.pcd'))
    depth_list = [p.replace('_intensity.pcd','.pcd') for p in intensity_list]
    logging.info(f'using colormap: {args.colormap}')
    
    for p_intensity,p_depth in zip(intensity_list,depth_list):
        pcd = merge_intensity_to_depth(p_intensity,p_depth,args.colormap)
        fname = os.path.basename(p_depth).replace('.pcd','_with_rgb.pcd')
        logging.info(fname)
        open3d.io.write_point_cloud(os.path.join(args.output,fname), pcd)
        