#%%
import glob
import os
import pcl_utils.point_cloud as pcloud

import matplotlib.pyplot as plt
import scipy.interpolate as interp
import numpy as np

#%%
def gensurface(input_path, output_path, suffix, color_option,zmin=None,zmax=None,Wout=None,Hout=None):
    # check for output dir
    if os.path.isdir(output_path):
        pass
    else:
        os.mkdir(output_path)

    current_cloud=pcloud.PointCloud() 
    if suffix is not None:
        fname_glob='*'+suffix+'.pcd'
    else:
        fname_glob='*.pcd'
    files = glob.glob(os.path.join(input_path,fname_glob))

    H,W,minZ,maxZ =[],[],[],[]
    for file in files:
        current_cloud.read_points(file)
        
        current_minz=np.min(current_cloud.z)
        current_maxz=np.max(current_cloud.z)
        
        minZ.append(current_minz)
        maxZ.append(current_maxz)

        if zmin is None:
            zmin_col=current_minz
        else:
            zmin_col=zmin

        if zmax is None:
            zmax_col=current_maxz
        else:
            zmax_col=zmax

        # added to clip noisy parts
        current_cloud.z[current_cloud.z<zmin_col]=zmin_col
        print(f'[INFO] {file}:z less than {zmin_col} being replaced by {zmin_col}')

        fname_base=os.path.split(file)[1]
        if color_option=='rgb':    
            fname_png=os.path.splitext(fname_base)[0]+'_rgb.png' 
            current_cloud.convert_points_to_color_image(normalize=False,rgb_mapping='rgb',contrast_enhancement=False,zmin=zmin_col,zmax=zmax_col)
        elif color_option=='rainbow':    
            fname_png=os.path.splitext(fname_base)[0]+'_rainbow.png' 
            current_cloud.convert_points_to_color_image(normalize=False,rgb_mapping='rainbow',contrast_enhancement=True,zmin=zmin_col,zmax=zmax_col)
        elif color_option=='gray':
            fname_png=os.path.splitext(fname_base)[0]+'_gray.png' 
            current_cloud.convert_points_to_grayscale_image(normalize=False,contrast_enhancement=True,zmin=zmin_col,zmax=zmax_col)
        else:
            raise Exception('Invalid surface option.  Choose rgb, rainbow, gray.')

        outfile=os.path.join(output_path,fname_png)
        height,width,channel=current_cloud.img.shape
        H.append(height)
        W.append(width)
        if (Hout is not None) and (Wout is not None):
            current_cloud.pad(int(Hout),int(Wout))
        current_cloud.save_img(outfile)
    
    print(f'[INFO] min z: {np.min(minZ)}')
    print(f'[INFO] max z: {np.max(maxZ)}')
    print(f'[INFO] W max: {np.max(W)}')
    print(f'[INFO] H max: {np.max(H)}')

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_path',required=True,help='path to directory with input .pcd clouds')
    ap.add_argument('--output_path',required=True,help='path to directory with output .png images')
    ap.add_argument('--file_suffix',required=False, default=None,help='file suffix filter')
    ap.add_argument('--color_option',required=False, default='rgb',help='color option: rgb/gray/rainbow')
    ap.add_argument('--zmin',required=False, default=None,help='color map min')
    ap.add_argument('--zmax',required=False, default=None,help='color map max')
    ap.add_argument('--W',required=False, default=None,help='image width')
    ap.add_argument('--H',required=False, default=None,help='image height')
    args=vars(ap.parse_args())
    input_cloud_path=args['input_path']
    output_cloud_path=args['output_path']
    file_suffix=args['file_suffix']
    color_option=args['color_option']
    zmin=args['zmin']
    zmax=args['zmax']
    Wout=args['W']
    Hout=args['H']
    if zmin is not None:
        zmin=int(zmin)
    if zmax is not None:
        zmax=int(zmax)

    gensurface(input_cloud_path,output_cloud_path,file_suffix,color_option,zmin,zmax,Wout,Hout)