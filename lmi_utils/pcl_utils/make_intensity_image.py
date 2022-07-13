import numpy as np
import open3d
import cv2
import xml.etree.ElementTree as ET
import argparse
import os
import glob

def make_intensity_image(input_path,output_path):
    pcddata=open3d.io.read_point_cloud(input_path)
    pcd=np.asarray(pcddata.points)
    #%% find unique value for x and y
    x=pcd[:,0]
    y=pcd[:,1]
    z=pcd[:,2]

    #%%
    xval=np.unique(x)
    xval=np.sort(xval)
    yval=np.unique(y)
    yval=np.sort(yval)

    #%% determine x and y support
    xpos=np.arange(len(xval))
    ypos=np.arange(len(yval))

    #dictionary mapping x,y coordinates to x,y support
    xmap=dict(zip(xval,xpos))
    ymap=dict(zip(yval,ypos))
    #%% generate height map
    img=np.zeros([len(xmap),len(ymap),3],dtype=np.uint8)
    for index in range(x.shape[0]):
        xi=xmap.get(x[index])
        yi=ymap.get(y[index])
        zi=z[index]                
        img[xi,yi]=zi

    #%%
    cv2.imwrite(output_path,img)

#%% ------------------------------- MAIN ---------------------------------------
if __name__ == "__main__":
    #options
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input',required=True)
    ap.add_argument('-o','--output',required=True)
    args=ap.parse_args()

    input_path=args.input
    output_path=args.output

    if not os.path.isdir(input_path):
        make_intensity_image(input_path,output_path)
    else:
        print(f'[INFO] converting directory of pcds.')
        files=glob.glob(os.path.join(input_path,'*.pcd'))
        for current_file in files:
            print(f'[INFO] Reading: {current_file}')
            fname=os.path.split(current_file)[1]
            fname=os.path.splitext(fname)[0]+'.png'
            fout=os.path.join(output_path,fname)
            print(f'[INFO] writing: {fout}')
            make_intensity_image(current_file,fout)


