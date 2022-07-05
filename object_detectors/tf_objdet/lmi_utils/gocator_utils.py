import numpy as np
import open3d
import cv2
import xml.etree.ElementTree as ET
import argparse

def make_intensity_image(input_path,output_path):
    pcddata=open3d.read_point_cloud(input_path)
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
