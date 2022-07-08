#%% modules
import numpy as np 
import cv2
import imutils
import math
import os

#%% convert intensity pcd to png
def intensitypcd_2_png():
    pass

#%% get image contours
def getContours(img,minArea=200000,blur=(17,17),threshold=(20,150)):
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(grey,blur,0)
    T1=threshold[0]
    T2=threshold[1]
    canny=cv2.Canny(blurred,T1,T2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    dilated = cv2.dilate(canny, kernel)
    (contours,_) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntPrune=[]
    for (i,c) in enumerate(contours):
        area=cv2.contourArea(c)
        if area>minArea:
            print('Area = {}'.format(area))
            cntPrune.append(c)
    return cntPrune

#%% mask image by contour
def maskByContour(image,contour):
    mask=np.zeros(image.shape[:2],dtype='uint8') 
    cv2.drawContours(mask,contour,-1,(255,255,255),-1)
    masked=cv2.bitwise_and(image,image,mask=mask)
    return masked

#%% rotate by contour
def rotateByContour(image,contour):
    _,_,angle=cv2.fitEllipse(contour)
    M=cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #print('minBB angle ={}'.format(angle))
    #print('centerX = {}'.format(cx) + ' centerY = {}'.format(cy))
    rotated=imutils.rotate(image,-(180-angle),center=(cx,cy)) 
    return cx, cy, rotated

#%% get contour center
def getContCenter(contour):
    M=cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx,cy

#%% rotate image by minimum contour bounding box
def rotateByMinBB(image,contour):
    M=cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    rect=cv2.minAreaRect(contour)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    delx30=box[3][0]-box[0][0]
    dely30=box[3][1]-box[0][1]
    delx10=box[1][0]-box[0][0]
    dely10=box[1][1]-box[0][1]
    angle=np.arctan(dely30/delx30)*180/np.pi
    da=0
    #Check rotation for long edge or short edge, want long edge up
    mag30=np.sqrt(delx30*delx30+dely30*dely30)
    mag10=np.sqrt(delx10*delx10+dely10*dely10)
    if mag30>mag10:
        da+=90   
    rotated=imutils.rotate(image,angle+da,center=(cx,cy)) 
    return cx,cy,rotated

#%% crop image by contour bounding box 
def cropByBB(image,contour,delta=0):
    rect=cv2.minAreaRect(contour)
    #print('rect='+str(rect))
    box=cv2.boxPoints(rect)
    #print('box float='+str(box))
    box=np.int0(box)
    #print('box int='+str(box))
#    delx=box[3][0]-box[0][0]
#    dely=box[3][1]-box[0][1]
#    angle=np.arctan(dely/delx)*180/np.pi
#    print('angle ={}'.format(angle))
#    cv2.drawContours(image,[box],0,(0,0,255),2)
    (ymax,xmax)=image.shape[:2]
    xmin=0
    ymin=0
    
    xl=max(xmin,np.min(box[:,0]))+delta
    xr=min(xmax,np.max(box[:,0]))-delta
    yt=max(ymin,np.min(box[:,1]))+delta
    yb=min(ymax,np.max(box[:,1]))-delta
    
    newImage=image[yt:yb,xl:xr]
    return newImage

#%% extract box regions from JSON file
import json
def extract_UniformBox_ROI_from_JSON(json_file_path,input_image_dir_path,output_image_dir_path,render_defects=False):
    # computes the max of a list of lists
    def max_list_of_lists(myList):
        n=len(myList)
        myMax=[None]*n
        for i in range(n):
            myMax[i]=max(myList[i])
        return max(myMax)

    # parse the labeling data
    with open(json_file_path,'r') as read_file:
        defect_labels=json.load(read_file)

    # keys for each region
    keys=list(defect_labels.keys())

    # extract top left corner and box dimensions
    xx=[None]*len(keys)
    yy=[None]*len(keys)
    ww=[None]*len(keys)
    hh=[None]*len(keys)
    for i,key in enumerate(keys):
        regions=defect_labels[key]['regions']
        xj=[None]*len(regions)
        yj=[None]*len(regions)
        wj=[None]*len(regions)
        hj=[None]*len(regions)
        for j,region in enumerate(regions):
            xj[j]=regions[j]['shape_attributes']['x']
            yj[j]=regions[j]['shape_attributes']['y']
            wj[j]=regions[j]['shape_attributes']['width']
            hj[j]=regions[j]['shape_attributes']['height']
        xx[i]=xj
        yy[i]=yj
        ww[i]=wj
        hh[i]=hj
    # get maximum box dimension
    w_max=max_list_of_lists(ww)
    h_max=max_list_of_lists(hh)
    print('[INFO] maximum width ',w_max)
    print('[INFO] maximum height ',h_max)
    # compute square window dimension
    wd=max((w_max,h_max))
    print('[INFO] window dimension ',wd)

    # write defect images
    if not os.path.isdir(output_image_dir_path):
        os.mkdir(output_image_dir_path)
    for i,key in enumerate(keys):
        fname=input_image_dir_path+'/'+defect_labels[key]['filename']
        image=cv2.imread(fname)
        xi=xx[i] 
        yi=yy[i]
        wi=ww[i]
        hi=hh[i]
        for j in range(len(xi)):
            xj=xi[j]-(wd-wi[j])/2
            xj=math.floor(xj)
            yj=yi[j]-(wd-hi[j])/2
            yj=math.floor(yj)
            cropped=image[yj:yj+wd,xj:xj+wd]
            fcrop=output_image_dir_path+'/'+os.path.splitext(os.path.split(fname)[1])[0]+'_'+str(j)+'.png'
            cv2.imwrite(fcrop,cropped)
    if render_defects:
        for i,key in enumerate(keys):
            fname=input_image_dir_path+'/'+defect_labels[key]['filename']
            image=cv2.imread(fname)
            green=(0,255,0)
            xi=xx[i] 
            yi=yy[i]
            wi=ww[i]
            hi=hh[i]
            for j in range(len(xi)):
                xj=xi[j]-(wd-wi[j])/2
                xj=math.floor(xj)
                yj=yi[j]-(wd-hi[j])/2
                yj=math.floor(yj)
                cv2.rectangle(image,(xj,yj),(xj+wd,yj+wd),green,3)
            cv2.imshow('image',image)
            cv2.waitKey(5000)
    
    return wd

def align_and_crop(image_file_path,output_dir_path,minArea=40000,blur=(17,17),threshold=(20,150)):
    # import image
    image=cv2.imread(image_file_path)
    # add border around image that will enable outer contour extraction
    # expand top and bottom for vertical alignment
    shape_init=image.shape
    borderLR=50
    borderTB=borderLR
    if shape_init[1]>shape_init[0]:
        borderTB = (shape_init[1])+borderTB
    image= cv2.copyMakeBorder(image,borderTB,borderTB,borderLR,borderLR,cv2.BORDER_CONSTANT,value=(0,0,0))
    # extract outer contour
    contour=getContours(image,minArea,blur=blur,threshold=threshold)
    print('[INFO]: {0:2d} contours'.format(len(contour)))
    # rotate the image
    _,_,rotated=rotateByMinBB(image,contour[0])
    # get new contour
    contour=getContours(rotated,minArea,blur=blur,threshold=threshold)
    print('[INFO] {0:2d} contours'.format(len(contour)))
    # crop by bounding box
    cropped=cropByBB(rotated,contour[0],15)
    # save the cropped image
    if not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)
    fcrop=output_dir_path+'/'+os.path.splitext(os.path.split(image_file_path)[1])[0]+'_cropped.png'
    cv2.imwrite(fcrop,cropped)

def tile_image(image_file_path,tile_dir_path,window_dimension,steps_per_window):
    image=cv2.imread(image_file_path)
    wd=window_dimension
    # generate training images (will manually select negative regions for training)
    steps_per_window=2
    j_y=int(image.shape[0]//(wd/steps_per_window))
    i_x=int(image.shape[1]//(wd/steps_per_window))

    if not os.path.isdir(tile_dir_path):
        os.mkdir(tile_dir_path)

    for j in range(j_y-1):
        for i in range(i_x-1):
            wd_y=int(j*wd/steps_per_window)
            print('[INFO] wd_y=',wd_y)
            wd_x=int(i*wd/steps_per_window)
            print('[INFO] wd_x=',wd_x)
            windowed=image[wd_y:wd_y+wd,wd_x:wd_x+wd]
            print('[INFO] shape=',windowed.shape)
            fwin=tile_dir_path+'/'+os.path.splitext(os.path.split(image_file_path)[1])[0]+'_'+str(i)+'_'+str(j)+'.png'
            cv2.imwrite(fwin,windowed)

