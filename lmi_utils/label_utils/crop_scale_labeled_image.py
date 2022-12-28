import cv2
import numpy as np
from image_utils.img_resize import resize
import argparse

def make_border(img,p2h=None,p2w=None):
    '''
    Input image is padded on top and bottom to acheive p2h.
    Input images taller than p2h will be cropped from the top.
    '''

    (h0,w0)=img.shape[:2]
    print('[INFO] Make_border() input image dimensions: w=%2d,h=%2d' %(w0,h0))

    top_border=0
    bottom_border=0
    left_border=0
    right_border=0

    if p2h is not None:
        if h0>p2h:
            crop_rows=h0-p2h
            img=img[crop_rows:,:]
            top_border=-crop_rows
            bottom_border=top_border+p2h
            print('[INFO] Scaled image is too tall.  Clipping top %2d rows'%(crop_rows))
        else:
            top_border=(p2h-h0)//2
            bottom_border=p2h-h0-top_border
            img=cv2.copyMakeBorder(img,top_border,bottom_border,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
            print('[INFO] Adding top/bottom borders to maintain image shape.')
    if p2w is not None:
        if w0>p2w:
            crop_cols=w0-p2w
            img=img[:,crop_cols:]
            left_border=-crop_cols
            right_border=left_border+p2w
            print('[INFO] Scaled image is too wide.  Clipping left %2d rows'%(crop_cols))
        else:
            left_border=(p2w-w0)//2
            right_border=p2w-w0-left_border
            img=cv2.copyMakeBorder(img,0,0,left_border,right_border,cv2.BORDER_CONSTANT,value=[0,0,0])
            print('[INFO] Adding left/right borders to maintain image shape.')

    (h1,w1)=img.shape[:2]
    print('[INFO] New scaled image section shape: W=%2d, H=%2d.'%(w1,h1))
    return img,top_border,bottom_border,left_border,right_border


# bounding box: ((ulx,uly),(lrx,lry))
def crop_scale_labeled_image(image,boundingbox,masks=None,new_width=None,new_height=None,p2h=None,p2w=None):
    '''
    Crops ROI defined by bounding box.
    Then resizes the image and masks, keeping aspect ratio.
    Then adds/top bottom borders for fixed shape image.
    Exceptions when:
    -mask regions outside of bounding box.
    -resized region is taller than the new height.
    '''

    if masks is not None:
        if not masks:
            masks=None 

    # extract bounding box coordinates
    ulx=boundingbox[0][0]
    uly=boundingbox[0][1]
    lrx=boundingbox[1][0]
    lry=boundingbox[1][1]

    # crop image by bounding box
    cropped_image=image[uly:lry,ulx:lrx]
    new_image=cropped_image
    
    new_masks=[]
    # recenter masks in cropped image
    if masks is not None:  
        for mask in masks:
            x0=mask[:,0]
            y0=mask[:,1]
            x=x0-ulx
            y=y0-uly

            is_left=np.all(np.less(x0,ulx))
            is_top=np.all(np.less(y0,uly))
            is_right=np.all(np.greater(x0,lrx))
            is_bottom=np.all(np.greater(y0,lry))
            if (is_left or is_right or is_top or is_bottom):
                x[:]=np.NaN
                y[:]=np.NaN
            else:
                #crop left,top
                x[x<0]=0
                y[y<0]=0
                #crop right,bottom
                x[x>lrx-ulx]=lrx-ulx
                y[y>lry-uly]=lry-uly             
            new_masks.append(np.stack((x,y),axis=1))
        masks=new_masks

    # if new_masks<0:
    #     raise Exception('Mask regions are outside bounding box')

    # rescale image and masks    
    if (new_width is not None) or (new_height is not None):
        resized_image=resize(cropped_image,width=new_width, height=new_height)
        new_image=resized_image
        if masks is not None:
            (y0,x0)=cropped_image.shape[0:2]
            (y1,x1)=resized_image.shape[0:2]
            sclx=x1/x0
            scly=y1/y0
            new_masks=[]
            # scale masks based on image scaling
            for mask in masks:
                x=mask[:,0]*sclx
                # correct invalid mask points
                # if (np.any(x>new_width)) or (np.any(x<0)):
                #     print('[INFO] invalid mask point, setting to bounding box edge.')
                #     x[x>new_width]=0
                #     x[x<0]=0
                y=mask[:,1]*scly
                xy=np.stack((x,y),axis=1).astype(np.int32)
                new_masks.append(xy)
    
    # add borders and shift masks down
    if (p2h is not None) or (p2w is not None):
        (h,w)=new_image.shape[:2]
        new_image,top_border,_,left_border,_=make_border(new_image,p2h,p2w)
        for i,mask in enumerate(new_masks):
            mask[:,1]=mask[:,1]+top_border
            mask[:,1][mask[:,1]<top_border]=top_border
            mask[:,0]=mask[:,0]+left_border
            mask[:,0][mask[:,0]<left_border]=left_border
            new_masks[i]=mask
        
    return new_image, new_masks 


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--image',required=True,help='Base image file.')
    ap.add_argument('--bounding_box',required=True,help='Bounding box for crop.')
    ap.add_argument('--masks',default=None,help='Input masks.')
    ap.add_argument('--new_width',default=None,help='training to validation split.')
    ap.add_argument('--pad2h',default=None)
    ap.add_argument('--pad2w',default=None)

    args=vars(ap.parse_args())
    new_image,new_masks=crop_scale_labeled_image(args['image'],args['bouding_box'],args['masks'],args['new_width'],args['pad2h'],args['pad2w'])