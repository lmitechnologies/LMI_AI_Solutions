import numpy as np
import cv2
import random
import os
import json
import torch
import logging


logger = logging.getLogger().setLevel(logging.DEBUG)


def fit_array_to_size(im,W,H):
    is_rgb = 1
    if len(im.shape)==2:
        is_rgb = 0
    h_im,w_im=im.shape[:2]
    # pad or chop width
    if W >= w_im:
        pad_L=(W-w_im)//2
        pad_R=W-w_im-pad_L
        if is_rgb:
            im = np.pad(im, pad_width=((0,0),(pad_L,pad_R),(0,0)), mode='constant')
        else:
            im = np.pad(im, pad_width=((0,0),(pad_L,pad_R)), mode='constant')
    else:
        pad_L = (w_im-W)//2
        pad_R = w_im-W-pad_L
        if is_rgb:
            im = im[:,pad_L:-pad_R,:]
        else:
            im = im[:,pad_L:-pad_R]
        pad_L = -pad_L
    # pad or chop height
    if H >= h_im:
        pad_T=(H-h_im)//2
        pad_B=H-h_im-pad_T
        if is_rgb:
            im = np.pad(im, pad_width=((pad_T,pad_B),(0,0),(0,0)), mode='constant')
        else:
            im = np.pad(im, pad_width=((pad_T,pad_B),(0,0)), mode='constant')
    else:
        pad_T = (h_im-H)//2
        pad_B = h_im-H-pad_T
        if is_rgb:
            im = im[pad_T:-pad_B,:]
        else:
            im = im[pad_T:-pad_B]
        pad_T = -pad_T
    return im, pad_L, pad_T

        
        
def plot_one_box(box, img, mask=None, mask_threshold:float=0.0, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box and mask (optinal) on image img,
                 this function comes from YoLov5 project.
    param: 
        box:    a box likes [x1,y1,x2,y2]
        img:    a opencv image object in BGR format
        mask:   a binary mask for the box
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    if isinstance(box, list):
        box = np.array([x for x in box])
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
        
    x1,y1,x2,y2 = box.astype(int)
    c1, c2 = (x1, y1), (x2, y2)
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if mask is not None:
        # mask *= 255
        m = mask>mask_threshold
        blended = (0.4 * np.array(color,dtype=float) + 0.6 * img[m]).astype(np.uint8)
        img[m] = blended
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 4,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )



def revert_mask_to_origin(mask, operations:list):
    h,w = mask.shape[:2]
    mask2 = mask.copy()
    for operator in reversed(operations):
        h,w = mask2.shape[:2]
        if 'resize' in operator:
            r = operator['resize']
            nh,nw = int(h/r[0]), int(w/r[1])
            mask2 = cv2.resize(mask2,(nw,nh))
        if 'pad' in operator:
            pad = operator['pad']
            nw,nh = w-pad[0]*2,h-pad[1]*2
            mask2 = fit_array_to_size(mask2,nw,nh)
        # if 'stretch' in operator:
        #     s = operator['stretch']
        #     nx,ny = nx/s[0], ny/s[1]
    return mask2


def revert_to_origin(pts:np.ndarray, operations:list):
    """
    revert the points to original image coordinates
    This func reverts the operation in operations list IN ORDER.
    The operations list contains items as dictionary. The items are listed as follows: 
        1. <stretch: [stretch_ratio_x, stretch_ratio_y]>
        2. <pad: [pad_left_pixels, pad_top_pixels]> 
        3. <resize: [resize_ratio_x, resize_ratio_y]>
    args:
        pts: Nx2 or Nx4, where each row =(X_i,Y_i)
        operations : list of dict
    """

    def revert(x,y, operations):
        nx,ny = x,y
        for operator in reversed(operations):
            if 'resize' in operator:
                r = operator['resize']
                nx,ny = nx/r[0], ny/r[1]
            if 'pad' in operator:
                pad = operator['pad']
                nx,ny = nx-pad[0],ny-pad[1]
            if 'stretch' in operator:
                s = operator['stretch']
                nx,ny = nx/s[0], ny/s[1]
        return [max(nx,0),max(ny,0)]

    pts2 = []
    if isinstance(pts, list):
        pts = np.array(pts)
    for pt in pts:
        if len(pt)==0:
            continue
        if len(pt)==2:
            x,y = pt
            pts2.append(revert(x,y,operations))
        elif len(pt)==4:
            x1,y1,x2,y2 = pt
            pts2.append(revert(x1,y1,operations)+revert(x2,y2,operations))
        else:
            raise Exception(f'does not support pts neither Nx2 nor Nx4. Got shape: {pt.shape} with val: {pt}')
    return pts2



def apply_operations(pts:np.ndarray, operations:list):
    """
    apply operations to pts.
    The operations list contains each item as a dictionary. The items are listed as follows: 
        1. <stretch: [stretch_ratio_x, stretch_ratio_y]>
        2. <pad: [pad_left_pixels, pad_top_pixels]> 
        3. <resize: [resize_ratio_x, resize_ratio_y]>
    args:
        pts: Nx2 or Nx4, where each row =(X_i,Y_i)
        operations : list of dict
    """

    def apply(x,y, operations):
        nx,ny = x,y
        for operator in operations:
            if 'resize' in operator:
                r = operator['resize']
                nx,ny = nx*r[0], ny*r[1]
            if 'pad' in operator:
                pad = operator['pad']
                nx,ny = nx+pad[0],ny+pad[1]
            if 'stretch' in operator:
                s = operator['stretch']
                nx,ny = nx*s[0], ny*s[1]
        return [max(nx,0),max(ny,0)]

    pts2 = []
    if isinstance(pts, list):
        pts = np.array(pts)
    for pt in pts:
        if len(pt)==0:
            continue
        if len(pt)==2:
            x,y = pt
            pts2.append(apply(x,y,operations))
        elif len(pt)==4:
            x1,y1,x2,y2 = pt
            pts2.append(apply(x1,y1,operations)+apply(x2,y2,operations))
        else:
            raise Exception(f'does not support pts neither Nx2 nor Nx4. Got shape: {pt.shape} with val: {pt}')
    return pts2
    
    
def convert_key_to_int(dt):
    """
    convert the class map <id, class name> to integer class id
    """
    return {int(k):dt[k] for k in dt}  


def val_to_key(dt):
    return {dt[k]:k for k in dt}

    
def get_img_path_batches(batch_size, img_dir, fmt='png'):
    ret = []
    batch = []
    cnt_images = 0
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if name.find(f'.{fmt}')==-1:
                continue
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
            cnt_images += 1
    logger.info(f'loaded {cnt_images} files')
    if len(batch) > 0:
        ret.append(batch)
    return ret



def load_pipeline_def(filepath):
    with open(filepath) as f:
        dt_all = json.load(f)
        l = dt_all['configs_def']
        kwargs = {}
        for dt in l:
            kwargs[dt['name']] = dt['default_value']
    return kwargs