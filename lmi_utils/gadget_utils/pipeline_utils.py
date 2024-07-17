import numpy as np
import cv2
import random
import os
import json
import torch
import logging
import glob
from torch.nn import functional as F


BLACK=(0,0,0)
TWO_TO_FIFTEEN = 2**15

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@torch.no_grad()
def resize_image(im, W=None, H=None, mode='bilinear'):
    """
    args: 
        im(np array | torch.tensor): the image of the shape (H,W) or (H,W,C)
        W(int): width
        H:(int): Height
        mode(str): 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'bilinear'
    """
    if W is None and H is None:
        return im
    
    # get the target width and height
    h,w = im.shape[:2]
    if W is None:
        W = int(w*H/h)
    elif H is None:
        H = int(h*W/w)
    
    # convert to tensor
    is_numpy = isinstance(im, np.ndarray)
    if is_numpy:
        im = torch.from_numpy(im)
    
    # deal with 1 channel image 
    one_channel = im.ndim==2
    if one_channel:
        im = im.unsqueeze(-1)
        
    im2 = F.interpolate(im.permute(2,0,1).unsqueeze(0).float(), size=(H,W), mode=mode)
    im2 = im2.squeeze(0).permute(1,2,0).to(torch.uint8)
    
    # back to 1 channel
    if one_channel:
        im2 = im2.squeeze(-1)
    
    return im2.numpy() if is_numpy else im2


@torch.no_grad()
def fit_im_to_size(im, W=None, H=None, value=0):
    """
    description:
        pad/crop the image to the size [W,H] with BLACK pixels
    arguments:
        im(np.array or torch.Tensor): the image of the shape (H,W) or (H,W,C)
        W(int): the target width. If None, the width will not be changed
        H(int): the target height. If None, the height will not be changed
        value(int): the value to pad
    return:
        im(torch.Tensor): the padded/cropped image 
        pad_l(int): number of pixels padded to left
        pad_r(int): number of pixels padded to right
        pad_t(int): number of pixels padded to top
        pad_b(int): number of pixels padded to bottom
    """
    
    if W is None and H is None:
        return im, 0, 0, 0, 0
    h,w = im.shape[:2]
    if W is None:
        W = w
    elif H is None:
        H = h
    
    is_numpy = isinstance(im, np.ndarray)
    if is_numpy:
        im = torch.from_numpy(im)

    # deal with 1 channel image
    one_channel = im.ndim==2
    if one_channel:
        im = im.unsqueeze(-1)

    # convert to CHW format    
    im = im.permute(2, 0, 1)

    # pad/crop width
    if W >= w:
        pad_L = (W - w) // 2
        pad_R = W - w - pad_L
        im = F.pad(im, (pad_L, pad_R, 0, 0), value=value)  
    else:
        pad_L = (w - W) // 2
        pad_R = w - W - pad_L
        im = im[:, :, pad_L:-pad_R]
        pad_L *= -1
        pad_R *= -1

    # pad/crop height
    if H >= h:
        pad_T = (H - h) // 2
        pad_B = H - h - pad_T
        im = F.pad(im, (0, 0, pad_T, pad_B), value=value)
    else:
        pad_T = (h - H) // 2
        pad_B = h - H - pad_T
        im = im[:, pad_T:-pad_B, :]
        pad_T *= -1
        pad_B *= -1

    # convert back to HWC format
    im = im.permute(1, 2, 0)
    
    # back to 1 channel
    if one_channel:
        im = im.squeeze(-1)

    if is_numpy:
        im = im.numpy()
    return im, pad_L, pad_R, pad_T, pad_B


def fit_array_to_size(im,W,H):
    """
    DEPRECATED
    description:
        pad/crop the image to the size [W,H] with BLACK pixels
    arguments:
        im(np array): the numpy array of a image
        W(int): the target width
        H(int): the target height
    return:
        im(np array): the padded/cropped image 
        pad_l(int): number of pixels padded to left
        pad_r(int): number of pixels padded to right
        pad_t(int): number of pixels padded to top
        pad_b(int): number of pixels padded to bottom
    """
    h_im,w_im=im.shape[:2]
    # pad or crop width
    if W >= w_im:
        pad_L=(W-w_im)//2
        pad_R=W-w_im-pad_L
        im=cv2.copyMakeBorder(im,0,0,pad_L,pad_R,cv2.BORDER_CONSTANT,value=BLACK)
    else:
        pad_L = (w_im-W)//2
        pad_R = w_im-W-pad_L
        im = im[:,pad_L:-pad_R]
        pad_L *= -1
        pad_R *= -1
    # pad or crop height
    if H >= h_im:
        pad_T=(H-h_im)//2
        pad_B=H-h_im-pad_T
        im=cv2.copyMakeBorder(im,pad_T,pad_B,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    else:
        pad_T = (h_im-H)//2
        pad_B = h_im-H-pad_T
        im = im[pad_T:-pad_B,:]
        pad_T *= -1
        pad_B *= -1
    return im, pad_L, pad_R, pad_T, pad_B


def uint16_to_int16(profile):
    """
    convert uint16 profile image to int16
    """
    if profile.dtype != np.uint16:
        raise Exception(f'dtype should be uint16, got {profile.dtype}')
    return profile.view(np.int16) + np.int16(-TWO_TO_FIFTEEN)


def profile_to_xyz(profile, resolution, offset):
    """
    convert profile image to 3d sensor space
    args:
        profile(np array): the profile image
        resolution(tuple): (x_resolution, y_resolution, z_resolution)
        offset(tuple): (x_offset, y_offset, z_offset)
    return:
        X(np array): the x coordinates in 3d space, same shape as profile
        Y(np array): the y coordinates in 3d space, same shape as profile
        Z(np array): the z coordinates in 3d space, same shape as profile
        mask(np array): the mask of the profile image to remove background
    """
    if profile.dtype != np.int16:
        raise Exception(f'profile.dtype should be int16, got {profile.dtype}')
    
    h,w = profile.shape[:2]
    x1,y1 = 0,0
    x2,y2 = w,h
    mask = profile != -TWO_TO_FIFTEEN
    xx,yy = np.meshgrid(np.arange(x1,x2), np.arange(y1,y2))
    X = offset[0] + xx * resolution[0]
    Y = offset[1] + yy * resolution[1]
    Z = offset[2] + profile*resolution[2]
    # xyz = np.stack((X[mask],Y[mask],Z[mask]), axis=-1)
    return X,Y,Z,mask


def pts_to_xyz(pts, profile, resolution, offset):
    """
    convert list of 2d pts to 3d sensor space
    args:
        pts(list): list of (x,y) points, with shape of Nx2
        profile(np array): the profile image
        resolution(tuple): (x_resolution, y_resolution, z_resolution)
        offset(tuple): (x_offset, y_offset, z_offset)
    """
    if profile.dtype != np.int16:
        raise Exception(f'profile.dtype should be int16, got {profile.dtype}')
    
    xyz = []
    for pt in pts:
        if len(pt)!=2:
            raise Exception(f'pts should be a list of (x,y) points, got {pt}')
        x,y = map(int,pt)
        nx = offset[0] + x * resolution[0]
        ny = offset[1] + y * resolution[1]
        nz = offset[2] + profile[y][x]*resolution[2]
        xyz += [[nx,ny,nz]]
    return np.array(xyz)


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


def plot_one_rbox(box, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding rotated bbox on image img
    param: 
        box:    a box likes [[x,y],[x,y],[x,y],[x,y]]
        img:    a opencv image object in BGR format
        mask:   a binary mask for the box
        color:  color to draw polygon, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    if len(box) != 4:
        raise Exception(f'box should be a list of 4 points, got {len(box)} points')
    if isinstance(box, list):
        box = np.array(box).astype(np.int32)
    
    
    cv2.polylines(img, [box], isClosed=True, color=color, thickness=tl)
        
    if label:
        highest_point = min(box, key=lambda point: point[1])
        text_position = (highest_point[0], highest_point[1] - 10)
        
        if text_position[1] < 0:  # If the text would be outside the image, move it below the lowest point instead
            lowest_point = max(box, key=lambda point: point[1])
            text_position = (lowest_point[0], lowest_point[1] + 20)
        
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        cv2.rectangle(img, text_position, (text_position[0] + t_size[0], text_position[1] - t_size[1] - 3), color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            text_position,
            0,
            tl / 4,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


@torch.no_grad()
def revert_mask_to_origin(mask, operations:list):
    """
    This func reverts a single mask image according to the operations list IN ORDER.
    The operations list contains items as dictionary. The items are listed as follows: 
        1. <pad: [pad_left_pixels, pad_right_pixels, pad_top_pixels, pad_bottom_pixels]> 
        2. <resize: [resized_w, resized_h, orig_w, orig_h]>
    """
    for operator in reversed(operations):
        if 'resize' in operator:
            _,_,nw,nh = operator['resize']
            mask = resize_image(mask,nw,nh)
        if 'pad' in operator:
            h,w = mask.shape[:2]
            pad_L,pad_R,pad_T,pad_B = operator['pad']
            nw,nh = w-pad_L-pad_R,h-pad_T-pad_B
            mask,_,_,_,_ = fit_im_to_size(mask,nw,nh)
    return mask


def revert_masks_to_origin(masks, operations:list):
    results = []
    if len(masks)==0:
        return results
    is_tensor = isinstance(masks[0], torch.Tensor)
    is_numpy = isinstance(masks, np.ndarray)
    for m in masks:
        results.append(revert_mask_to_origin(m, operations))
    if is_tensor:
        return torch.stack(results)
    return np.stack(results) if is_numpy else results


@torch.no_grad()
def revert_to_origin(pts, operations:list):
    """
    revert the points to original image space.
    This func executes operations in the REVERSED order.
    The operations list contains items as dictionary. The supported items are listed following: 
        1. <stretch: [stretch_ratio_x, stretch_ratio_y]>
        2. <pad: [pad_left, pad_right, pad_top, pad_bottom]> 
        3. <resize: [resized_w, resized_h, orig_w, orig_h]>
    args:
        pts: Nx2 or Nx4, where each row =(X_i,Y_i)
        operations : list of dict
    """
    is_tensor = isinstance(pts, torch.Tensor)
    is_numpy = isinstance(pts, np.ndarray)
    if not is_tensor:
        pts = torch.from_numpy(pts) if is_numpy else torch.as_tensor(pts)
    
    r,c = pts.shape
    if c!=2 and c!=4:
        raise Exception(f'supports pts Nx2 or Nx4. Got shape: {pts.shape}')
    for op in reversed(operations):
        if 'resize' in op:
            tw,th,orig_w,orig_h = op['resize']
            r = torch.tensor([tw/orig_w,th/orig_h],device=pts.device)
            if c==4:
                r = r.repeat(2).unsqueeze(0)
            pts = pts/r
        if 'pad' in op:
            pad_L,pad_R,pad_T,pad_B = op['pad']
            p = torch.tensor([pad_L,pad_T],device=pts.device)
            if c==4:
                p = p.repeat(2).unsqueeze(0)
            pts = pts - p
        if 'stretch' in op:
            s = torch.tensor(op['stretch'],device=pts.device)
            if c==4:
                s = s.repeat(2).unsqueeze(0)
            pts = pts/s
            
    pts = pts.round().clamp(min=0)
    if is_tensor:
        return pts
    return pts.numpy() if is_numpy else pts.tolist()


def apply_operations(pts:np.ndarray, operations:list):
    """
    apply operations to pts.
    The operations list contains each item as a dictionary. The items are listed as follows: 
        1. <stretch: [stretch_ratio_x, stretch_ratio_y]>
        2. <pad: [pad_left_pixels, pad_right_pixels, pad_top_pixels, pad_bottom_pixels]> 
        3. <resize: [resized_w, resized_h, orig_w, orig_h]>
    args:
        pts: Nx2 or Nx4, where each row =(X_i,Y_i)
        operations : list of dict
    """

    def apply(x,y, operations):
        nx,ny = x,y
        for operator in operations:
            if 'resize' in operator:
                tw,th,orig_w,orig_h = operator['resize']
                r = [tw/orig_w,th/orig_h]
                nx,ny = nx*r[0], ny*r[1]
            if 'pad' in operator:
                pad_L,pad_R,pad_T,pad_B = operator['pad']
                nx,ny = nx+pad_L,ny+pad_T
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


def get_gadget_img_batches(batch_size, profile_dir, intensity_dir, fmt='png'):
    profile_list = glob.glob(os.path.join(profile_dir,"*."+fmt))
    intensity_list = glob.glob(os.path.join(intensity_dir,"*."+fmt))

    profile_list.sort()
    intensity_list.sort()

    ret = []
    batch = []
    cnt_images = 0
    for profile, intensity in zip(profile_list, intensity_list):
        if len(batch) == batch_size:
            ret.append(batch)
            batch = []
        batch.append({"profile":profile, "intensity":intensity})
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
