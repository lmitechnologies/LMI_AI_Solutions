import numpy as np
import cv2
import random


def pad_crop_array_to_size(im,W,H):
    is_rgb = 1
    if len(im.shape)==2:
        is_rgb = 0
    h_im,w_im=im.shape[:2]
    # pad or chop width
    if W >= w_im:
        pad_L=(W-w_im)//2
        pad_R=W-w_im-pad_L
        if is_rgb:
            im = np.pad(im, pad_width=((0,0),(pad_L,pad_R),(0,0)))
        else:
            im = np.pad(im, pad_width=((0,0),(pad_L,pad_R)))
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
            im = np.pad(im, pad_width=((pad_T,pad_B),(0,0),(0,0)))
        else:
            im = np.pad(im, pad_width=((pad_T,pad_B),(0,0)))
    else:
        pad_T = (h_im-H)//2
        pad_B = h_im-H-pad_T
        if is_rgb:
            im = im[pad_T:-pad_B,:]
        else:
            im = im[pad_T:-pad_B]
        pad_T = -pad_T
    return im, pad_L, pad_T


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object in BGR format
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
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
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
        
        
def plot_results(boxes, image, pred_classes, result_scores):
    """plot detection results on the image
    Args:
        boxes (list): a list of boxes, each box is in [x1,y1,x2,y2] format.
        image (np.ndarray): the image will be annotated
        pred_classes (list): a list of pred classes
        result_scores (list): a list of scores corresponding to the pred_classes
    """
    for j in range(len(boxes)):
        plot_one_box(
            boxes[j],
            image,
            label="{}:{:.2f}".format(
                pred_classes[j], result_scores[j]
            )
        )


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
        operations : Nx2
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
        operations : Nx2
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

def bbox_resize(box,  W, H, box_definition='TF_OD'):
        """
        DESCRIPTION:
            bbox_resize() generate absolute coordinates for normalized bounding box
    
        ARGUMENTS:
            box: normalized bounding box coordinates (0->1) (Y_ul, X_ul, Y_lr, X_lr)
            W: image width
            H: image height
            box_definition: TF_OD or YOLO
        
        RETURNS:
            tuple including: (startYnew, startXnew, endYnew, endXnew) where each new coordinate is scaled pixel coordinates
        """
        if box_definition=='TF_OD':
            (startY,startX,endY,endX)=box
        elif box_definition=='YOLO':
            (startX,startY,endX,endY)=box 
        startXnew = int(startX*W)
        startYnew = int(startY*H)
        endXnew = int(endX*W)
        endYnew = int(endY*H)
        
        return (startYnew, startXnew, endYnew, endXnew)