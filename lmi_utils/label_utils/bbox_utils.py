import numpy as np
import logging

logging.basicConfig()


def xyxy_to_xywh(x1,y1,x2,y2):
    return x1, y1, x2-x1+1, y2-y1+1
    
def xywh_to_xyxy(x,y,w,h):
    return x, y, x+w-1, y+h-1

def convert_from_ls(result):
    """convert annotations from label studio format to image pixel coordinate system

    Returns:
        tuple: x,y,w,h,angle, where angle is counterclockwise rotation angle in degree
    """
    value = result['value']
    w, h = result['original_width'], result['original_height']
    
    if not all([key in value for key in ['x', 'y', 'width', 'height']]):
        raise Exception('missing "x", "y", "width", or "height" in json file')
    
    # angle in degree
    angle = value['rotation'] if 'rotation' in value else 0
    
    # convert angle from anticlockwise to clockwise
    if angle>180:
        angle -= 360
        
    if angle>45 or angle<-45:
        logging.warning(f'found angle of {angle} out of constrains')
    return w * value['x'] / 100.0, h * value['y'] / 100.0, \
           w * value['width'] / 100.0, h * value['height'] / 100.0, \
           angle

def get_lst_bbox_to_xywh(x, y, w, h, img_width, img_height):
    pixel_x = x / 100.0 * img_width
    pixel_y = y / 100.0 * img_height
    pixel_width = w / 100.0 * img_width
    pixel_height = h / 100.0 * img_height
    return pixel_x, pixel_y, pixel_width, pixel_height


def rescale_oriented_bbox(result, original_size, new_size):
    bbox = result['value']
    pixel_x, pixel_y, pixel_width, pixel_height = get_lst_bbox_to_xywh(bbox['x'],bbox['y'], bbox['width'],bbox['height'],result['original_width'], result['original_height'])
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    new_x = pixel_x * scale_x
    new_y = pixel_y * scale_y
    new_width = pixel_width * scale_x
    new_height = pixel_height * scale_y
    return new_x, new_y, new_width, new_height

def convert_ls_obb_to_yolo(result):
    bbox = result['value']
    cx, cy = bbox["x"], bbox["y"]
    width, height = bbox["width"], bbox["height"]
    angle = np.deg2rad(bbox["rotation"])
    half_width, half_height = width / 2, height / 2
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    corners = []
    for dx, dy in [(-half_width, -half_height), (-half_width, half_height),
                   (half_width, half_height), (half_width, -half_height)]:
        x = cx + (dx * cos_angle - dy * sin_angle)
        y = cy + (dx * sin_angle + dy * cos_angle)
        corners += [int(x) / result['original_width'], int(y)/ result['original_height']]
    return corners

def rotate(x,y,w,h,angle=0.0,rot_center='up_left',unit='degree'):
    """rotate the bbox from [x,y,w,h] using the angle to a array of [4,2].
    
    Args:
        angle(float): the rotation angle in current unit
        rot_center(str): the rotation center, either 'up_left' or 'center'
        unit(str): the current unit, either 'degree' or 'radian'. defalt unit is 'degree'

    Returns:
        np.ndarray: 4x2
    """
    if unit=='degree':
        ANGLE = np.deg2rad(angle)
    elif unit=='radian':
        ANGLE = angle
    else:
        raise Exception('Does not recognize the unit other than "degree" and "radian"')
    points = [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
    if rot_center=='up_left':
        xc,yc = x,y
    elif rot_center=='center':
        xc,yc = np.mean(points, axis=0)
    else:
        raise Exception('Does not recognize the rotation center other than "up_left" and "center"')
    return np.array(
        [
            [
                xc + np.cos(ANGLE) * (px - xc) - np.sin(ANGLE) * (py - yc),
                yc + np.sin(ANGLE) * (px - xc) + np.cos(ANGLE) * (py - yc)
            ]
            for px, py in points
        ]
    ).astype(int)