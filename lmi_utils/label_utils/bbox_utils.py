import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def xyxy_to_xywh(x1,y1,x2,y2):
    return x1, y1, x2-x1, y2-y1
    
def xywh_to_xyxy(x,y,w,h):
    return x, y, x+w, y+h

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
    
    return w * value['x'] / 100.0, h * value['y'] / 100.0, \
           w * value['width'] / 100.0, h * value['height'] / 100.0, \
           angle

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