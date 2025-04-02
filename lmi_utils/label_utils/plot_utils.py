import cv2
import random
import numpy as np
import logging
import colorsys

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_distinct_colors(n):
    """Generate a list of distinct colors in RGB format.

    Args:
        n (int): Number of distinct colors to generate.

    Returns:
        list: A list of tuples representing RGB colors.
    """
    colors = []
    for i in range(n):
        hue = i / n  # Equally spaced hue values
        saturation = 0.9  # High saturation
        value = 0.9  # High value
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(255 * x) for x in rgb))
    return colors


def plot_one_pt(pt, img, color=None, label=None, radius=3, line_thickness=None):
    """
    description: Plots one point on image img,
                 this function comes from YoLov5 project.
    arguments:
        pt(list):
        img(np array):    a opencv image object in BGR format
        color(tuple):  color to draw rectangle, such as (0,255,0)
        label(str):  the class name
        line_thickness(int): the thickness of the line
    """
    color = color or [random.randint(0, 255) for _ in range(3)]
    x,y = pt
    cv2.circle(img, (int(x), int(y)), radius, color, -1)
    if label:
        tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  # line/font thickness
        tf = max(tl - 1, 1)
        cv2.putText(
            img,
            label,
            (int(x), int(y) - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    arguments:
        x(list):      a box likes [x1,y1,x2,y2]
        img(np array):    a opencv image object in BGR format
        color(tuple):  color to draw rectangle, such as (0,255,0)
        label(str):  the class name
        line_thickness(int): the thickness of the line
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
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
        
        
def plot_one_polygon(pts, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    arguments:
        pts(np array):      a numpy array of size [N,2]
        img(np array):    a opencv image object in BGR format
        color(tuple):  color to draw rectangle, such as (0,255,0)
        label(str):  the class name
        line_thickness(int): the thickness of the line
    return:
        no return
    """

    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    pts = pts.reshape(-1,1,2).astype(int)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=tl)
    if label:
        c1 = (int(np.min(pts[:,:,0])), int(np.min(pts[:,:,1])))
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
        
        
def plot_one_brush(xs, ys, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    arguments:
        xs(list): a list of x positons, where I is a binary image and I[x,y] = 1
        ys(list): a list of y positons, where I is a binary image and I[x,y] = 1
        img(np array): a opencv image object in BGR format
        color(tuple): color to draw rectangle, such as (0,255,0)
        label(str): the class name
        line_thickness(int): the thickness of the line
    return:
        no return
    """

    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    xs,ys = list(map(round,xs)),list(map(round,ys))
    colors = np.array([color]*len(xs),dtype=img.dtype)
    if img[ys,xs].shape[0]==0:
        logger.warning(f'Got an invalid polygon. Skip')
        return
    img[ys,xs] = cv2.addWeighted(img[ys,xs],0.6,colors,0.4,0)
    
    if label:
        c1 = (int(np.min(xs)), int(np.min(ys)))
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
