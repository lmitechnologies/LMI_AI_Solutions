import colorsys
import logging
import random

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
        saturation = 0.5  # High saturation
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
    x, y = pt
    cv2.circle(img, (int(x), int(y)), radius, color, -1)
    if label:
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
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

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
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

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    pts = pts.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=tl)
    if label:
        c1 = (int(np.min(pts[:, :, 0])), int(np.min(pts[:, :, 1])))
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

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    xs, ys = list(map(round, xs)), list(map(round, ys))
    colors = np.array([color] * len(xs), dtype=img.dtype)
    if img[ys, xs].shape[0] == 0:
        logger.warning("Got an invalid polygon. Skip")
        return
    img[ys, xs] = cv2.addWeighted(img[ys, xs], 0.6, colors, 0.4, 0)

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


def plot_tile_grid(boxes, img, color=(220, 220, 220), alpha=0.7, line_thickness=None):
    """Outline a tile grid on img as semi-transparent dotted lines, in place.

    arguments:
        boxes: (N, 4) xyxy tile rects, e.g. from ``Tiler.tile_boxes()``
        img(np array): a opencv image object
        color(tuple): line color
        alpha(float): line opacity, 0 to 1
        line_thickness(int): the thickness of the line; dots and gaps are twice that long
    return:
        no return
    """
    boxes = np.asarray(boxes.detach().cpu() if hasattr(boxes, "detach") else boxes, dtype=float).reshape(-1, 4)
    if not len(boxes):
        return
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    h, w = img.shape[:2]
    # one dot pattern for the whole image, so edges shared by overlapping tiles land on the same dots
    dots_x = np.arange(w) % (4 * tl) < 2 * tl
    dots_y = (np.arange(h) % (4 * tl) < 2 * tl)[:, None]
    mask = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes.round().astype(int):
        for y in (y0, y1):
            mask[max(y - tl // 2, 0) : y - tl // 2 + tl, x0:x1] |= dots_x[x0:x1]
        for x in (x0, x1):
            mask[y0:y1, max(x - tl // 2, 0) : x - tl // 2 + tl] |= dots_y[y0:y1]
    # one blend per pixel, so crossing lines are no darker than single ones
    img[mask] = (img[mask] * (1 - alpha) + np.asarray(color) * alpha).round().astype(img.dtype)


def plot_boxes_by_group(boxes, img, groups, colors=None, line_thickness=None):
    """Plots boxes on image img, colored by a per-box group code, in place.

    arguments:
        boxes: (N, 4) xyxy
        img(np array): a opencv image object
        groups: (N,) integer code per box, e.g. a tile merge's ``merge_origin``
        colors(list): one color per code, cycled if short. Defaults to distinct hues, one per code seen.
        line_thickness(int): the thickness of the line
    return:
        no return
    """
    boxes = np.asarray(boxes.detach().cpu() if hasattr(boxes, "detach") else boxes, dtype=float).reshape(-1, 4)
    groups = np.asarray(groups.detach().cpu() if hasattr(groups, "detach") else groups, dtype=int).reshape(-1)
    if len(groups) != len(boxes):
        raise ValueError(f"plot_boxes_by_group: got {len(boxes)} boxes and {len(groups)} group codes")
    if not len(boxes):
        return
    colors = colors or get_distinct_colors(int(groups.max()) + 1)
    for box, g in zip(boxes, groups):
        plot_one_box(box, img, color=colors[int(g) % len(colors)], line_thickness=line_thickness)
