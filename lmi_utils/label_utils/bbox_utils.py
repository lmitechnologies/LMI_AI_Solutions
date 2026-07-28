import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def xyxy_to_xywh(x1, y1, x2, y2):
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h


def convert_from_ls(result):
    """convert annotations from label studio format to image pixel coordinate system

    Returns:
        tuple: x,y,w,h,angle, where angle is counterclockwise rotation angle in degree
    """
    value = result["value"]
    w, h = result["original_width"], result["original_height"]

    if not all([key in value for key in ["x", "y", "width", "height"]]):
        raise Exception('missing "x", "y", "width", or "height" in json file')

    # angle in degree
    angle = value["rotation"] if "rotation" in value else 0

    return (
        w * value["x"] / 100.0,
        h * value["y"] / 100.0,
        w * value["width"] / 100.0,
        h * value["height"] / 100.0,
        angle,
    )


def rotate(x, y, w, h, angle=0.0, rot_center="up_left", unit="degree"):
    """rotate the bbox from [x,y,w,h] using the angle to a array of [4,2].

    Args:
        angle(float): the rotation angle in current unit
        rot_center(str): the rotation center, either 'up_left' or 'center'
        unit(str): the current unit, either 'degree' or 'radian'. defalt unit is 'degree'

    Returns:
        np.ndarray: 4x2 of floats. Callers that rasterize the corners must round them to int themselves.
    """
    if unit == "degree":
        ANGLE = np.deg2rad(angle)
    elif unit == "radian":
        ANGLE = angle
    else:
        raise Exception('Does not recognize the unit other than "degree" and "radian"')
    points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    if rot_center == "up_left":
        xc, yc = x, y
    elif rot_center == "center":
        xc, yc = np.mean(points, axis=0)
    else:
        raise Exception('Does not recognize the rotation center other than "up_left" and "center"')
    return np.array(
        [
            [
                xc + np.cos(ANGLE) * (px - xc) - np.sin(ANGLE) * (py - yc),
                yc + np.sin(ANGLE) * (px - xc) + np.cos(ANGLE) * (py - yc),
            ]
            for px, py in points
        ]
    )


def get_rotated_bbox(pts: np.ndarray) -> list:
    """Get the rotated bbox from polygon points.

    The result is what ``rotate`` consumes: the pivot is the corner ``rotate`` turns the box about, and
    ``angle`` measures the box's own width axis from that pivot. Both are read off the corners rather than
    from ``minAreaRect``'s angle, whose range OpenCV has redefined between releases (4.5 and 4.13 disagree on
    its sign); a pivot chosen for one convention places the box on the wrong side of itself under the other.

    Args:
        pts (np.ndarray): Polygon points in shape [N, 2]

    Returns:
        list: [x, y, w, h, angle], where (x,y) is the pivot of the rotated bbox
    """
    bbox = np.array(pts, dtype=np.float32)
    box_points = cv2.boxPoints(cv2.minAreaRect(bbox))

    # the pivot is the topmost corner (minimum y, then minimum x if tied)
    idx = np.lexsort((box_points[:, 0], box_points[:, 1]))[0]
    x, y = (float(v) for v in box_points[idx])

    # Of the two edges meeting at the pivot, the width axis is the one the height axis follows under a
    # quarter turn in the direction `rotate` turns, which is the positive cross product in image coordinates
    width_axis = box_points[(idx + 1) % 4] - box_points[idx]
    height_axis = box_points[idx - 1] - box_points[idx]
    cross = width_axis[0] * height_axis[1] - width_axis[1] * height_axis[0]
    if cross < 0:
        width_axis, height_axis = height_axis, width_axis

    w = float(np.linalg.norm(width_axis))
    h = float(np.linalg.norm(height_axis))
    angle = float(np.degrees(np.arctan2(width_axis[1], width_axis[0])))

    return [x, y, w, h, angle]
