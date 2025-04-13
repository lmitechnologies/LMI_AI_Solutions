import cv2
import os
import numpy as np
import logging

#LMI packages
from dataset_utils.representations import  AnnotationType
from dataset_utils.mask_encoder import mask2rle
from label_utils.bbox_utils import rotate, get_rotated_bbox


logger = logging.getLogger(__name__)


def order_points(pts):
    """
    Orders 4 points in the order:
      top-left, top-right, bottom-right, bottom-left.

    Args:
        pts (np.array): A (4, 2) array of points.

    Returns:
        np.array: A (4, 2) array of ordered points.
    """
    # Initialize a list of coordinates that will be ordered.
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference
    # (y - x), whereas the bottom-left will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def rotate_bbox_corners(corners, M):
    """
    Rotates the corners of a bounding box using the affine transformation
    matrix M and sorts the resulting corners in the order:
    top-left, top-right, bottom-right, bottom-left.

    Args:
        corners (array-like): A (4,2) array (or list) of bounding box corners.
        M (np.array): A 2x3 affine transformation matrix (e.g., from cv2.getRotationMatrix2D).

    Returns:
        np.array: A (4,2) array of the rotated and ordered bounding box corners.
    """
    # Convert corners to a numpy array of type float32.
    corners = np.array(corners, dtype="float32")
    
    # Convert corners to homogeneous coordinates by appending a column of ones.
    ones = np.ones((corners.shape[0], 1), dtype="float32")
    corners_hom = np.hstack([corners, ones])  # shape (4, 3)
    
    # Apply the affine transformation to each corner.
    rotated_corners = np.dot(M, corners_hom.T).T  # shape (4, 2)
    
    # Order the rotated corners.
    ordered_corners = order_points(rotated_corners)
    
    return ordered_corners.astype(np.float32)

def rotate_dataset(dataset, images,angle,counter_clockwise=False):
    """
    Rotate annotations clockwise or counterclockwise
    """
    rotated_images = {}
    if counter_clockwise is False:
        angle = -angle
    for file in dataset.files:
        
        logger.debug(f'rotating {os.path.basename(file.path)} with angle {angle} degrees counter_clockwise : {counter_clockwise}')
        
        img = images[file.path]
        
        # rotate the image
        height, width = img.shape[:2]
        file.height = height
        file.width = width
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Calculate the sine and cosine (i.e., the rotation components)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])

        # Compute new bounding dimensions of the image
        new_width = int((height * abs_sin) + (width * abs_cos))
        new_height = int((height * abs_cos) + (width * abs_sin))

        # Adjust the rotation matrix to account for the translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))
        
        
        for annot in file.annotations:
            if annot.type == AnnotationType.BOX:
                x_min, y_min, x_max, y_max, theta = annot.value.coords()
                if theta != 0:
                    corners = rotate(x=x_min, y=y_min, w=x_max-x_min, h=y_max-y_min, angle=theta,rot_center='up_left')
                    corners = rotate_bbox_corners(corners, rotation_matrix)
                    x, y, w, h, theta = get_rotated_bbox(corners)
                    x_min, y_min, x_max, y_max = float(x), float(y), float(x+w), float(y+h)
                    annot.value.x_min, annot.value.y_min = x_min, y_min
                    annot.value.x_max, annot.value.y_max = x_max, y_max
                    annot.value.angle = theta
                    
                    # annot.value.theta = theta
                else:
                    corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
                    corners = rotate_bbox_corners(corners, rotation_matrix)
                    x_min, y_min, x_max, y_max = corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()
                    annot.value.x_min, annot.value.y_min = float(x_min), float(y_min)
                    annot.value.x_max, annot.value.y_max = float(x_max), float(y_max)
                    
                
            elif annot.type == AnnotationType.POLYGON:
                points = annot.value.to_numpy()
                for i in range(len(points)):
                    points[i] = np.dot(rotation_matrix[:, :2], [points[i][0], points[i][1]]) + rotation_matrix[:, 2]
                annot.value.points = points.astype(float).tolist()
            
            elif annot.type == AnnotationType.KEYPOINT:
                x, y = annot.value.x, annot.value.y
                new_x, new_y = np.dot(rotation_matrix[:, :2], [x, y]) + rotation_matrix[:, 2]
                annot.value.x, annot.value.y = new_x.astype(float), new_y.astype(float)
            
            elif annot.type == AnnotationType.MASK:
                mask = annot.value.to_numpy(h=height, w=width)
                mask = cv2.warpAffine(mask, rotation_matrix, (new_width, new_height))
                annot.value.mask = mask2rle(mask)
            
            else:
                logging.warning(f'unsupported annotation type: {annot.type}')

        # update the file dimensions
        file.height, file.width = rotated_img.shape[:2]
        rotated_images[file.path] = rotated_img
    return rotated_images, dataset