import cv2
import numpy as np
import pytest
from label_utils.bbox_utils import get_rotated_bbox


# --- Helper to generate rotated rectangles ---
def create_rotated_rect_points(center, size, angle):
    rect = (center, size, angle)
    pts = cv2.boxPoints(rect)
    return pts


@pytest.mark.parametrize("angle", [0, 30, 45, 90, -15])
@pytest.mark.parametrize("size", [(100, 50), (50, 100), (50, 50)])
def test_geometric_consistency(angle, size):
    """
    Verifies that the returned (x,y) is mathematically a corner of the
    box defined by the returned (w,h, angle).
    """
    center = (200, 200)
    pts = create_rotated_rect_points(center, size, angle)

    # Run function
    x, y, w, h, out_angle = get_rotated_bbox(pts)

    # Get the ground truth box from OpenCV directly for comparison
    gt_rect = cv2.minAreaRect(pts)
    gt_corners = cv2.boxPoints(gt_rect)

    # Check: Is the returned (x,y) one of the true corners?
    distances = [np.linalg.norm(c - np.array([x, y])) for c in gt_corners]
    min_dist = min(distances)

    assert min_dist < 1e-4, f"Returned point ({x},{y}) is not a corner of the fitted box."


def test_top_left_logic():
    """
    Verifies that the returned point is the 'Top-right' and angle is 90 degrees when the box is horizontal.
    """
    # Create a box where the "top" is flat (0 degrees)
    pts = create_rotated_rect_points((100, 100), (40, 20), 0)

    x, y, w, h, angle = get_rotated_bbox(pts)
    assert angle == pytest.approx(90, abs=0.01)
    assert x == pytest.approx(120, abs=0.01)
    assert y == pytest.approx(90, abs=0.01)


def test_diamond_shape_top_vertex():
    """
    Test a 45 degree rotation (Diamond shape).
    The 'Top-Left' logic (Min Y) should pick the top-most vertex.
    """
    # Diamond centered at 100,100
    pts = create_rotated_rect_points((100, 100), (50, 50), 45)

    x, y, w, h, angle = get_rotated_bbox(pts)

    # The top-most point of a diamond is the one with smallest Y
    gt_rect = cv2.minAreaRect(pts)
    corners = cv2.boxPoints(gt_rect)
    min_y_in_corners = np.min(corners[:, 1])

    assert y == pytest.approx(min_y_in_corners, abs=1e-4), "Did not return the top-most vertex (smallest Y) for a diamond shape"


def test_area_conservation():
    """
    The area of the returned w*h should match the area of the input box
    (within float tolerance).
    """
    w_in, h_in = 60, 30
    pts = create_rotated_rect_points((100, 100), (w_in, h_in), 15)

    x, y, w_out, h_out, angle = get_rotated_bbox(pts)

    area_in = w_in * h_in
    area_out = w_out * h_out

    assert area_out == pytest.approx(area_in, rel=1e-3)


def test_input_resilience():
    """
    Ensure it handles list input vs numpy array input and integer vs float.
    """
    # Integer List
    pts_list = [[10, 10], [20, 10], [20, 20], [10, 20]]
    x, y, w, h, a = get_rotated_bbox(pts_list)
    assert w > 0 and h > 0

    # Float Numpy Array
    pts_float = np.array(pts_list, dtype=np.float32)
    x2, y2, w2, h2, a2 = get_rotated_bbox(pts_float)

    assert x == pytest.approx(x2)
    assert y == pytest.approx(y2)
