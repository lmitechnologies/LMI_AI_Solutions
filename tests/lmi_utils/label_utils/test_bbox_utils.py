import cv2
import numpy as np
import pytest

from lmi_utils.label_utils.bbox_utils import get_rotated_bbox, ls_box_to_pixels, ls_points_to_pixels, rotate


# --- Helper to generate rotated rectangles ---
def create_rotated_rect_points(center, size, angle):
    rect = (center, size, angle)
    pts = cv2.boxPoints(rect)
    return pts


@pytest.mark.parametrize("angle", [0, 30, 45, 90, -15])
@pytest.mark.parametrize("size", [(100, 50), (50, 100), (50, 50)])
def test_geometric_consistency(angle, size):
    """
    Verifies that the reconstructed rotated rectangle is the same as the original one.
    """
    center = (200, 200)
    pts = create_rotated_rect_points(center, size, angle)

    # Run function
    x, y, w, h, out_angle = get_rotated_bbox(pts)
    recon_corners = rotate(x, y, w, h, out_angle)

    # Get the ground truth box from OpenCV directly for comparison
    gt_rect = cv2.minAreaRect(pts)
    gt_corners = cv2.boxPoints(gt_rect)

    # sort the corners
    recon_corners = np.array(sorted(recon_corners, key=lambda p: (p[0], p[1])))
    gt_corners = np.array(sorted(gt_corners, key=lambda p: (p[0], p[1])))

    assert np.allclose(recon_corners, gt_corners), "Reconstructed corners do not match ground truth corners."


def test_pivot_logic():
    """
    Verifies that a horizontal box is returned unrotated, from its top-left corner.
    """
    # Create a box where the "top" is flat (0 degrees)
    pts = create_rotated_rect_points((100, 100), (40, 20), 0)

    x, y, w, h, angle = get_rotated_bbox(pts)
    assert (x, y, w, h, angle) == pytest.approx((80, 90, 40, 20, 0))


def _restated(rect):
    """The same rectangle, stated in the angle range the other OpenCV releases use."""
    (cx, cy), (w, h), angle = rect
    return (cx, cy), (h, w), angle + (-90 if angle > 0 else 90)


@pytest.mark.parametrize("angle", [0, 1, 30, 45, 89, 90, -15, -89])
def test_reconstruction_is_independent_of_the_minarearect_angle_range(angle, monkeypatch):
    """
    The reconstruction must survive whichever angle range the installed OpenCV reports.

    OpenCV has redefined minAreaRect's range across releases -- 4.5 through 4.12 return (0, 90], 4.13
    returns [-90, 0) -- and the same rectangle is describable in either. Reading the corners as if they came
    from the other range picks the wrong pivot, which reflects the box across itself.
    """
    pts = create_rotated_rect_points((200, 200), (100, 50), angle)

    def corners():
        return np.array(sorted(rotate(*get_rotated_bbox(pts)), key=lambda p: (p[0], p[1])))

    as_reported = corners()
    real_min_area_rect = cv2.minAreaRect
    monkeypatch.setattr(cv2, "minAreaRect", lambda points: _restated(real_min_area_rect(points)))

    assert np.allclose(corners(), as_reported)


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


def ls_result(img_w, img_h, **value):
    return {"original_width": img_w, "original_height": img_h, "value": value}


def test_ls_box_edges_truncate_from_the_true_edge():
    # x=10.6, w=9.8: the right edge is at 20.4, so x2 is 20, not int(10.6) + int(9.8) = 19
    result = ls_result(100, 100, x=10.6, y=5.0, width=9.8, height=10.0)
    assert ls_box_to_pixels(result) == (10, 5, 20, 15, 0)


def test_ls_pixels_survive_float_noise():
    # 530 / 1000 * 100 / 100 * 1000 evaluates just under 530
    x = 530 / 1000 * 100
    assert ls_box_to_pixels(ls_result(1000, 1000, x=x, y=x, width=x, height=x))[:4] == (530, 530, 1060, 1060)
    xs, ys = ls_points_to_pixels(ls_result(1000, 1000, points=[[x, x]]))
    assert (xs[0], ys[0]) == (530, 530)
