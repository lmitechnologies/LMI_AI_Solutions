"""The training-time rotate must match the inference-time rotate op, or the model sees shifted data."""

import cv2
import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.ops.rotate import RotateConfig, RotateOperation, rotate_transform

W, H = 12, 8
QUARTER_TURNS = [0, 90, 180, 270, 360, -90]
ARBITRARY = [7.0, 30.0, 45.0, 123.5, 200.0, 359.0]


def _matrix(width, height, angle):
    """What dataset_rotate hands to cv2.warpAffine."""
    affine, new_w, new_h = rotate_transform(width, height, angle)
    return np.array(affine, dtype=np.float32).reshape(2, 3), new_w, new_h


def _unique_image():
    """Distinct value per pixel, so any dropped or duplicated pixel shows up as a set difference."""
    return np.arange(H * W, dtype=np.float32).reshape(H, W) + 1


def _op_forward(img, angle):
    out, meta = RotateOperation().forward([torch.from_numpy(img.copy())], RotateConfig(angle=angle))
    return out[0].numpy(), meta


def _op_apply(key, tensor, meta):
    return RotateOperation().apply_coords([{key: tensor}], meta)[0][key][0].numpy()


@pytest.mark.parametrize("angle", QUARTER_TURNS)
def test_quarter_turns_are_lossless(angle):
    M, new_w, new_h = _matrix(W, H, angle)
    img = _unique_image()
    rotated = cv2.warpAffine(img, M, (new_w, new_h))
    assert set(rotated.ravel().tolist()) == set(img.ravel().tolist())


@pytest.mark.parametrize("angle", QUARTER_TURNS)
def test_quarter_turn_images_match_the_op_exactly(angle):
    M, new_w, new_h = _matrix(W, H, angle)
    img = _unique_image()
    ours, _ = _op_forward(img, angle)
    assert np.array_equal(cv2.warpAffine(img, M, (new_w, new_h)), ours)


@pytest.mark.parametrize("angle", QUARTER_TURNS + ARBITRARY)
def test_canvas_size_matches_the_op(angle):
    _, new_w, new_h = _matrix(W, H, angle)
    _, meta = _op_forward(_unique_image(), angle)
    assert [new_w, new_h] == meta.dst_sizes[0]


@pytest.mark.parametrize("angle", QUARTER_TURNS + ARBITRARY)
def test_points_match_the_op(angle):
    pts = np.array([[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0], [3.5, 2.5]])
    M, _, _ = _matrix(W, H, angle)
    expected = (M[:, :2] @ pts.T).T + M[:, 2]

    _, meta = _op_forward(_unique_image(), angle)
    got = _op_apply("points", torch.tensor(pts, dtype=torch.float32).reshape(1, -1, 2), meta)
    assert got == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("angle", QUARTER_TURNS + ARBITRARY)
def test_corners_stay_inside_the_expanded_canvas(angle):
    """The canvas must round up far enough that no corner is clipped."""
    corners = np.array([[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]])
    M, new_w, new_h = _matrix(W, H, angle)
    moved = (M[:, :2] @ corners.T).T + M[:, 2]
    assert moved[:, 0].min() > -0.5 and moved[:, 1].min() > -0.5
    assert moved[:, 0].max() < new_w - 0.5 and moved[:, 1].max() < new_h - 0.5


@pytest.mark.parametrize("angle", [90, 270, 30.0])
def test_masks_match_the_op(angle):
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[2:6, 3:9] = 1
    M, new_w, new_h = _matrix(W, H, angle)
    expected = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST)

    _, meta = _op_forward(mask, angle)
    got = _op_apply("masks", torch.from_numpy(mask.copy()).unsqueeze(0), meta)
    assert np.array_equal(got, expected)


def test_positive_angle_is_clockwise():
    """A point on the +x axis must move down in y-down image coords."""
    M, _, _ = _matrix(W, H, 90)
    center = np.array([(W - 1) / 2, (H - 1) / 2])
    right_of_center = center + np.array([3.0, 0.0])
    moved = M[:, :2] @ right_of_center + M[:, 2]
    new_center = M[:, :2] @ center + M[:, 2]
    assert moved[1] > new_center[1] + 2.9
