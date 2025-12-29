import numpy as np
import torch
from typing import Tuple
from numba import njit
from numba.np.extensions import cross2d
import torch.nn.functional as F
import cv2

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@torch.jit.script
def rescale_mask_func(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


@torch.jit.script
def rescale_masks(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    image_shape: Tuple[int, int],
    threshold: float = 0.5,
):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu" or torch.jit.is_scripting():
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert num_chunks <= N, "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N,
        img_h,
        img_w,
        device=device,
        dtype=torch.bool if threshold >= 0 else torch.uint8,
    )
    for inds in chunks:
        masks_chunk, spatial_inds = rescale_mask_func(
            masks[inds, None, :, :],
            boxes[inds],
            img_h,
            img_w,
            skip_empty=device.type == "cpu",
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks


@njit(
    "(float64[:,:], int64[:], int64, int64)",
    nopython=True,
)
def process(S, P, a, b):
    """
    Recursively processes a set of points to find a subset that forms a convex hull.

    Args:
        S (numpy.ndarray): An array of points in 2D space.
        P (numpy.ndarray): An array of indices corresponding to points in S.
        a (int): The index of the starting point in S.
        b (int): The index of the ending point in S.

    Returns:
        list: A list of indices that form the convex hull between points a and b.
    """
    signed_dist = cross2d(S[P] - S[a], S[b] - S[a])
    # Use Boolean indexing instead of a loop
    mask = (signed_dist > 0) & (P != a) & (P != b)
    K = P[mask]

    if len(K) == 0:
        return [a, b]

    c = P[np.argmax(signed_dist)]
    return process(S, K, a, c)[:-1] + process(S, K, c, b)


@njit("(float64[:,:],)", nopython=True)
def quickhull(S: np.ndarray) -> np.ndarray:
    """
    Computes the convex hull of a set of 2D points using the Quickhull algorithm.

    Parameters:
    S (np.ndarray): A 2D numpy array of shape (n, 2) representing the set of points.

    Returns:
    np.ndarray: A 2D numpy array representing the vertices of the convex hull in counter-clockwise order.
    """
    a = np.argmin(S[:, 0])
    max_index = np.argmax(S[:, 0])
    return process(S, np.arange(S.shape[0]), a, max_index)[:-1] + process(S, np.arange(S.shape[0]), max_index, a)[:-1]


def points_to_segments(points):
    """
    Converts a set of points into segments using the Quickhull algorithm.

    Args:
        points (torch.Tensor or numpy.ndarray): A set of points to be converted into segments.
            If a torch.Tensor is provided, it will be converted to a numpy.ndarray.

    Returns:
        numpy.ndarray: An array of points representing the segments formed by the Quickhull algorithm.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    return points[quickhull(points.astype(np.float64))]


def mask_to_polygon(mask, value=1.0):
    """
    Converts a mask to a polygon using the Quickhull algorithm.

    Args:
        mask (torch.Tensor or numpy.ndarray): A mask to be converted into a polygon.
            If a torch.Tensor is provided, it will be converted to a numpy.ndarray.

    Returns:
        numpy.ndarray: An array of points representing the polygon formed by the Quickhull algorithm.
    """
    if isinstance(mask, torch.Tensor):
        points = torch.nonzero(mask == value, as_tuple=False)[:, [1, 0]].cpu().numpy()
    else:
        points = np.vstack(np.where(mask == value))[::-1].T
    return points_to_segments(points.astype(np.float64))


def segment_to_obb(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns the center coordinates (cx, cy), angle of rotation, width, height,
    and a set of points representing the corners of the bounding box.
    original source: https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput

    :param points: an nx2 matrix of coordinates
    :return: (cx, cy, angle, width, height), rval (4x2 matrix of coordinates)
    """
    pi2 = np.pi / 2.0

    # Calculate edge angles
    edges = points[1:] - points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # Find rotation matrices
    rotations = np.vstack([np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # Apply rotations to the hull
    rot_points = np.dot(rotations, points.T)

    # Find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # Find the box with the best (smallest) area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # Dimensions of the best box
    width = max_x[best_idx] - min_x[best_idx]
    height = max_y[best_idx] - min_y[best_idx]

    # Rotation and corners of the best box
    r = rotations[best_idx]
    x1, x2 = max_x[best_idx], min_x[best_idx]
    y1, y2 = max_y[best_idx], min_y[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)  # top-left
    rval[1] = np.dot([x2, y2], r)  # top-right
    rval[2] = np.dot([x2, y1], r)  # bottom-right
    rval[3] = np.dot([x1, y1], r)  # bottom-left

    # Center of the bounding box
    cx, cy = rval.mean(axis=0)

    # Angle of rotation in radians
    angle = angles[best_idx]

    return np.array([cx, cy, width, height, angle]), rval


def mask_to_obb(mask):
    polygon = mask_to_polygon(mask)
    return segment_to_obb(polygon)


def mask_to_polygon_cv2(mask):
    """
    Converts a binary mask to the largest polygon.

    Parameters:
        mask (numpy.ndarray): A 2D binary mask (values are 0 or 255).

    Returns:
        list: A list of (x, y) tuples representing the largest polygon.
    """
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a 2D array.")

    # Find contours of the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Simplify the contour and convert it to a numpy array of (x, y) tuples
    polygon = np.squeeze(largest_contour, axis=1)

    return polygon
