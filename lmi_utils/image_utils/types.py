from typing import List, Union

import numpy as np
import torch

# A single HWC image as either a numpy array or a torch tensor.
ImageLike = Union[np.ndarray, torch.Tensor]


def assert_image_like(image: object) -> None:
    """Raise TypeError if *image* is not a numpy array or torch tensor.

    Args:
        image: Object to validate.

    Raises:
        TypeError: If image is neither np.ndarray nor torch.Tensor.
    """
    if not isinstance(image, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(image).__name__}")


def assert_uint8(image: ImageLike) -> None:
    """Raise ValueError if *image* dtype is not uint8.

    Args:
        image: np.ndarray or torch.Tensor to validate.

    Raises:
        ValueError: If image dtype is not uint8.
    """
    if image.dtype not in [np.uint8, torch.uint8]:
        raise ValueError(f"Expected image dtype uint8, got {image.dtype}")


def assert_ndim(image: ImageLike) -> None:
    """Raise ValueError if *image* is not a 2-dimensional (HW) or 3-dimensional (HWC) array/tensor.

    Args:
        image: np.ndarray or torch.Tensor to validate.

    Raises:
        ValueError: If image.ndim is not 2 or 3.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D (HW) or 3D (HWC) image, got {image.ndim}D array with shape {image.shape}")


def to_3channel(image: ImageLike) -> ImageLike:
    """Expand a single-channel image to 3 channels by repeating it.

    Channel *order* is never touched: a 3-channel input is returned as-is, so BGR in is BGR out. Backends expect RGB, so callers
    that load with cv2 must convert first.

    Args:
        image: np.ndarray or torch.Tensor with shape (H, W), (H, W, 1), or (H, W, 3).

    Returns:
        Image with shape (H, W, 3). (H, W, 3) inputs are returned unchanged.

    Raises:
        ValueError: If the channel dimension is not 1 or 3.
    """
    if image.ndim == 2:
        if isinstance(image, torch.Tensor):
            image = image.unsqueeze(-1)
        else:
            image = np.expand_dims(image, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D array with shape {image.shape}")
    c = image.shape[-1]
    if c == 3:
        return image
    if c == 1:
        if isinstance(image, torch.Tensor):
            return image.repeat(1, 1, 3)
        return np.repeat(image, 3, axis=-1)
    raise ValueError(f"Expected image with 1 or 3 channels, got {c} channels with shape {image.shape}")


# Any supported batch input: a single HW/HWC image, a list of HW/HWC images, or a BHWC array/tensor.
ImageBatch = Union[ImageLike, List[ImageLike]]


def normalize_image_batch(image: ImageBatch) -> List[ImageLike]:
    """Normalize any image input to a flat list of images.

    Accepts a single HW or HWC image (numpy or tensor), a list of HW/HWC images,
    or a BHWC batch (numpy or tensor), and always returns a plain list
    preserving the original type. 2D (HW) images are passed through as-is;
    callers are responsible for expanding channels (e.g. via to_3channel).

    Args:
        image: A single HW/HWC image, list of HW/HWC images, or BHWC batch.

    Returns:
        List of images (numpy arrays or torch tensors).

    Raises:
        TypeError: If image is not a numpy array, torch tensor, or list thereof.
        TypeError: If images in the batch mix numpy arrays and torch tensors.
        ValueError: If any image dtype is not uint8.
        ValueError: If any image is not 2-dimensional (HW) or 3-dimensional (HWC).
    """
    if isinstance(image, list):
        for img in image:
            assert_image_like(img)
            assert_uint8(img)
            assert_ndim(img)
        _assert_consistent_types(image)
        return image

    assert_image_like(image)
    assert_uint8(image)
    if image.ndim == 4:
        return list(image)
    assert_ndim(image)
    return [image]


def _assert_consistent_types(images: List[ImageLike]) -> None:
    """Raise TypeError if images mix numpy arrays and torch tensors.

    Args:
        images: List of HWC images to check.

    Raises:
        TypeError: If the list contains both numpy arrays and torch tensors.
    """
    if not images:
        return
    has_numpy = any(isinstance(img, np.ndarray) for img in images)
    has_tensor = any(isinstance(img, torch.Tensor) for img in images)
    if has_numpy and has_tensor:
        types = [type(img).__name__ for img in images]
        raise TypeError(f"All images in a batch must be the same type, got mixed types: {types}")
