from typing import List, Union

import numpy as np
import torch

# A single HWC image as either a numpy array or a torch tensor.
ImageLike = Union[np.ndarray, torch.Tensor]

# Any supported batch input: a single HWC image, a list of HWC images,
# or a BHWC array/tensor.
ImageBatch = Union[ImageLike, List[ImageLike]]


def normalize_image_batch(image: ImageBatch) -> List[ImageLike]:
    """Normalize any image input to a flat list of HWC images.

    Accepts a single HWC image (numpy or tensor), a list of HWC images,
    or a BHWC batch (numpy or tensor), and always returns a plain list
    of HWC images preserving the original type.

    Args:
        image: A single HWC image, list of HWC images, or BHWC batch.

    Returns:
        List of HWC images (numpy arrays or torch tensors).

    Raises:
        TypeError: If images in the batch mix numpy arrays and torch tensors.
    """
    if isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim == 4:
        return list(image)
    if isinstance(image, list):
        _assert_consistent_types(image)
        return image
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
