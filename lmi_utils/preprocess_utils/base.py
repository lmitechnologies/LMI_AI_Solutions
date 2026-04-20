from typing import Any, Dict, List

import numpy as np
import torch

from lmi_utils.image_utils.types import ImageLike


class BaseProcessor:
    _STEP_REQUIRED_KEYS = {"type", "configuration"}
    _HISTORY_REQUIRED_KEYS = {"type", "metadata"}

    def to_tensor_list(self, images: List[ImageLike]) -> tuple[List[torch.Tensor], bool]:
        """
        Convert image list to tensors if needed.

        Returns:
            (tensor_list, is_numpy): List of tensors and flag indicating if input was numpy
        """
        is_numpy = isinstance(images[0], np.ndarray)
        if is_numpy:
            return [torch.from_numpy(img) for img in images], True
        return images, False

    def from_tensor_list(self, images: List[torch.Tensor], to_numpy: bool) -> List[ImageLike]:
        """Convert tensor list back to numpy if needed."""
        if to_numpy:
            return [img.cpu().numpy() for img in images]
        return images

    def validate_image_list(self, images: List[ImageLike], stage: str = "processing") -> None:
        """Validate that images is a proper list of tensors or arrays."""
        if not isinstance(images, list):
            raise TypeError("Images must be a list.")

        if not images:
            raise ValueError(f"No input images provided for {stage}.")

        first_type = type(images[0])
        if first_type not in (torch.Tensor, np.ndarray):
            raise TypeError("Images must be torch.Tensors or np.ndarrays")
        if not all(isinstance(img, first_type) for img in images):
            raise TypeError("All images must be the same type")
        for img in images:
            if img.ndim not in (2, 3):
                raise ValueError(f"Expected 2D (HW) or 3D (HWC) image, got {img.ndim}D array with shape {img.shape}")

    def validate_handler_output(self, output: Any, handler_name: str, expected_type: str = "handler") -> None:
        """Validate handler output format."""
        if not isinstance(output, list):
            raise TypeError(f"{expected_type.capitalize()} '{handler_name}' must return a list of images, got {type(output)}")
        if not all(isinstance(img, torch.Tensor) for img in output):
            raise TypeError(f"{expected_type.capitalize()} '{handler_name}' returned non-tensor images")

    def validate_handler_metadata(self, metadata: Any, handler_name: str) -> None:
        """Validate that handler metadata is a dict containing the required 'metadata' key."""
        if not isinstance(metadata, dict):
            raise TypeError(f"Handler '{handler_name}' must return metadata as dict, got {type(metadata)}")
        if "metadata" not in metadata:
            raise KeyError(f"Handler '{handler_name}' metadata dict must contain key 'metadata'")

    def validate_step_keys(self, step: Dict[str, Any], required_keys: set) -> None:
        """Validate that a step dictionary contains required keys."""
        if not isinstance(step, dict):
            raise TypeError("Each processing step must be a dictionary.")
        if not required_keys.issubset(step.keys()):
            raise ValueError(f"Each step must contain keys: {required_keys}")

    def validate_steps(self, steps: List[Dict[str, Any]]) -> None:
        """Validate that steps is a proper list of metadata dictionaries."""
        if not isinstance(steps, list):
            raise TypeError("Steps must be a list.")
        if not all(isinstance(step, dict) for step in steps):
            raise TypeError("All steps must be dictionaries.")

        # validate each step
        for step in steps:
            self.validate_step_keys(step, self._STEP_REQUIRED_KEYS)

    def validate_history_steps(self, steps: List[Dict[str, Any]]) -> None:
        """Validate that history steps each contain 'type' and 'metadata' keys."""
        if not isinstance(steps, list):
            raise TypeError("Steps must be a list.")
        if not all(isinstance(step, dict) for step in steps):
            raise TypeError("All steps must be dictionaries.")

        for step in steps:
            self.validate_step_keys(step, self._HISTORY_REQUIRED_KEYS)
