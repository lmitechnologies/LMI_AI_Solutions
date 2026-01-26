from typing import Any, Dict, List, Union

import numpy as np
import torch


class BaseProcessor:
    _STEP_REQUIRED_KEYS = {"type", "configuration"}

    def to_tensor_list(self, images: List[Union[torch.Tensor, np.ndarray]]) -> tuple[List[torch.Tensor], bool]:
        """
        Convert image list to tensors if needed.

        Returns:
            (tensor_list, is_numpy): List of tensors and flag indicating if input was numpy
        """
        is_numpy = isinstance(images[0], np.ndarray)
        if is_numpy:
            return [torch.from_numpy(img) for img in images], True
        return images, False

    def from_tensor_list(self, images: List[torch.Tensor], to_numpy: bool) -> List[Union[torch.Tensor, np.ndarray]]:
        """Convert tensor list back to numpy if needed."""
        if to_numpy:
            return [img.cpu().numpy() for img in images]
        return images

    def validate_image_list(self, images: List[Union[torch.Tensor, np.ndarray]], stage: str = "processing") -> None:
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

    def validate_handler_output(self, output: Any, handler_name: str, expected_type: str = "handler") -> None:
        """Validate handler output format."""
        if not isinstance(output, list):
            raise TypeError(f"{expected_type.capitalize()} '{handler_name}' must return a list of images, got {type(output)}")
        if not all(isinstance(img, torch.Tensor) for img in output):
            raise TypeError(f"{expected_type.capitalize()} '{handler_name}' returned non-tensor images")

    def validate_step_keys(self, step: Dict[str, Any], required_keys: set) -> None:
        """Validate that a step dictionary contains required keys."""
        if not isinstance(step, dict):
            raise TypeError("Each processing step must be a dictionary.")
        if not required_keys.issubset(step.keys()):
            raise ValueError(f"Each step must contain keys: {required_keys}")
