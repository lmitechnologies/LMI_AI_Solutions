from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from lmi_utils.image_utils.types import ImageLike


class BaseProcessor:
    _STEP_REQUIRED_KEYS = {"type", "configuration"}
    _HISTORY_REQUIRED_KEYS = {"type", "metadata"}
    _COORD_FIELDS = frozenset({"boxes", "segments", "points", "masks"})

    def to_tensor_list(self, images: List[ImageLike]) -> Tuple[List[torch.Tensor], bool]:
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

    def to_tensor_results(self, per_image: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Convert coordinate fields in result dicts to float32 tensors.

        Converts boxes, segments, points, masks. Leaves classes and scores untouched.
        Returns (converted_results, is_numpy).
        """
        is_numpy = self._detect_results_numpy(per_image)
        if not is_numpy:
            return per_image, False
        return [self._result_to_tensor(r) for r in per_image], True

    def from_tensor_results(self, per_image: List[Dict[str, Any]], to_numpy: bool) -> List[Dict[str, Any]]:
        """Convert coordinate fields back to numpy if to_numpy is True."""
        if not to_numpy:
            return per_image
        return [self._result_from_tensor(r) for r in per_image]

    def _detect_results_numpy(self, per_image: List[Dict[str, Any]]) -> bool:
        """Return True if coordinate fields are numpy arrays. Raises if types are mixed."""
        first_type = None
        for r in per_image:
            for field in self._COORD_FIELDS:
                val = r.get(field)
                if val is None or len(val) == 0:
                    continue
                probe = val[0] if field == "segments" else val
                if not len(probe):
                    continue
                t = type(probe)
                if first_type is None:
                    if t not in (torch.Tensor, np.ndarray):
                        raise TypeError(f"Coordinate field '{field}' must be torch.Tensor or np.ndarray, got {t}")
                    first_type = t
                elif t is not first_type:
                    raise TypeError(f"Mixed coordinate field types: expected {first_type.__name__}, got {t.__name__} in field '{field}'")
        return first_type is np.ndarray

    def _result_to_tensor(self, result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)
        for field in self._COORD_FIELDS:
            val = result.get(field)
            if val is None or len(val) == 0:
                continue
            if field == "segments":
                out[field] = [torch.from_numpy(s.astype(np.float32)) if isinstance(s, np.ndarray) and len(s) else s for s in val]
            elif isinstance(val, np.ndarray):
                out[field] = torch.from_numpy(val.astype(np.float32))
        return out

    def _result_from_tensor(self, result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)
        for field in self._COORD_FIELDS:
            val = result.get(field)
            if val is None or len(val) == 0:
                continue
            if field == "segments":
                out[field] = [s.cpu().numpy() if isinstance(s, torch.Tensor) and len(s) else s for s in val]
            elif isinstance(val, torch.Tensor):
                out[field] = val.cpu().numpy()
        return out

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

    def validate_image_handler_output(self, output: Any, handler_name: str, expected_type: str = "handler") -> None:
        """Validate handler output format."""
        if not isinstance(output, list):
            raise TypeError(f"{expected_type.capitalize()} '{handler_name}' must return a list of images, got {type(output)}")
        if not all(isinstance(img, torch.Tensor) for img in output):
            raise TypeError(f"{expected_type.capitalize()} '{handler_name}' returned non-tensor images")

    def validate_coord_handler_output(self, output: Any, handler_name: str) -> None:
        """Validate coord handler output is a list of dicts containing all _COORD_FIELDS."""
        if not isinstance(output, list) or not all(isinstance(d, dict) for d in output):
            raise TypeError(f"Revert coordinate handler '{handler_name}' must return a list of dicts, got {type(output)}")
        if output and "boxes" not in output[0]:
            raise KeyError(f"Revert coordinate handler '{handler_name}' output missing required key 'boxes'")

    def validate_handler_metadata(self, metadata: Any, handler_name: str) -> None:
        """Validate that handler metadata is a dict containing the required 'metadata' key."""
        if not isinstance(metadata, dict):
            raise TypeError(f"Handler '{handler_name}' must return metadata as dict, got {type(metadata)}")
        if "metadata" not in metadata:
            raise KeyError(f"Handler '{handler_name}' metadata dict must contain key 'metadata'")

    def _validate_steps(self, steps: List[Dict[str, Any]], required_keys: set) -> None:
        if not isinstance(steps, list):
            raise TypeError("Steps must be a list.")
        if not all(isinstance(step, dict) for step in steps):
            raise TypeError("All steps must be dictionaries.")
        for step in steps:
            if not required_keys.issubset(step.keys()):
                raise ValueError(f"Each step must contain keys: {required_keys}")

    def validate_steps(self, steps: List[Dict[str, Any]]) -> None:
        self._validate_steps(steps, self._STEP_REQUIRED_KEYS)

    def validate_history_steps(self, steps: List[Dict[str, Any]]) -> None:
        self._validate_steps(steps, self._HISTORY_REQUIRED_KEYS)
