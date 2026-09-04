from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from lmi_utils.image_utils.types import ImageLike


class BaseProcessor:
    _COORD_FIELDS = frozenset({"boxes", "segments", "points", "masks"})
    # Subset of _COORD_FIELDS whose value is a list-of-tensors instead of a single tensor.
    _COORD_LIST_FIELDS = frozenset({"segments"})
    # Per-instance fields that carry no coordinates. Converted alongside the coord fields so ops
    # that filter or concatenate instances (tiling) can index them, but exempt from drop validation.
    _INSTANCE_FIELDS = frozenset({"scores"})
    _TENSOR_FIELDS = _COORD_FIELDS | _INSTANCE_FIELDS

    @staticmethod
    def as_image_list(images: Any) -> List[ImageLike]:
        """Normalize input into a flat list of HW(C) images.

        Accepts a single HW/HWC image, a BHWC batch (4D), or an existing list and always returns a list.
        """
        if isinstance(images, list):
            return images
        if hasattr(images, "ndim") and images.ndim == 4:
            return list(images)
        return [images]

    def to_tensor_list(self, images: List[ImageLike]) -> Tuple[List[torch.Tensor], bool]:
        """Convert image list to tensors if needed; make contiguous."""
        is_numpy = isinstance(images[0], np.ndarray)
        if is_numpy:
            return [torch.from_numpy(np.ascontiguousarray(img)) for img in images], True
        return [img.contiguous() for img in images], False

    def from_tensor_list(self, images: List[torch.Tensor], to_numpy: bool) -> List[ImageLike]:
        images = [img.contiguous() for img in images]
        if to_numpy:
            return [img.cpu().numpy() for img in images]
        return images

    def to_tensor_results(self, per_image: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        is_numpy = self._detect_results_numpy(per_image)
        if not is_numpy:
            return per_image, False
        return [self._result_to_tensor(r) for r in per_image], True

    def from_tensor_results(self, per_image: List[Dict[str, Any]], to_numpy: bool) -> List[Dict[str, Any]]:
        if not to_numpy:
            return per_image
        return [self._result_from_tensor(r) for r in per_image]

    def _detect_results_numpy(self, per_image: List[Dict[str, Any]]) -> bool:
        first_type = None
        for r in per_image:
            for field in self._COORD_FIELDS:
                val = r.get(field)
                if val is None or len(val) == 0:
                    continue
                probe = val[0] if field in self._COORD_LIST_FIELDS else val
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
        for field in self._TENSOR_FIELDS:
            val = result.get(field)
            if val is None or len(val) == 0:
                continue
            if field in self._COORD_LIST_FIELDS:
                out[field] = [torch.from_numpy(s.astype(np.float32)) if isinstance(s, np.ndarray) and len(s) else s for s in val]
            elif isinstance(val, np.ndarray):
                out[field] = torch.from_numpy(val.astype(np.float32))
        return out

    def _result_from_tensor(self, result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)
        for field in self._TENSOR_FIELDS:
            val = result.get(field)
            if val is None or len(val) == 0:
                continue
            if field in self._COORD_LIST_FIELDS:
                out[field] = [s.cpu().numpy() if isinstance(s, torch.Tensor) and len(s) else s for s in val]
            elif isinstance(val, torch.Tensor):
                out[field] = val.cpu().numpy()
        return out

    def validate_image_list(self, images: List[ImageLike], stage: str = "processing") -> None:
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
        if not isinstance(output, list):
            raise TypeError(f"{expected_type.capitalize()} '{handler_name}' must return a list of images, got {type(output)}")
        if not all(isinstance(img, torch.Tensor) for img in output):
            raise TypeError(f"{expected_type.capitalize()} '{handler_name}' returned non-tensor images")

    def validate_coord_handler_output(self, output: Any, handler_name: str, input_populated: set = None) -> set:
        """Validate a coord handler's output and return its populated coord fields."""
        if not isinstance(output, list) or not all(isinstance(d, dict) for d in output):
            raise TypeError(f"Revert coordinate handler '{handler_name}' must return a list of dicts, got {type(output)}")
        output_populated = self._populated_coord_fields(output)
        if input_populated:
            dropped = input_populated - output_populated
            if dropped:
                raise KeyError(
                    f"Revert coordinate handler '{handler_name}' dropped non-empty coord field(s): {sorted(dropped)}. "
                    f"Input had {sorted(input_populated)}, output has {sorted(output_populated)}."
                )
        return output_populated

    def _populated_coord_fields(self, per_image: List[Dict[str, Any]]) -> set:
        populated = set()
        for r in per_image:
            for field in self._COORD_FIELDS:
                if field in populated:
                    continue
                val = r.get(field)
                if val is None:
                    continue
                if field in self._COORD_LIST_FIELDS:
                    if any(len(s) for s in val):
                        populated.add(field)
                elif len(val):
                    populated.add(field)
        return populated
