import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

TensorOrArray = Union[torch.Tensor, np.ndarray]


class Results:
    """Object detection results for a single image."""

    def __init__(
        self,
        boxes: Optional[TensorOrArray] = None,
        scores: Optional[TensorOrArray] = None,
        classes: Optional[List[str]] = None,
        masks: Optional[TensorOrArray] = None,
        segments: Optional[List[TensorOrArray]] = None,
        points: Optional[TensorOrArray] = None,
    ):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.masks = masks
        self.segments = segments
        self.points = points
        self._keys = "boxes", "scores", "masks", "points", "segments"
        self._all_keys = self._keys + ("classes",)

    def new(self):
        return Results(classes=self.classes)

    def _apply(self, fn: str, *args, **kwargs):
        """Apply a tensor method to all tensors; numpy arrays are passed through unchanged."""
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
            elif isinstance(v, np.ndarray):
                setattr(r, k, v)
            elif k == "segments" and v is not None:
                segs = []
                for s in v:
                    if isinstance(s, torch.Tensor):
                        segs.append(getattr(s, fn)(*args, **kwargs))
                    else:
                        segs.append(s)
                setattr(r, k, segs)
        return r

    def to(self, *args, **kwargs):
        """Move all tensors to a device. Numpy arrays are passed through unchanged."""
        return self._apply("to", *args, **kwargs)

    def cpu(self):
        """Move all tensors to CPU. Numpy arrays are passed through unchanged."""
        return self._apply("cpu")

    def numpy(self):
        """Convert all tensors to numpy arrays. Numpy arrays are returned as-is."""
        return self._apply("numpy")

    def cuda(self):
        """Move all tensors to GPU. Raises TypeError if any value is a numpy array."""
        for k in self._keys:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                raise TypeError(f"Cannot move numpy array '{k}' to CUDA. Convert to tensor first.")
            if k == "segments" and v is not None:
                for s in v:
                    if isinstance(s, np.ndarray):
                        raise TypeError("Cannot move numpy array in 'segments' to CUDA. Convert to tensor first.")
        return self._apply("cuda")

    def to_dict(self, return_tensor: bool) -> Dict[str, Union[TensorOrArray, List[TensorOrArray]]]:
        """Convert results to a dictionary.
        Return tensors or numpy arrays based on `return_tensor`.
        """
        dt = {}
        r = self if return_tensor else self.cpu().numpy()
        for k in r._all_keys:
            v = getattr(r, k)
            if v is not None and len(v) > 0:
                dt[k] = v
        return dt
