import logging
from typing import Dict, List, Optional, Union

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Results:
    """Object detection results for a single image."""

    def __init__(
        self,
        boxes: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        classes: Optional[List[str]] = None,
        masks: Optional[torch.Tensor] = None,
        segments: Optional[List[torch.Tensor]] = None,
        points: Optional[torch.Tensor] = None,
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
        """Apply a function to all tensors."""
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
            elif k == "segments" and v is not None:
                setattr(r, k, [getattr(s, fn)(*args, **kwargs) for s in v])
        return r

    def to(self, *args, **kwargs):
        """Move all tensors to a device."""
        return self._apply("to", *args, **kwargs)

    def cpu(self):
        """Move all tensors to CPU."""
        return self._apply("cpu")

    def numpy(self):
        """Convert all tensors to numpy arrays."""
        return self._apply("numpy")

    def cuda(self):
        """Move all tensors to GPU."""
        return self._apply("cuda")

    def to_dict(self, return_tensor: bool) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
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
