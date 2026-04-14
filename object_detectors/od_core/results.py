import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Results:
    """Object detection results for a single image. All numeric fields are torch.Tensor.

    boxes, scores, and classes always default to empty tensors/list so that to_dict()
    always includes them even when there are no detections.
    """

    _keys = ("boxes", "scores", "masks", "points", "segments")
    _all_keys = _keys + ("classes",)

    # Default empty values — defined once here and reused in __init__ and to_dict.
    # Lists are copied on assignment to avoid sharing a mutable default across instances.
    EMPTY_BOXES: torch.Tensor = torch.zeros((0, 4), dtype=torch.float32)
    EMPTY_SCORES: torch.Tensor = torch.zeros((0,), dtype=torch.float32)
    EMPTY_MASKS: torch.Tensor = torch.zeros((0,), dtype=torch.float32)
    EMPTY_CLASSES: np.ndarray = np.array([], dtype=np.str_)
    EMPTY_SEGMENTS: List[torch.Tensor] = []

    def __init__(
        self,
        boxes: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        classes: Optional[np.ndarray] = None,
        masks: Optional[torch.Tensor] = None,
        segments: Optional[List[torch.Tensor]] = None,
        points: Optional[torch.Tensor] = None,
        is_seg: bool = False,
    ):
        self.boxes = boxes if boxes is not None else self.EMPTY_BOXES
        self.scores = scores if scores is not None else self.EMPTY_SCORES
        if classes is None:
            self.classes = self.EMPTY_CLASSES.copy()
        elif isinstance(classes, np.ndarray):
            self.classes = classes
        else:
            self.classes = np.array(classes, dtype=np.str_)
        self.masks = masks
        self.segments = segments
        self.points = points
        self.is_seg = is_seg

    def new(self):
        return Results(classes=self.classes.copy(), is_seg=self.is_seg)

    def _apply(self, fn: str, *args, **kwargs):
        """Apply a tensor method to all tensor fields."""
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is None:
                continue
            if k == "segments":
                setattr(r, k, [getattr(s, fn)(*args, **kwargs) for s in v])
            else:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def to(self, *args, **kwargs):
        """Move all tensors to a device."""
        return self._apply("to", *args, **kwargs)

    def cpu(self):
        """Move all tensors to CPU."""
        return self._apply("cpu")

    def cuda(self):
        """Move all tensors to GPU."""
        return self._apply("cuda")

    def to_dict(self, return_numpy: bool = False) -> Dict[str, Union[torch.Tensor, np.ndarray, List]]:
        """Convert results to a dictionary.
        If return_numpy is True, convert tensors to numpy arrays; otherwise return as-is.
        boxes, scores, and classes are always present (empty tensor/list when no detections).
        Other keys (masks, points, segments) are included only when not None.
        When is_seg is True, 'masks' and 'segments' are always included even if there are no detections.
        """
        r = self.cpu() if return_numpy else self

        def to_numpy(v):
            return v.numpy() if return_numpy and isinstance(v, torch.Tensor) else v

        dt = {}
        for k in r._all_keys:
            v = getattr(r, k)
            if v is not None:
                dt[k] = [to_numpy(s) for s in v] if k == "segments" else to_numpy(v)
            elif r.is_seg:
                if k == "masks":
                    dt[k] = to_numpy(Results.EMPTY_MASKS)
                elif k == "segments":
                    dt[k] = list(Results.EMPTY_SEGMENTS)
        return dt
