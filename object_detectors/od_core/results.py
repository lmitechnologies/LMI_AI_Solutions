from typing import List, Optional, Union, Dict
import numpy as np
import torch

class Results:
    """Object detection results for a single image.
    """

    def __init__(
        self, 
        boxes: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        classes: Optional[list[str]] = None,
        masks: Optional[torch.Tensor] = None,
        segments: Optional[List[np.ndarray]] = None,
        points: Optional[torch.Tensor] = None
    ):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.masks = masks
        self.segments = segments
        self.points = points
        self._keys = "boxes", "scores", "masks", "points"
        self._all_keys = self._keys + ("segments", "classes")


    def new(self):
        return Results(self.names)
    
    
    def _apply(self, fn:str, *args, **kwargs):
        """Apply a function to all tensors."""
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
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
    
    
    def to_dict(self):
        """Convert results to a dictionary."""
        return {k: getattr(self, k) for k in self._all_keys if getattr(self, k) is not None}
        