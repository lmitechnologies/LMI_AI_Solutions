"""Trace masks into segments, for the tile merge and for backends without their own outline rule.

A segment is the outer outline of a mask: holes are dropped. The mask keeps the exact shape.
"""

from typing import List

import cv2
import numpy as np
import torch


def masks_to_segments(masks) -> List[np.ndarray]:
    """(M, 2) float32 outline of each (H, W) mask, in px; (0, 2) for an empty mask. Values above 0.5 are inside.

    Of pieces that do not touch, keeps the largest.
    """
    if isinstance(masks, torch.Tensor):
        masks = (masks > 0.5).to(torch.uint8).cpu().numpy()
    else:
        masks = (np.asarray(masks) > 0.5).astype(np.uint8)
    return [trace_mask(m) for m in masks]


def trace_mask(mask: np.ndarray) -> np.ndarray:
    """``masks_to_segments`` for one (H, W) uint8 mask of 0 and 1."""
    contours = cv2.findContours(np.ascontiguousarray(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if mask.size else ()
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
