"""Class-aware NMS: the IoU rule, the containment rule, and shape-vs-box containment."""

import numpy as np
import pytest
import torch

from lmi_utils.postprocess_utils.nms import class_aware_nms


def _result(boxes, scores, classes=None, **extra):
    out = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "classes": np.array(classes if classes is not None else [0] * len(scores), dtype=np.int32),
    }
    out.update(extra)
    return out


def test_containment_drops_a_small_box_nested_in_a_large_one():
    # IoU is only 0.0625, far under the threshold; containment is what catches it.
    out = class_aware_nms(_result([[0, 0, 80, 80], [20, 20, 40, 40]], [0.9, 0.6]), 0.5, 0.8)
    assert out["boxes"].shape == (1, 4)
    assert out["scores"].item() == pytest.approx(0.9)


def test_containment_is_class_aware():
    out = class_aware_nms(_result([[0, 0, 80, 80], [20, 20, 40, 40]], [0.9, 0.6], classes=[0, 1]), 0.5, 0.8)
    assert out["boxes"].shape == (2, 4)


def test_containment_below_threshold_keeps_both():
    # Half of the smaller box sticks out, so it is not contained.
    out = class_aware_nms(_result([[0, 0, 80, 80], [70, 20, 90, 40]], [0.9, 0.6]), 0.5, 0.8)
    assert out["boxes"].shape == (2, 4)


def test_containment_is_measured_on_the_mask_not_its_box():
    # An object sitting in the hole of a ring is inside the ring's box but not inside the ring.
    ring = torch.zeros((100, 100), dtype=torch.uint8)
    ring[10:90, 10:90] = 1
    ring[30:70, 30:70] = 0
    inner = torch.zeros((100, 100), dtype=torch.uint8)
    inner[40:60, 40:60] = 1
    merged = _result([[10, 10, 90, 90], [40, 40, 60, 60]], [0.9, 0.6], masks=torch.stack([ring, inner]))
    out = class_aware_nms(merged, 0.5, 0.8)
    assert out["masks"].shape[0] == 2


def test_iou_rule_still_applies_when_containment_is_disabled():
    out = class_aware_nms(_result([[0, 0, 80, 80], [1, 1, 81, 81]], [0.9, 0.6]), 0.5, None)
    assert out["boxes"].shape == (1, 4)


def test_no_thresholds_is_a_no_op():
    merged = _result([[0, 0, 80, 80], [20, 20, 40, 40]], [0.9, 0.6])
    assert class_aware_nms(merged, None, None) is merged
