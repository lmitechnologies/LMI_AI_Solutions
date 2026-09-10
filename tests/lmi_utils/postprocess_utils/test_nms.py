"""Class-aware NMS: the IoU rule, the containment rule, and shape-vs-box containment."""

import numpy as np
import pytest
import torch

import lmi_utils.postprocess_utils.nms as nms
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


def _dense_overlap(masks):
    """Every pixel of every pair: the formulation the box-cropped _mask_overlap must reproduce."""
    flat = (masks.flatten(1) != 0).float()
    return flat @ flat.t(), flat.sum(dim=1)


def _scattered_masks(n, size=24):
    """Speckled patches at random spots, every fifth mask empty, so many pairs never touch."""
    g = torch.Generator().manual_seed(n)
    masks = torch.zeros(n, size, size, dtype=torch.bool)
    for i in range(n):
        if i % 5 == 4:
            continue
        x, y = torch.randint(0, size - 6, (2,), generator=g).tolist()
        masks[i, y : y + 7, x : x + 7] = torch.rand(7, 7, generator=g) > 0.3
    return masks


@pytest.mark.parametrize(
    "masks", [torch.rand(0, 5, 5) > 0.5, torch.rand(1, 5, 5) > 0.5, torch.rand(7, 5, 5) > 0.5, _scattered_masks(16), _scattered_masks(40)]
)
def test_mask_overlap_matches_the_dense_formulation(masks):
    inter, area = nms._mask_overlap(masks)
    want_inter, want_area = _dense_overlap(masks)
    assert torch.equal(inter, want_inter)
    assert torch.equal(area, want_area)


def test_boxes_from_masks_bound_the_pixels_with_exclusive_max_edges():
    from torchvision.ops import masks_to_boxes

    masks = _scattered_masks(10)
    boxes = nms.boxes_from_masks(masks)
    full = masks.flatten(1).any(dim=1)
    assert torch.equal(boxes[full], masks_to_boxes(masks[full]) + torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert not boxes[~full].any()


def test_in_place_filter_matches_a_copy_without_allocating_new_masks():
    masks = _scattered_masks(12)
    keep = torch.tensor([1, 3, 4, 9, 11])
    want = nms.filter_instances({"masks": masks.clone()}, keep)["masks"]
    out = nms.filter_instances({"masks": masks}, keep, in_place=True)["masks"]
    assert torch.equal(out, want)
    assert out.untyped_storage().data_ptr() == masks.untyped_storage().data_ptr()


def test_in_place_filter_rejects_unsorted_indices():
    with pytest.raises(ValueError, match="ascending"):
        nms.filter_instances({"masks": _scattered_masks(4)}, torch.tensor([2, 0]), in_place=True)


def test_mask_overlap_area_counts_pixels_not_promoted_sums():
    masks = torch.zeros(2, 4, 4, dtype=torch.bool)
    masks[0, :2, :2] = True
    masks[1, 0, 0] = True
    _, area = nms._mask_overlap(masks)
    assert area.tolist() == [4.0, 1.0]


@pytest.mark.parametrize(
    "masks, want",
    [
        (torch.tensor([[[0, 1]]], dtype=torch.uint8), [[[False, True]]]),
        (torch.tensor([[[0.0, 0.4, 0.6]]]), [[[False, False, True]]]),  # soft masks threshold at 0.5
        (torch.tensor([[[False, True]]]), [[[False, True]]]),
    ],
)
def test_binarize_masks(masks, want):
    out = nms.binarize_masks(masks)
    assert out.dtype == torch.bool
    assert out.tolist() == want
