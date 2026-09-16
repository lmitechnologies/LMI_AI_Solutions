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


def _densify(overlap):
    """The sparse (pairs, intersection, area) overlap as the full matrix the dense formulation gives."""
    pairs, inter, area = overlap
    dense = torch.diag(area)
    for (i, j), v in zip(pairs.tolist(), inter.tolist()):
        dense[i, j] = dense[j, i] = v
    return dense


@pytest.mark.parametrize(
    "masks", [torch.rand(0, 5, 5) > 0.5, torch.rand(1, 5, 5) > 0.5, torch.rand(7, 5, 5) > 0.5, _scattered_masks(16), _scattered_masks(40)]
)
def test_mask_overlap_matches_the_dense_formulation(masks):
    overlap = nms._mask_overlap(masks)
    want_inter, want_area = _dense_overlap(masks)
    assert torch.equal(_densify(overlap), want_inter)
    assert torch.equal(overlap[2], want_area)


def test_mask_overlap_only_lists_pairs_that_actually_intersect():
    masks = torch.zeros(3, 10, 10, dtype=torch.bool)
    masks[0, 0:4, 0:4] = True
    masks[1, 2:6, 2:6] = True  # overlaps mask 0
    masks[2, 8:10, 8:10] = True  # touches nothing
    pairs, inter, _ = nms._mask_overlap(masks)
    assert pairs.tolist() == [[0, 1]]
    assert inter.tolist() == [4.0]


def test_mask_overlap_area_counts_pixels_not_promoted_sums():
    masks = torch.zeros(2, 4, 4, dtype=torch.bool)
    masks[0, :2, :2] = True
    masks[1, 0, 0] = True
    assert nms._mask_overlap(masks)[2].tolist() == [4.0, 1.0]


def test_disjoint_shapes_never_suppress_each_other():
    # containment is a fraction of the suppressed shape, so a zero intersection must not clear a zero threshold
    merged = _result([[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]], [0.9, 0.8, 0.7])
    assert len(class_aware_nms(merged, 0.5, 0.0)["boxes"]) == 3


def test_suppression_leaves_the_kept_box_untouched():
    out = class_aware_nms(_result([[0, 0, 10, 10], [2, 0, 12, 10]], [0.9, 0.6]), 0.5)
    assert torch.allclose(out["boxes"][0], torch.tensor([0.0, 0.0, 10.0, 10.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("field", ["boxes", "masks", "segments"])
def test_nms_on_cuda_geometry_matches_the_cpu_answer(field):
    boxes = [[0, 0, 40, 40], [10, 10, 50, 50], [5, 5, 20, 20], [200, 200, 240, 240]]
    extra = {}
    if field == "masks":
        masks = torch.zeros(4, 260, 260, dtype=torch.uint8)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            masks[i, y0:y1, x0:x1] = 1
        extra["masks"] = masks
    if field == "segments":
        extra["segments"] = [torch.tensor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=torch.float32) for x0, y0, x1, y1 in boxes]

    cpu = _result(boxes, [0.9, 0.8, 0.7, 0.6], **extra)
    gpu = {k: (v.cuda() if isinstance(v, torch.Tensor) else [s.cuda() for s in v] if k == "segments" else v) for k, v in cpu.items()}
    out = class_aware_nms(gpu, 0.5, 0.8)
    assert out["scores"].is_cuda
    assert out["scores"].cpu().tolist() == class_aware_nms(cpu, 0.5, 0.8)["scores"].tolist()
