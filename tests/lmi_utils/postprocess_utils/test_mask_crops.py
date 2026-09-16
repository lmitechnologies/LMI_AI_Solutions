"""MaskCrops: masks stored as the crop of their box must behave like the full-image masks."""

import pytest
import torch

from lmi_utils.postprocess_utils.mask_crops import MaskCrops, binarize_masks, boxes_from_masks


def _scattered_masks(n, h=24, w=30):
    g = torch.Generator().manual_seed(n)
    masks = torch.zeros(n, h, w, dtype=torch.bool)
    for i in range(n):
        if i % 5 == 4:
            continue
        x, y = torch.randint(0, w - 6, (1,), generator=g).item(), torch.randint(0, h - 6, (1,), generator=g).item()
        masks[i, y : y + 7, x : x + 7] = torch.rand(7, 7, generator=g) > 0.3
    return masks


def test_boxes_from_masks_bound_the_pixels_with_exclusive_max_edges():
    from torchvision.ops import masks_to_boxes

    masks = _scattered_masks(10)
    boxes = boxes_from_masks(masks)
    full = masks.flatten(1).any(dim=1)
    assert torch.equal(boxes[full], masks_to_boxes(masks[full]) + torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert not boxes[~full].any()


@pytest.mark.parametrize(
    "masks, want",
    [
        (torch.tensor([[[0, 1]]], dtype=torch.uint8), [[[False, True]]]),
        (torch.tensor([[[0.0, 0.4, 0.6]]]), [[[False, False, True]]]),  # soft masks threshold at 0.5
        (torch.tensor([[[False, True]]]), [[[False, True]]]),
    ],
)
def test_binarize_masks(masks, want):
    out = binarize_masks(masks)
    assert out.dtype == torch.bool
    assert out.tolist() == want


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.float32])
def test_round_trip_keeps_pixels_and_dtype(dtype):
    masks = _scattered_masks(12).to(dtype)
    crops = MaskCrops.from_masks(masks, (24, 30))
    out = crops.paste()
    assert out.dtype == dtype
    assert torch.equal(out, masks)
    assert crops.areas().tolist() == masks.bool().flatten(1).sum(dim=1).float().tolist()


def test_offset_drops_pixels_outside_the_image():
    masks = _scattered_masks(12)
    crops = MaskCrops.from_masks(masks, (20, 25), offset=(10, 6))
    want = torch.zeros(12, 20, 25, dtype=torch.bool)
    want[:, 6:, 10:] = masks[:, :14, :15]
    assert torch.equal(crops.paste(), want)
    empty = ~want.flatten(1).any(dim=1)
    assert not crops.boxes[empty].any()


def test_intersection_union_and_indexing_match_full_masks():
    masks = _scattered_masks(15)
    crops = MaskCrops.from_masks(masks, (24, 30))
    for a in range(15):
        for b in range(15):
            assert crops.intersection(a, b) == float((masks[a] & masks[b]).sum())

    members = [0, 2, 3, 7]
    crop, box = crops.union(members)
    picked = crops[torch.tensor([1, 3])]
    picked.set(0, crop, box)
    want = torch.stack([masks[members].any(dim=0), masks[3]])
    assert torch.equal(picked.paste(), want)
    assert torch.equal(crops.paste(), masks)  # editing the indexed copy leaves the source alone
    assert picked.areas().tolist() == want.flatten(1).sum(dim=1).float().tolist()


def test_cat_and_empty():
    masks = _scattered_masks(6)
    parts = [MaskCrops.from_masks(masks[:2], (24, 30)), MaskCrops.from_masks(masks[2:], (24, 30))]
    assert torch.equal(MaskCrops.cat(parts).paste(), masks)
    none = MaskCrops.from_masks(masks, (24, 30))[torch.zeros(0, dtype=torch.long)]
    assert none.paste().shape == (0, 24, 30)
