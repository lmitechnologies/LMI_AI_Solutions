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
    a, b = torch.cartesian_prod(torch.arange(15), torch.arange(15)).unbind(1)
    assert crops.intersections(a, b).tolist() == (masks[a] & masks[b]).flatten(1).sum(dim=1).float().tolist()

    members = [0, 2, 3, 7]
    data, boxes = crops.unions(torch.tensor(members), torch.zeros(len(members), dtype=torch.long), 1)
    picked = crops[torch.tensor([1, 3])]
    picked.set_runs(torch.tensor([0]), data, boxes)
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


def test_boxes_in_regions_finds_the_pixels_inside_the_region():
    masks = torch.zeros(2, 24, 30, dtype=torch.bool)
    masks[0, 4:12, 5:20] = True
    masks[1, 2:6, 25:29] = True
    crops = MaskCrops.from_masks(masks, (24, 30))
    lo = torch.tensor([[10.0, 0, 10, 0], [0.0, 0, 0, 0]])
    hi = torch.tensor([[30.0, 24, 30, 24], [10.0, 24, 10, 24]])
    empty = torch.full((2, 4), -1.0)
    got = crops.boxes_in_regions(torch.tensor([0, 1]), lo, hi, empty)
    assert got[0].tolist() == [10, 4, 20, 12]  # the slice of mask 0 right of x=10
    assert got[1].tolist() == [-1, -1, -1, -1]  # mask 1 has no pixel left of x=10


def test_boxes_in_regions_with_no_region_of_any_width():
    # every region misses its mask on the x axis while still overlapping it on the y axis
    masks = torch.zeros(1, 24, 30, dtype=torch.bool)
    masks[0, 4:12, 5:20] = True
    crops = MaskCrops.from_masks(masks, (24, 30))
    empty = torch.full((1, 4), -1.0)
    got = crops.boxes_in_regions(torch.tensor([0]), torch.tensor([[22.0, 0, 22, 0]]), torch.tensor([[26.0, 24, 26, 24]]), empty)
    assert got.tolist() == empty.tolist()
