"""Focused tests for TileOperation."""

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.ops.tile import TileConfig
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _empty_results(n=1, **overrides):
    results = {
        "boxes": [torch.zeros((0, 4)) for _ in range(n)],
        "scores": [torch.zeros((0,)) for _ in range(n)],
        "classes": [np.zeros((0,), dtype=np.int32) for _ in range(n)],
        "segments": [[] for _ in range(n)],
        "points": [torch.zeros((0, 1, 3)) for _ in range(n)],
    }
    results.update(overrides)
    return results


def _tile_pipeline(im_size=200, tile=100, stride=100, scale_mode="padding"):
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(im_size, im_size, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=tile, stride=stride, scale_mode=scale_mode)])
    return pre, rec, history


def test_tile_apply_coords_xyxy_clips_and_drops_per_tile():
    pre, rec, history = _tile_pipeline()
    results = {
        "boxes": [torch.tensor([[80.0, 50.0, 140.0, 90.0]])],
        "scores": [torch.tensor([0.9])],
        "classes": [np.array([5], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    out = rec.apply_coordinates(results, history)
    assert torch.allclose(out["boxes"][0], torch.tensor([[80.0, 50.0, 100.0, 90.0]]))
    assert out["scores"][0].tolist() == [pytest.approx(0.9)]
    assert out["classes"][0].tolist() == [5]
    assert torch.allclose(out["boxes"][1], torch.tensor([[0.0, 50.0, 40.0, 90.0]]))
    assert out["classes"][1].tolist() == [5]
    assert out["boxes"][2].shape == (0, 4)
    assert out["boxes"][3].shape == (0, 4)
    assert len(out["scores"][2]) == 0 and len(out["classes"][3]) == 0


def test_tile_apply_coords_obb_clipped_and_refit_to_min_area_rect():
    _, rec, history = _tile_pipeline()
    obb = torch.tensor([[[70.0, 40.0], [130.0, 40.0], [130.0, 80.0], [70.0, 80.0]]])
    results = {
        "boxes": [obb],
        "scores": [torch.tensor([0.5])],
        "classes": [np.array([0], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    out = rec.apply_coordinates(results, history)

    assert out["boxes"][0].shape == (1, 4, 2)
    tl = out["boxes"][0][0]
    assert tl[:, 0].min().item() >= 70.0 - 1e-4 and tl[:, 0].max().item() <= 100.0 + 1e-4
    assert tl[:, 1].min().item() >= 40.0 - 1e-4 and tl[:, 1].max().item() <= 80.0 + 1e-4

    assert out["boxes"][1].shape == (1, 4, 2)
    tr = out["boxes"][1][0]
    assert tr[:, 0].min().item() >= 0.0 - 1e-4 and tr[:, 0].max().item() <= 30.0 + 1e-4
    assert tr[:, 1].min().item() >= 40.0 - 1e-4 and tr[:, 1].max().item() <= 80.0 + 1e-4

    assert out["boxes"][2].shape == (0, 4, 2)
    assert out["boxes"][3].shape == (0, 4, 2)


def test_tile_apply_coords_points_visibility_zeroed_outside_tile():
    _, rec, history = _tile_pipeline()
    pts = torch.tensor([[[30.0, 30.0, 2.0], [150.0, 30.0, 2.0], [30.0, 150.0, 2.0]]])
    results = {
        "boxes": [torch.zeros((0, 4))],
        "scores": [torch.zeros((0,))],
        "classes": [np.zeros((0,), dtype=np.int32)],
        "segments": [[]],
        "points": [pts],
    }
    out = rec.apply_coordinates(results, history)
    assert out["points"][0].shape == (1, 3, 3)
    assert out["points"][0][0, 0].tolist() == [pytest.approx(30.0), pytest.approx(30.0), pytest.approx(2.0)]
    assert out["points"][0][0, 1, 2].item() == 0.0
    assert out["points"][0][0, 2, 2].item() == 0.0
    assert out["points"][3].shape == (0, 3, 3)


def test_tile_apply_coords_segments_clipped_to_tile_rect():
    _, rec, history = _tile_pipeline()
    poly = torch.tensor([[80.0, 40.0], [140.0, 40.0], [140.0, 80.0], [80.0, 80.0]])
    results = {
        "boxes": [torch.zeros((0, 4))],
        "scores": [torch.zeros((0,))],
        "classes": [np.zeros((0,), dtype=np.int32)],
        "segments": [[poly]],
        "points": [torch.zeros((0, 1, 3))],
    }
    out = rec.apply_coordinates(results, history)
    seg00 = out["segments"][0][0]
    assert seg00.shape[0] == 4
    xs, ys = seg00[:, 0], seg00[:, 1]
    assert xs.min().item() >= 80.0 - 1e-4 and xs.max().item() <= 100.0 + 1e-4
    assert ys.min().item() >= 40.0 - 1e-4 and ys.max().item() <= 80.0 + 1e-4
    seg01 = out["segments"][1][0]
    assert seg01[:, 0].max().item() <= 40.0 + 1e-4
    assert seg01[:, 0].min().item() >= 0.0 - 1e-4
    assert out["segments"][2] == []
    assert out["segments"][3] == []


def test_tile_apply_coords_masks_sliced_per_tile_and_drop_when_empty():
    _, rec, history = _tile_pipeline()
    mask = torch.zeros((1, 200, 200), dtype=torch.uint8)
    mask[0, 30:80, 30:80] = 1
    results = {
        "boxes": [torch.zeros((0, 4))],
        "scores": [torch.zeros((0,))],
        "classes": [np.zeros((0,), dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
        "masks": [mask],
    }
    out = rec.apply_coordinates(results, history)
    assert out["masks"][0].shape == (1, 100, 100)
    assert out["masks"][0][0, 30:80, 30:80].all() and out["masks"][0].sum() == 50 * 50
    for i in (1, 2, 3):
        assert out["masks"][i].shape == (0, 100, 100)


def test_tile_apply_coords_interpolation_mode_scales_coords():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(150, 150, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=100, scale_mode="interpolation")])
    results = {
        "boxes": [torch.tensor([[60.0, 60.0, 90.0, 90.0]])],
        "scores": [torch.tensor([1.0])],
        "classes": [np.array([0], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    out = rec.apply_coordinates(results, history)
    expected = {
        0: [[80.0, 80.0, 100.0, 100.0]],
        1: [[0.0, 80.0, 20.0, 100.0]],
        2: [[80.0, 0.0, 100.0, 20.0]],
        3: [[0.0, 0.0, 20.0, 20.0]],
    }
    for i, exp in expected.items():
        assert torch.allclose(out["boxes"][i], torch.tensor(exp), atol=1e-4), f"tile {i} got {out['boxes'][i]}"


def test_tile_2d_grayscale_round_trip_preserves_shape_and_values():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.arange(100 * 100, dtype=torch.float32).reshape(100, 100)
    tiles, history = pre.preprocess([img], [steps.tile(tile_size=50, stride=50)])

    assert len(tiles) == 4
    for t in tiles:
        assert t.dim() == 2
        assert t.shape == (50, 50)

    restored = rec.reconstruct_images(tiles, history)
    assert restored[0].dim() == 2
    assert restored[0].shape == (100, 100)
    assert torch.equal(restored[0], img)


def test_tile_forward_missing_required_keys_raises():
    with pytest.raises(ValueError, match="'tile_size' and 'stride' are required"):
        TileConfig(tile_size=16)
    with pytest.raises(ValueError, match="'tile_size' and 'stride' are required"):
        TileConfig(stride=16)


def test_tile_segments_variable_length_concat_across_tiles():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros((100, 100, 3))
    _tiles, history = pre.preprocess([img], [steps.tile(tile_size=50, stride=50)])

    seg_short = torch.tensor([[3.0, 4.0]])
    seg_long = torch.tensor([[1.0, 2.0], [10.0, 12.0], [25.0, 30.0], [40.0, 45.0], [49.0, 48.0]])
    results = _empty_results(n=4, segments=[[], [seg_short], [], [seg_long]])
    reverted = rec.reconstruct_coordinates(results, history)

    out_segs = reverted["segments"][0]
    assert len(out_segs) == 2
    assert torch.allclose(out_segs[0], torch.tensor([[53.0, 4.0]]))
    assert torch.allclose(
        out_segs[1],
        torch.tensor([[51.0, 52.0], [60.0, 62.0], [75.0, 80.0], [90.0, 95.0], [99.0, 98.0]]),
    )


def test_tile_masks_under_interpolation_mode():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros((90, 90, 3))
    _tiles, history = pre.preprocess([img], [steps.tile(tile_size=60, stride=60, scale_mode="interpolation")])

    tile3_mask = torch.ones((1, 60, 60), dtype=torch.float32)
    empty_mask = torch.zeros((0, 60, 60), dtype=torch.float32)
    results = _empty_results(n=4, masks=[empty_mask, empty_mask, empty_mask, tile3_mask])
    reverted = rec.reconstruct_coordinates(results, history)

    out = reverted["masks"][0]
    assert out.shape == (1, 90, 90)
    assert out[0, 45:90, 45:90].eq(1).all()
    assert out[0, :45, :].eq(0).all()
    assert out[0, :, :45].eq(0).all()


def test_tile_masks_under_padding_mode_cropped_to_im_size():
    # im_size 90 pads to scale_size 120 (tile 60, stride 60). Reverted masks must come
    # back at im_size to match the reverted image, not the padded scale_size.
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros((90, 90, 3))
    _tiles, history = pre.preprocess([img], [steps.tile(tile_size=60, stride=60, scale_mode="padding")])
    assert history[0].scale_sizes[0] == [120, 120]

    tile0_mask = torch.ones((1, 60, 60), dtype=torch.float32)
    empty_mask = torch.zeros((0, 60, 60), dtype=torch.float32)
    results = _empty_results(n=4, masks=[tile0_mask, empty_mask, empty_mask, empty_mask])
    reverted = rec.reconstruct_coordinates(results, history)

    out = reverted["masks"][0]
    assert out.shape == (1, 90, 90)
    assert out[0, :60, :60].eq(1).all()
    assert out[0, 60:, :].eq(0).all()
    assert out[0, :, 60:].eq(0).all()


def test_tile_apply_coords_image_level_label_propagates_to_all_tiles():
    # Classification-style result: only scores/classes, no geometry. With nothing to clip
    # against, the label should propagate to every tile rather than being dropped.
    _, rec, history = _tile_pipeline()
    results = {
        "scores": [torch.tensor([0.7])],
        "classes": [np.array([3], dtype=np.int32)],
    }
    out = rec.apply_coordinates(results, history)
    for i in range(4):
        assert out["scores"][i].tolist() == [pytest.approx(0.7)]
        assert out["classes"][i].tolist() == [3]


def test_tile_revert_coords_cursor_mismatch_raises():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros((100, 100, 3))
    _tiles, history = pre.preprocess([img], [steps.tile(tile_size=50, stride=50)])

    results = _empty_results(n=5)
    with pytest.raises(RuntimeError, match="Tile coord reconstruction mismatch"):
        rec.reconstruct_coordinates(results, history)


def test_tile_list_form_tile_size_and_stride_non_square():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.arange(80 * 120 * 3, dtype=torch.float32).reshape(80, 120, 3)
    tiles, history = pre.preprocess([img], [steps.tile(tile_size=[40, 60], stride=[40, 60])])

    assert len(tiles) == 4
    for t in tiles:
        assert t.shape == (40, 60, 3)

    restored = rec.reconstruct_images(tiles, history)
    assert restored[0].shape == (80, 120, 3)
    assert torch.equal(restored[0], img)

    det = torch.tensor([[5.0, 7.0, 25.0, 30.0]])
    results = _empty_results(n=4, boxes=[torch.zeros((0, 4)), torch.zeros((0, 4)), torch.zeros((0, 4)), det])
    reverted = rec.reconstruct_coordinates(results, history)
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[65.0, 47.0, 85.0, 70.0]]))


def test_tile_overlap_mode_max_differs_from_average():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.ones((80, 80, 3))

    def run(mode):
        tiles, history = pre.preprocess([img], [steps.tile(tile_size=50, stride=30, overlap_mode=mode)])
        assert history[0].overlap_modes[0] == mode
        tiles[0] = torch.zeros_like(tiles[0])
        return rec.reconstruct_images(tiles, history)[0]

    restored_max = run("max")
    restored_avg = run("average")

    px = (40, 40, 0)
    assert restored_max[px].item() == pytest.approx(1.0)
    assert restored_avg[px].item() == pytest.approx(0.75)


def test_tile_revert_images_cursor_mismatch_raises():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros((100, 100, 3))
    tiles, history = pre.preprocess([img], [steps.tile(tile_size=50, stride=50)])

    extra = tiles + [tiles[0].clone()]
    with pytest.raises(RuntimeError, match="Tile reconstruction mismatch"):
        rec.reconstruct_images(extra, history)
