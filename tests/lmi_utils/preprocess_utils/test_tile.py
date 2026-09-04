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


def test_tile_apply_coords_rejects_obb():
    _, rec, history = _tile_pipeline()
    obb = torch.tensor([[[70.0, 40.0], [130.0, 40.0], [130.0, 80.0], [70.0, 80.0]]])
    results = {
        "boxes": [obb],
        "scores": [torch.tensor([0.5])],
        "classes": [np.array([0], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    with pytest.raises(ValueError, match="does not support oriented boxes"):
        rec.apply_coordinates(results, history)


def test_tile_apply_coords_rejects_keypoints():
    _, rec, history = _tile_pipeline()
    pts = torch.tensor([[[30.0, 30.0, 2.0], [150.0, 30.0, 2.0], [30.0, 150.0, 2.0]]])
    results = {
        "boxes": [torch.zeros((0, 4))],
        "scores": [torch.zeros((0,))],
        "classes": [np.zeros((0,), dtype=np.int32)],
        "segments": [[]],
        "points": [pts],
    }
    with pytest.raises(ValueError, match="does not support keypoints"):
        rec.apply_coordinates(results, history)


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


def _overlap_history(nms_iou=0.5, containment=None):
    # 150x150 with tile 100 / stride 50 -> 2x2 overlapping tiles; box [55,55,95,95] lands in all 4.
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(150, 150, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=50, nms_iou=nms_iou, containment=containment)])
    return rec, history


# Per-tile boxes that all map back to the same [55,55,95,95] object after un-shifting.
_DUP_TILE_BOXES = [
    torch.tensor([[55.0, 55.0, 95.0, 95.0]]),
    torch.tensor([[5.0, 55.0, 45.0, 95.0]]),
    torch.tensor([[55.0, 5.0, 95.0, 45.0]]),
    torch.tensor([[5.0, 5.0, 45.0, 45.0]]),
]


def test_tile_revert_coords_dedupe_suppresses_duplicate_boxes():
    rec, history = _overlap_history()
    results = _empty_results(
        n=4,
        boxes=_DUP_TILE_BOXES,
        scores=[torch.tensor([s]) for s in (0.9, 0.6, 0.7, 0.8)],
        classes=[np.array([0], np.int32) for _ in range(4)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([[55.0, 55.0, 95.0, 95.0]]))
    assert out["scores"][0].item() == pytest.approx(0.9)  # highest-scoring duplicate kept
    assert out["classes"][0].tolist() == [0]


def test_tile_revert_coords_dedupe_disabled_keeps_all_duplicates():
    rec, history = _overlap_history(nms_iou=None)
    results = _empty_results(
        n=4,
        boxes=_DUP_TILE_BOXES,
        scores=[torch.tensor([s]) for s in (0.9, 0.6, 0.7, 0.8)],
        classes=[np.array([0], np.int32) for _ in range(4)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (4, 4)


def test_tile_revert_coords_dedupe_is_class_aware():
    rec, history = _overlap_history()
    results = _empty_results(
        n=4,
        boxes=_DUP_TILE_BOXES,
        scores=[torch.tensor([s]) for s in (0.9, 0.8, 0.7, 0.6)],
        classes=[np.array([c], np.int32) for c in (0, 1, 0, 1)],
    )
    out = rec.reconstruct_coordinates(results, history)
    # one survivor per class: class 0 -> 0.9, class 1 -> 0.8
    assert out["boxes"][0].shape == (2, 4)
    assert sorted(out["scores"][0].tolist()) == [pytest.approx(0.8), pytest.approx(0.9)]
    assert sorted(out["classes"][0].tolist()) == [0, 1]


def test_tile_revert_coords_dedupe_masks_by_iou():
    rec, history = _overlap_history()
    # Same global region [55:95, 55:95] expressed in two overlapping top tiles' local coords.
    m0 = torch.zeros((1, 100, 100), dtype=torch.uint8)
    m0[0, 55:95, 55:95] = 1
    m1 = torch.zeros((1, 100, 100), dtype=torch.uint8)
    m1[0, 55:95, 5:45] = 1  # col offset 50 -> same global cols 55:95
    empty = torch.zeros((0, 100, 100), dtype=torch.uint8)
    results = _empty_results(
        n=4,
        scores=[torch.tensor([0.9]), torch.tensor([0.7]), torch.zeros((0,)), torch.zeros((0,))],
        masks=[m0, m1, empty, empty],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["masks"][0].shape == (1, 150, 150)
    assert out["masks"][0][0, 55:95, 55:95].all() and int(out["masks"][0].sum()) == 40 * 40
    assert out["scores"][0].item() == pytest.approx(0.9)


def test_tile_revert_coords_dedupe_segments_via_polygon_iou():
    rec, history = _overlap_history()
    # identical global square polygon seen in two overlapping top tiles
    seg0 = torch.tensor([[55.0, 55.0], [95.0, 55.0], [95.0, 95.0], [55.0, 95.0]])
    seg1 = torch.tensor([[5.0, 55.0], [45.0, 55.0], [45.0, 95.0], [5.0, 95.0]])  # +50 in x -> same square
    results = _empty_results(
        n=4,
        scores=[torch.tensor([0.9]), torch.tensor([0.7]), torch.zeros((0,)), torch.zeros((0,))],
        segments=[[seg0], [seg1], [], []],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert len(out["segments"][0]) == 1
    assert torch.allclose(out["segments"][0][0], torch.tensor([[55.0, 55.0], [95.0, 55.0], [95.0, 95.0], [55.0, 95.0]]))
    assert out["scores"][0].item() == pytest.approx(0.9)


def test_tile_merge_fragments_spans_three_tiles():
    # 250x100 with tile 100 / stride 80 -> 3 tiles at x = 0, 80, 160; the object covers all three
    # and the middle tile sees only its interior, with no edge of the object visible.
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(100, 250, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=80, merge_fragments=True)])
    results = _empty_results(
        n=3,
        boxes=[
            torch.tensor([[40.0, 20.0, 100.0, 50.0]]),
            torch.tensor([[0.0, 20.0, 100.0, 50.0]]),
            torch.tensor([[0.0, 20.0, 80.0, 50.0]]),
        ],
        scores=[torch.tensor([s]) for s in (0.4, 0.3, 0.5)],
        classes=[np.array([0], np.int32) for _ in range(3)],
        points=[torch.zeros((0, 1, 3)) for _ in range(3)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([[40.0, 20.0, 240.0, 50.0]]))
    assert out["scores"][0].item() == pytest.approx(0.5)


def test_tile_merge_fragments_rejects_zero_overlap():
    with pytest.raises(ValueError, match="needs overlap"):
        steps.tile(tile_size=100, stride=100, merge_fragments=True)


def test_tile_revert_coords_rejects_keypoints():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(100, 250, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=80)])
    results = _empty_results(n=3, points=[torch.zeros((1, 2, 3)) for _ in range(3)])
    with pytest.raises(ValueError, match="does not support keypoints"):
        rec.reconstruct_coordinates(results, history)


def test_tile_containment_nms_drops_a_fragment_nested_in_a_whole_detection():
    rec, history = _overlap_history(containment=0.8)
    results = _empty_results(
        n=4,
        boxes=[
            torch.tensor([[10.0, 10.0, 90.0, 90.0]]),  # whole object, tile (0, 0)
            torch.tensor([[10.0, 10.0, 30.0, 30.0]]),  # -> global [60, 10, 80, 30], nested in it
            torch.zeros((0, 4)),
            torch.zeros((0, 4)),
        ],
        scores=[torch.tensor([0.9]), torch.tensor([0.6]), torch.zeros((0,)), torch.zeros((0,))],
        classes=[np.array([0], np.int32), np.array([0], np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([[10.0, 10.0, 90.0, 90.0]]))


def test_tile_containment_disabled_keeps_the_nested_detection():
    rec, history = _overlap_history(containment=None)
    results = _empty_results(
        n=4,
        boxes=[
            torch.tensor([[10.0, 10.0, 90.0, 90.0]]),
            torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            torch.zeros((0, 4)),
            torch.zeros((0, 4)),
        ],
        scores=[torch.tensor([0.9]), torch.tensor([0.6]), torch.zeros((0,)), torch.zeros((0,))],
        classes=[np.array([0], np.int32), np.array([0], np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (2, 4)


def _padded_history():
    # 120x120 with tile 100 / stride 50 pads out to 150x150, so x/y 120..150 is padding.
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(120, 120, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=50, nms_iou=None, containment=None)])
    return rec, history


def test_tile_drops_predictions_that_land_in_the_padding():
    rec, history = _padded_history()
    # tile 1 starts at x = 50; local [80, 10, 95, 30] -> global [130, 10, 145, 30], all padding.
    results = _empty_results(
        n=4,
        boxes=[torch.zeros((0, 4)), torch.tensor([[80.0, 10.0, 95.0, 30.0]]), torch.zeros((0, 4)), torch.zeros((0, 4))],
        scores=[torch.zeros((0,)), torch.tensor([0.9]), torch.zeros((0,)), torch.zeros((0,))],
        classes=[np.zeros((0,), np.int32), np.array([0], np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (0, 4)


def test_tile_clips_predictions_that_straddle_the_image_edge():
    rec, history = _padded_history()
    # global [110, 10, 140, 30] -> clipped to the 120-wide image.
    results = _empty_results(
        n=4,
        boxes=[torch.zeros((0, 4)), torch.tensor([[60.0, 10.0, 90.0, 30.0]]), torch.zeros((0, 4)), torch.zeros((0, 4))],
        scores=[torch.zeros((0,)), torch.tensor([0.9]), torch.zeros((0,)), torch.zeros((0,))],
        classes=[np.zeros((0,), np.int32), np.array([0], np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert torch.allclose(out["boxes"][0], torch.tensor([[110.0, 10.0, 120.0, 30.0]]))


def test_tile_score_threshold_applies_after_merging():
    # Neither fragment clears 0.6 alone; the merged object does.
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(100, 250, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=80, merge_fragments=True, score_threshold=0.6)])
    results = _empty_results(
        n=3,
        boxes=[
            torch.tensor([[40.0, 20.0, 100.0, 50.0]]),
            torch.tensor([[0.0, 20.0, 100.0, 50.0]]),
            torch.tensor([[0.0, 20.0, 80.0, 50.0]]),
        ],
        scores=[torch.tensor([s]) for s in (0.4, 0.3, 0.7)],
        classes=[np.array([0], np.int32) for _ in range(3)],
        points=[torch.zeros((0, 1, 3)) for _ in range(3)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (1, 4)
    assert out["scores"][0].item() == pytest.approx(0.7)


def test_tile_apply_coords_drops_sliver_labels_but_keeps_interior_fragments():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(200, 200, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=100, min_label_size=5)])
    results = {
        # 1) a wide label leaving a 2px sliver in tile 1; 2) a label wholly inside tile 0.
        "boxes": [torch.tensor([[50.0, 50.0, 102.0, 90.0], [10.0, 10.0, 40.0, 40.0]])],
        "scores": [torch.tensor([0.9, 0.8])],
        "classes": [np.array([0, 1], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    out = rec.apply_coordinates(results, history)
    assert out["boxes"][0].shape == (2, 4)  # tile 0 keeps both
    assert out["boxes"][1].shape == (0, 4)  # tile 1's 2px sliver is dropped


def test_tile_merge_fragments_rejects_interpolation():
    with pytest.raises(ValueError, match="scale_mode='padding'"):
        steps.tile(tile_size=100, stride=80, merge_fragments=True, scale_mode="interpolation")


def _merge_history(**kwargs):
    # 150x100 with tile 100 / stride 60 -> 2 tiles at x = 0 and x = 60, overlapping by 40.
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(100, 150, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=60, merge_fragments=True, **kwargs)])
    return rec, history


def _two_tile_results(boxes, scores):
    return _empty_results(
        n=2,
        boxes=[torch.tensor(b, dtype=torch.float32) for b in boxes],
        scores=[torch.tensor(s, dtype=torch.float32) for s in scores],
        classes=[np.zeros(len(s), np.int32) for s in scores],
        points=[torch.zeros((0, 1, 3)) for _ in range(2)],
    )


def test_tile_two_objects_abutting_at_a_seam_survive_as_two():
    # A is x 20..80, B is x 80..140. Each tile sees one whole and clips the other at the seam;
    # the fragments must neither merge the two objects nor survive NMS.
    results = _two_tile_results(
        boxes=[[[20, 20, 80, 50], [80, 20, 100, 50]], [[0, 20, 20, 50], [20, 20, 80, 50]]],
        scores=[[0.9, 0.5], [0.5, 0.9]],
    )
    rec, history = _merge_history()
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (2, 4)
    assert sorted(out["boxes"][0][:, 0].tolist()) == [pytest.approx(20.0), pytest.approx(80.0)]
    assert sorted(out["boxes"][0][:, 2].tolist()) == [pytest.approx(80.0), pytest.approx(140.0)]


def test_tile_two_whole_objects_in_the_overlap_band_survive_as_two():
    # Both objects sit inside the overlap, so both tiles see both whole. Nothing is truncated;
    # NMS must collapse each pair of duplicates without merging the two objects together.
    results = _two_tile_results(
        boxes=[[[68, 20, 78, 50], [82, 20, 92, 50]], [[8, 20, 18, 50], [22, 20, 32, 50]]],
        scores=[[0.9, 0.8], [0.7, 0.6]],
    )
    rec, history = _merge_history()
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (2, 4)
    assert sorted(out["boxes"][0][:, 0].tolist()) == [pytest.approx(68.0), pytest.approx(82.0)]
    assert sorted(out["scores"][0].tolist()) == [pytest.approx(0.8), pytest.approx(0.9)]


def test_tile_merge_overlap_is_validated_per_axis():
    # 40px of overlap on one axis does not excuse 2px on the other.
    with pytest.raises(ValueError, match="needs overlap"):
        steps.tile(tile_size=[100, 100], stride=[60, 98], merge_fragments=True)


def test_tile_merge_fragments_with_non_square_tiles():
    # tile [80, 100] / stride [50, 60] over a 130x150 image -> 2x2 grid at y = 0, 50 and x = 0, 60.
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(130, 150, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=[80, 100], stride=[50, 60], merge_fragments=True)])
    results = _empty_results(
        n=4,
        boxes=[
            torch.tensor([[40.0, 30.0, 100.0, 80.0]]),  # tile (0, 0)
            torch.tensor([[0.0, 30.0, 60.0, 80.0]]),  # tile (0, 1) -> global x 60..120
            torch.tensor([[40.0, 0.0, 100.0, 60.0]]),  # tile (1, 0) -> global y 50..110
            torch.tensor([[0.0, 0.0, 60.0, 60.0]]),  # tile (1, 1)
        ],
        scores=[torch.tensor([s]) for s in (0.3, 0.4, 0.5, 0.6)],
        classes=[np.array([0], np.int32) for _ in range(4)],
        points=[torch.zeros((0, 1, 3)) for _ in range(4)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([[40.0, 30.0, 120.0, 110.0]]))


def test_tile_merge_with_no_predictions_at_all():
    rec, history = _merge_history()
    out = rec.reconstruct_coordinates(_empty_results(n=2), history)
    assert out["boxes"][0].shape == (0, 4)


def test_tile_min_overlap_follows_the_edge_tolerance():
    # tile 100 / stride 60 leaves 40px of overlap: fine by default, not enough at a tolerance of 20.
    steps.tile(tile_size=100, stride=60, merge_fragments=True)
    with pytest.raises(ValueError, match="2 x edge_tolerance 20"):
        steps.tile(tile_size=100, stride=60, merge_fragments=True, edge_tolerance=20)


def test_tile_edge_tolerance_reaches_the_merge_step():
    # Both halves stop 8px short of the seam, so only the wider tolerance pairs them.
    boxes = [[[40, 20, 92, 50]], [[0, 20, 60, 50]]]
    scores = [[0.4], [0.9]]
    rec, history = _merge_history()
    assert rec.reconstruct_coordinates(_two_tile_results(boxes, scores), history)["boxes"][0].shape == (2, 4)
    rec, history = _merge_history(edge_tolerance=10)
    out = rec.reconstruct_coordinates(_two_tile_results(boxes, scores), history)
    assert out["boxes"][0].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([[40.0, 20.0, 120.0, 50.0]]))


def _two_dogs_either_side_of_a_seam(stride):
    """Two separate same-class objects meeting at x=320: one ends at the seam, the other starts there."""
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(320, 576, 3)  # 2 tiles wide at either stride
    _, history = pre.preprocess([img], [steps.tile(tile_size=320, stride=[320, stride])])
    left_local = torch.tensor([[200.0, 100.0, 320.0, 200.0]])  # tile 0, ends on its right edge
    right_local = torch.tensor([[float(320 - stride), 100.0, float(440 - stride), 200.0]])  # tile 1, global 320..440
    results = _empty_results(
        n=2,
        boxes=[left_local, right_local],
        scores=[torch.tensor([0.9]), torch.tensor([0.8])],
        classes=[np.array([0], np.int32) for _ in range(2)],
        points=[torch.zeros((0, 1, 3)) for _ in range(2)],
    )
    return rec.reconstruct_coordinates(results, history)


def test_tile_merge_defaults_to_auto_and_skips_without_overlap():
    """Zero overlap puts both tile edges on one line, so two touching objects would fuse. Skip instead."""
    out = _two_dogs_either_side_of_a_seam(stride=320)
    assert out["boxes"][0].shape == (2, 4)


def test_tile_merge_auto_runs_once_there_is_overlap():
    """With overlap the neighbour sees the right-hand object whole, so the two stay separate..."""
    out = _two_dogs_either_side_of_a_seam(stride=256)
    assert out["boxes"][0].shape == (2, 4)


def test_tile_merge_auto_unions_a_genuinely_cut_object():
    """...while one object actually cut by the seam is rejoined, with no merge_fragments passed."""
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(100, 150, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=60)])  # 40px overlap, auto
    results = _empty_results(
        n=2,
        boxes=[torch.tensor([[40.0, 20.0, 100.0, 50.0]]), torch.tensor([[0.0, 20.0, 60.0, 50.0]])],
        scores=[torch.tensor([0.4]), torch.tensor([0.9])],
        classes=[np.array([0], np.int32) for _ in range(2)],
        points=[torch.zeros((0, 1, 3)) for _ in range(2)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([[40.0, 20.0, 120.0, 50.0]]))


def test_tile_revert_coords_rejects_oriented_boxes():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(100, 250, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=80)])
    obb = torch.tensor([[[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]]])
    results = _empty_results(
        n=3,
        boxes=[obb, torch.zeros((0, 4, 2)), torch.zeros((0, 4, 2))],
        scores=[torch.tensor([0.9]), torch.zeros((0,)), torch.zeros((0,))],
        classes=[np.array([0], np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32)],
    )
    with pytest.raises(ValueError, match="does not support oriented boxes"):
        rec.reconstruct_coordinates(results, history)


def test_tile_rejects_keypoints_even_with_merging_off():
    """Turning merging off is not a way in: tiling itself has no rule for keypoints."""
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(100, 250, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=80, merge_fragments=False)])
    results = _empty_results(n=3, points=[torch.zeros((1, 2, 3)) for _ in range(3)])
    with pytest.raises(ValueError, match="does not support keypoints"):
        rec.reconstruct_coordinates(results, history)


def test_tile_merge_false_stays_off_with_overlap():
    """Explicitly off must not merge a cut object, even where auto would."""
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(100, 150, 3)
    _, history = pre.preprocess([img], [steps.tile(tile_size=100, stride=60, merge_fragments=False)])
    results = _empty_results(
        n=2,
        boxes=[torch.tensor([[40.0, 20.0, 100.0, 50.0]]), torch.tensor([[0.0, 20.0, 60.0, 50.0]])],
        scores=[torch.tensor([0.4]), torch.tensor([0.9])],
        classes=[np.array([0], np.int32) for _ in range(2)],
        points=[torch.zeros((0, 1, 3)) for _ in range(2)],
    )
    out = rec.reconstruct_coordinates(results, history)
    assert out["boxes"][0].shape == (2, 4)


def test_tile_merge_auto_does_not_validate_overlap_at_construction():
    """Only an explicit True demands enough overlap; the default accepts any grid."""
    steps.tile(tile_size=100, stride=100)
    steps.tile(tile_size=100, stride=80, scale_mode="interpolation")
