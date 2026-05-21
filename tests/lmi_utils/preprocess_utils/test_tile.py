"""Focused tests for TileOperation behavior not covered by the pipeline/coord suites.

Existing coverage lives in:
- test_reconstruct_coordinates.py::TestTile / TestResizeTile / TestOBBReconstruction
- test_preprocess_pipeline.py (forward, nested tiling)
- test_error_handling.py (tile count mismatch)
- test_chain_pipeline.py (tile_with_flip_lossless)
- test_steps.py (builder kwargs)

This file fills the remaining gaps: 2D (grayscale, no channel axis) round-trip,
missing-required-keys validation in forward, and cursor mismatch in revert_images.
"""

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.ops.tile import TileOperation
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _empty_results(n=1, **overrides):
    """Per-image (per-tile, in tile context) results dict with empty defaults."""
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
    """Helper: build a Preprocessor+Reconstructor and run forward tile, returning (history, hwc image)."""
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(im_size, im_size, 3)
    cfg = {"tile_size": tile, "stride": stride, "scale_mode": scale_mode}
    _, history = pre.preprocess([img], [{"type": "tile", "configuration": cfg}])
    return pre, rec, history


def test_tile_apply_coords_xyxy_clips_and_drops_per_tile():
    """200x200 image, 4x 100x100 tiles. Box straddles vertical seam, stays in top row."""
    pre, rec, history = _tile_pipeline()
    results = {
        "boxes": [torch.tensor([[80.0, 50.0, 140.0, 90.0]])],
        "scores": [torch.tensor([0.9])],
        "classes": [np.array([5], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    out = rec.apply_coordinates(results, history)
    # Top-left tile: clipped to x in [80, 100]
    assert torch.allclose(out["boxes"][0], torch.tensor([[80.0, 50.0, 100.0, 90.0]]))
    assert out["scores"][0].tolist() == [pytest.approx(0.9)]
    assert out["classes"][0].tolist() == [5]
    # Top-right tile: clipped to x in [0, 40]
    assert torch.allclose(out["boxes"][1], torch.tensor([[0.0, 50.0, 40.0, 90.0]]))
    assert out["classes"][1].tolist() == [5]
    # Bottom-row tiles: dropped (no overlap)
    assert out["boxes"][2].shape == (0, 4)
    assert out["boxes"][3].shape == (0, 4)
    assert len(out["scores"][2]) == 0 and len(out["classes"][3]) == 0


def test_tile_apply_coords_obb_clipped_and_refit_to_min_area_rect():
    """OBB straddling x=100 seam is clipped per-tile then refit as a min-area rect (still 4 corners)."""
    _, rec, history = _tile_pipeline()
    # Axis-aligned quad centered at (100, 60), 60w x 40h.
    obb = torch.tensor([[[70.0, 40.0], [130.0, 40.0], [130.0, 80.0], [70.0, 80.0]]])
    results = {
        "boxes": [obb],
        "scores": [torch.tensor([0.5])],
        "classes": [np.array([0], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    out = rec.apply_coordinates(results, history)

    # Top-left tile: clipped to x in [70, 100], y in [40, 80] — refit is the same rect.
    assert out["boxes"][0].shape == (1, 4, 2)
    tl = out["boxes"][0][0]
    assert tl[:, 0].min().item() >= 70.0 - 1e-4 and tl[:, 0].max().item() <= 100.0 + 1e-4
    assert tl[:, 1].min().item() >= 40.0 - 1e-4 and tl[:, 1].max().item() <= 80.0 + 1e-4

    # Top-right tile: clipped to tile-local x in [0, 30], y in [40, 80].
    assert out["boxes"][1].shape == (1, 4, 2)
    tr = out["boxes"][1][0]
    assert tr[:, 0].min().item() >= 0.0 - 1e-4 and tr[:, 0].max().item() <= 30.0 + 1e-4
    assert tr[:, 1].min().item() >= 40.0 - 1e-4 and tr[:, 1].max().item() <= 80.0 + 1e-4

    # Bottom tiles dropped (no overlap).
    assert out["boxes"][2].shape == (0, 4, 2)
    assert out["boxes"][3].shape == (0, 4, 2)


def test_tile_apply_coords_points_visibility_zeroed_outside_tile():
    """Keypoints outside a tile get visibility=0; instance dropped only if all keypoints invisible."""
    _, rec, history = _tile_pipeline()
    # 1 instance, 3 keypoints: one inside top-left, one inside top-right, one inside bottom-left.
    pts = torch.tensor([[[30.0, 30.0, 2.0], [150.0, 30.0, 2.0], [30.0, 150.0, 2.0]]])
    results = {
        "boxes": [torch.zeros((0, 4))],
        "scores": [torch.zeros((0,))],
        "classes": [np.zeros((0,), dtype=np.int32)],
        "segments": [[]],
        "points": [pts],
    }
    out = rec.apply_coordinates(results, history)
    # Top-left tile: kp0 visible (30,30), others outside → vis=0
    assert out["points"][0].shape == (1, 3, 3)
    assert out["points"][0][0, 0].tolist() == [pytest.approx(30.0), pytest.approx(30.0), pytest.approx(2.0)]
    assert out["points"][0][0, 1, 2].item() == 0.0  # right kp zeroed
    assert out["points"][0][0, 2, 2].item() == 0.0  # bottom kp zeroed
    # Bottom-right tile: no keypoint inside → instance dropped
    assert out["points"][3].shape == (0, 3, 3)


def test_tile_apply_coords_segments_clipped_to_tile_rect():
    """A square segment straddling x=100 is clipped to a clean rect in each tile."""
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
    # Top-left: clipped to x in [80, 100], y in [40, 80] — 4 vertices.
    seg00 = out["segments"][0][0]
    assert seg00.shape[0] == 4
    xs, ys = seg00[:, 0], seg00[:, 1]
    assert xs.min().item() >= 80.0 - 1e-4 and xs.max().item() <= 100.0 + 1e-4
    assert ys.min().item() >= 40.0 - 1e-4 and ys.max().item() <= 80.0 + 1e-4
    # Top-right: clipped to x in [0, 40] in tile-local coords.
    seg01 = out["segments"][1][0]
    assert seg01[:, 0].max().item() <= 40.0 + 1e-4
    assert seg01[:, 0].min().item() >= 0.0 - 1e-4
    # Bottom tiles dropped (empty list).
    assert out["segments"][2] == []
    assert out["segments"][3] == []


def test_tile_apply_coords_masks_sliced_per_tile_and_drop_when_empty():
    """Single instance mask covering top-left quadrant: kept in (0,0), dropped elsewhere."""
    _, rec, history = _tile_pipeline()
    mask = torch.zeros((1, 200, 200), dtype=torch.uint8)
    mask[0, 30:80, 30:80] = 1  # entirely inside top-left tile
    results = {
        "boxes": [torch.zeros((0, 4))],
        "scores": [torch.zeros((0,))],
        "classes": [np.zeros((0,), dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
        "masks": [mask],
    }
    out = rec.apply_coordinates(results, history)
    # (0,0): kept; mask shape is tile size with ones in [30:80, 30:80]
    assert out["masks"][0].shape == (1, 100, 100)
    assert out["masks"][0][0, 30:80, 30:80].all() and out["masks"][0].sum() == 50 * 50
    # Other tiles: dropped (mask was zero there)
    for i in (1, 2, 3):
        assert out["masks"][i].shape == (0, 100, 100)


def test_tile_apply_coords_interpolation_mode_scales_coords():
    """In interpolation mode, coords are scaled into scale_size before per-tile shifting."""
    # im=150x150, scaled to scale=200x200 (so sx=sy=200/150=4/3), 2x2 tiles of 100x100.
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros(150, 150, 3)
    cfg = {"tile_size": 100, "stride": 100, "scale_mode": "interpolation"}
    _, history = pre.preprocess([img], [{"type": "tile", "configuration": cfg}])
    # Box at original (75, 75) → scale-space (100, 100) → on the seam.
    results = {
        "boxes": [torch.tensor([[60.0, 60.0, 90.0, 90.0]])],
        "scores": [torch.tensor([1.0])],
        "classes": [np.array([0], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    out = rec.apply_coordinates(results, history)
    # In scale-space, box is [80, 80, 120, 120]. Every tile should get a clipped slice.
    expected = {
        0: [[80.0, 80.0, 100.0, 100.0]],  # top-left clip
        1: [[0.0, 80.0, 20.0, 100.0]],  # top-right clip (x: 80-100 → tile-local 0-20)
        2: [[80.0, 0.0, 100.0, 20.0]],  # bottom-left clip
        3: [[0.0, 0.0, 20.0, 20.0]],  # bottom-right clip
    }
    for i, exp in expected.items():
        assert torch.allclose(out["boxes"][i], torch.tensor(exp), atol=1e-4), f"tile {i} got {out['boxes'][i]}"


def test_tile_2d_grayscale_round_trip_preserves_shape_and_values():
    """Forward+revert on a (H, W) image (no channel axis) returns the original (H, W)."""
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.arange(100 * 100, dtype=torch.float32).reshape(100, 100)  # 2D
    steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
    tiles, history = pre.preprocess([img], steps)

    assert len(tiles) == 4
    for t in tiles:
        assert t.dim() == 2
        assert t.shape == (50, 50)

    restored = rec.reconstruct_images(tiles, history)
    assert restored[0].dim() == 2
    assert restored[0].shape == (100, 100)
    assert torch.equal(restored[0], img)


def test_tile_forward_missing_required_keys_raises():
    """forward must reject configurations without tile_size or stride."""
    op = TileOperation()
    img = torch.zeros((50, 50, 3))
    with pytest.raises(ValueError, match="Tiler configuration must contain keys"):
        op.forward([img], {"tile_size": 16})  # missing 'stride'
    with pytest.raises(ValueError, match="Tiler configuration must contain keys"):
        op.forward([img], {"stride": 16})  # missing 'tile_size'


def test_tile_segments_variable_length_concat_across_tiles():
    """Segments are flattened across tiles; per-tile length can vary independently."""
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros((100, 100, 3))
    steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
    _tiles, history = pre.preprocess([img], steps)

    # 4 tiles row-major. Put a 1-point seg in tile (0,1) and a 5-point seg in tile (1,1).
    seg_short = torch.tensor([[3.0, 4.0]])  # tile (0,1) → +x=50, +y=0
    seg_long = torch.tensor([[1.0, 2.0], [10.0, 12.0], [25.0, 30.0], [40.0, 45.0], [49.0, 48.0]])  # tile (1,1) → +(50, 50)
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
    """scale_mode='interpolation' resizes instance masks by (sx, sy) and pastes at scaled offset."""
    pre, rec = Preprocessor(), Reconstructor()
    # 90x90, tile=60 stride=60 → scale_size=120 (90 doesn't fit 60 grid); sx=sy=90/120=0.75.
    img = torch.zeros((90, 90, 3))
    steps = [{"type": "tile", "configuration": {"tile_size": 60, "stride": 60, "scale_mode": "interpolation"}}]
    _tiles, history = pre.preprocess([img], steps)

    # Full-tile mask (1, 60, 60) of ones in tile (1, 1) → scaled to (45, 45), pasted at (45, 45).
    tile3_mask = torch.ones((1, 60, 60), dtype=torch.float32)
    empty_mask = torch.zeros((0, 60, 60), dtype=torch.float32)
    results = _empty_results(n=4, masks=[empty_mask, empty_mask, empty_mask, tile3_mask])
    reverted = rec.reconstruct_coordinates(results, history)

    out = reverted["masks"][0]
    assert out.shape == (1, 90, 90)
    assert out[0, 45:90, 45:90].eq(1).all()
    # Everything outside the pasted region stays zero.
    assert out[0, :45, :].eq(0).all()
    assert out[0, :, :45].eq(0).all()


def test_tile_revert_coords_cursor_mismatch_raises():
    """Passing more per-tile results than metadata accounts for triggers the coord cursor check."""
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros((100, 100, 3))
    steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
    _tiles, history = pre.preprocess([img], steps)

    # metadata expects 4 tiles; pass 5 → cursor=4, len=5 → mismatch.
    results = _empty_results(n=5)
    with pytest.raises(RuntimeError, match="Tile coord reconstruction mismatch"):
        rec.reconstruct_coordinates(results, history)


def test_tile_list_form_tile_size_and_stride_non_square():
    """tile_size and stride accept [h, w] lists; a non-square grid round-trips correctly."""
    pre, rec = Preprocessor(), Reconstructor()
    # 80(H)x120(W) with tile=[40, 60] stride=[40, 60] → 2x2 non-square tiles.
    img = torch.arange(80 * 120 * 3, dtype=torch.float32).reshape(80, 120, 3)
    steps = [{"type": "tile", "configuration": {"tile_size": [40, 60], "stride": [40, 60]}}]
    tiles, history = pre.preprocess([img], steps)

    assert len(tiles) == 4
    for t in tiles:
        assert t.shape == (40, 60, 3)

    # Image round-trip is exact (no overlap, no scaling).
    restored = rec.reconstruct_images(tiles, history)
    assert restored[0].shape == (80, 120, 3)
    assert torch.equal(restored[0], img)

    # Coord round-trip: a box in tile (1, 1) → offset (x=60, y=40).
    det = torch.tensor([[5.0, 7.0, 25.0, 30.0]])
    results = _empty_results(n=4, boxes=[torch.zeros((0, 4)), torch.zeros((0, 4)), torch.zeros((0, 4)), det])
    reverted = rec.reconstruct_coordinates(results, history)
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[65.0, 47.0, 85.0, 70.0]]))


def test_tile_overlap_mode_max_differs_from_average():
    """Non-default overlap_mode is recorded in metadata and forwarded to Tiler.untile.

    With max-mode blending, an overlap pixel takes the highest value among contributing
    tiles; with average it takes the mean. Zero-out one tile of an all-ones image and
    compare the reconstructed value at a 4-tile overlap pixel: max → 1.0, average → 0.75.
    """
    pre, rec = Preprocessor(), Reconstructor()
    # 80x80 image, tile=50 stride=30 → 2x2 grid, tiles overlap in rows 30-50 × cols 30-50.
    img = torch.ones((80, 80, 3))

    def run(mode):
        steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 30, "overlap_mode": mode}}]
        tiles, history = pre.preprocess([img], steps)
        assert history[0]["metadata"][0]["overlap_mode"] == mode
        tiles[0] = torch.zeros_like(tiles[0])  # zero out tile (0,0)
        return rec.reconstruct_images(tiles, history)[0]

    restored_max = run("max")
    restored_avg = run("average")

    # Pick a pixel inside the 4-tile overlap region (rows 30-50, cols 30-50).
    px = (40, 40, 0)
    assert restored_max[px].item() == pytest.approx(1.0)
    assert restored_avg[px].item() == pytest.approx(0.75)


def test_tile_revert_images_cursor_mismatch_raises():
    """Passing more tiles than the metadata accounts for triggers the cursor check."""
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.zeros((100, 100, 3))
    steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
    tiles, history = pre.preprocess([img], steps)

    # metadata expects 4 tiles; feeding 5 leaves one unconsumed → RuntimeError.
    extra = tiles + [tiles[0].clone()]
    with pytest.raises(RuntimeError, match="Tile reconstruction mismatch"):
        rec.reconstruct_images(extra, history)
