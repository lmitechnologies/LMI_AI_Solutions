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
