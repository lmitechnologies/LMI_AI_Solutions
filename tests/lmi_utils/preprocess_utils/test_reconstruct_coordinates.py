import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


@pytest.fixture
def pipeline():
    return Preprocessor(), Reconstructor()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_full_results(boxes_list, tile_hw=None):
    """
    Build a results dict with all coordinate fields populated.

    - boxes:    (N, 4) [x1, y1, x2, y2]
    - segments: 4-corner polygon per box, list of (4, 2) tensors — distinct structure
    - points:   box centroid + visibility=1, (N, 1, 3)             — distinct values
    - masks:    all-ones (N, H, W) — only included when tile_hw=(H, W) is given
    """
    segs_list, pts_list, masks_list = [], [], []
    for boxes in boxes_list:
        if len(boxes) == 0:
            segs_list.append([])
            pts_list.append(torch.zeros((0, 1, 3)))
            if tile_hw is not None:
                masks_list.append(torch.zeros((0, tile_hw[0], tile_hw[1])))
        else:
            segs_list.append([torch.stack([b[[0, 1]], b[[2, 1]], b[[2, 3]], b[[0, 3]]]) for b in boxes])
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            xy = torch.stack([cx, cy], dim=-1).unsqueeze(1)  # (N, 1, 2)
            pts_list.append(torch.cat([xy, torch.ones(len(boxes), 1, 1)], dim=-1))  # (N, 1, 3)
            if tile_hw is not None:
                masks_list.append(torch.ones(len(boxes), tile_hw[0], tile_hw[1]))

    result = {
        "boxes": boxes_list,
        "scores": [torch.ones(len(b)) for b in boxes_list],
        "classes": [np.zeros(len(b), dtype=np.int32) for b in boxes_list],
        "segments": segs_list,
        "points": pts_list,
    }
    if tile_hw is not None:
        result["masks"] = masks_list
    return result


def _corners(boxes):
    """Expected 4-corner polygon list matching _make_full_results segments."""
    return [torch.stack([b[[0, 1]], b[[2, 1]], b[[2, 3]], b[[0, 3]]]) for b in boxes]


def _centroids(boxes):
    """Expected (N, 1, 3) centroid+visibility tensor matching _make_full_results points."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    xy = torch.stack([cx, cy], dim=-1).unsqueeze(1)  # (N, 1, 2)
    return torch.cat([xy, torch.ones(len(boxes), 1, 1)], dim=-1)  # (N, 1, 3)


def _assert_mask_quadrant(canvas, det_idx, y_slice, x_slice, h, w):
    """Assert a single mask is 1.0 inside the given slices and 0.0 everywhere else."""
    ref = torch.zeros(h, w)
    ref[y_slice, x_slice] = 1.0
    assert torch.all(canvas[det_idx] == ref), f"mask[{det_idx}] placement mismatch"


def _assert_coords(reverted, image_idx, expected_boxes, atol=1.0):
    """Assert boxes, segments, and points all match the expected transformation."""
    boxes = reverted["boxes"][image_idx].float()
    assert torch.allclose(boxes, expected_boxes.float(), atol=atol), f"boxes mismatch: {boxes}"

    for seg, exp in zip(reverted["segments"][image_idx], _corners(expected_boxes)):
        assert torch.allclose(seg.float(), exp.float(), atol=atol), "segment mismatch"

    pts = reverted["points"][image_idx].float()
    assert torch.allclose(pts, _centroids(expected_boxes).float(), atol=atol), "points mismatch"


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


class TestResize:
    def test_uniform_scale(self, pipeline):
        """2x downscale (200→100) should double every coordinate on revert; masks are resized back."""
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": False}}]
        _, history = prep.preprocess(image, steps)

        boxes = torch.tensor([[10.0, 20.0, 40.0, 45.0]])
        results = _make_full_results([boxes])
        results["masks"] = [torch.ones(1, 100, 100)]
        reverted = recon.reconstruct_coordinates(results, history)
        _assert_coords(reverted, 0, torch.tensor([[20.0, 40.0, 80.0, 90.0]]))
        assert reverted["masks"][0].shape == (1, 200, 200)

    def test_independent_xy_scales(self, pipeline):
        """x and y axes scale independently; each reverts by its own factor; masks revert to original dims."""
        prep, recon = pipeline
        # 300(H)×200(W) → 100×100: x_revert=×2, y_revert=×3
        image = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
        steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": False}}]
        _, history = prep.preprocess(image, steps)

        boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
        results = _make_full_results([boxes])
        results["masks"] = [torch.ones(1, 100, 100)]
        reverted = recon.reconstruct_coordinates(results, history)
        # x: ×2 → [20, 80];  y: ×3 → [60, 180]
        _assert_coords(reverted, 0, torch.tensor([[20.0, 60.0, 80.0, 180.0]]))
        assert reverted["masks"][0].shape == (1, 300, 200)

    def test_empty_boxes_all_keys_preserved(self, pipeline):
        """All result keys survive revert even when every coordinate field is an empty tensor."""
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": False}}]
        _, history = prep.preprocess(image, steps)

        empty = torch.zeros((0, 4))
        results = _make_full_results([empty])
        reverted = recon.reconstruct_coordinates(results, history)

        assert set(reverted.keys()) == set(results.keys())
        assert len(reverted["boxes"][0]) == 0
        assert len(reverted["segments"][0]) == 0
        assert len(reverted["points"][0]) == 0

    def test_multiple_images_resize(self, pipeline):
        """Reversion is applied independently per image; each image's masks revert to its own original dims."""
        prep, recon = pipeline
        images = [np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8) for _ in range(2)]
        steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": False}}]
        _, history = prep.preprocess(images, steps)

        boxes0 = torch.tensor([[10.0, 15.0, 30.0, 35.0]])
        boxes1 = torch.tensor([[20.0, 25.0, 45.0, 40.0]])
        results = _make_full_results([boxes0, boxes1])
        results["masks"] = [torch.ones(1, 100, 100), torch.ones(1, 100, 100)]
        reverted = recon.reconstruct_coordinates(results, history)

        assert len(reverted["boxes"]) == 2
        _assert_coords(reverted, 0, torch.tensor([[20.0, 30.0, 60.0, 70.0]]))
        _assert_coords(reverted, 1, torch.tensor([[40.0, 50.0, 90.0, 80.0]]))
        assert reverted["masks"][0].shape == (1, 200, 200)
        assert reverted["masks"][1].shape == (1, 200, 200)

    def test_preserve_aspect_resize(self, pipeline):
        """preserve_aspect=True pads after resize; masks are unpadded then resized back to original dims."""
        prep, recon = pipeline
        # 200(H)×100(W) → 100×100 preserve_aspect: resize to 50×100, pad 25 left/right
        image = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
        steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": True}}]
        _, history = prep.preprocess(image, steps)

        boxes = torch.tensor([[35.0, 10.0, 75.0, 90.0]])
        results = _make_full_results([boxes])
        results["masks"] = [torch.ones(1, 100, 100)]
        reverted = recon.reconstruct_coordinates(results, history)

        # Undo pad (crop 25 from each side): 100×100 → 100×50
        # Undo resize (to orig 200H×100W): 100×50 → 200×100
        assert reverted["masks"][0].shape == (1, 200, 100)
        _assert_coords(reverted, 0, torch.tensor([[20.0, 20.0, 100.0, 180.0]]))


# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------


class TestTile:
    def test_shifts_all_fields_by_tile_offset(self, pipeline):
        """Each tile's boxes, segments, points, and masks are shifted by that tile's offset."""
        prep, recon = pipeline
        # 100×100, tile=50, stride=50 → 4 non-overlapping tiles, canvas=(100,100)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
        _, history = prep.preprocess(image, steps)

        # Row-major order: (0,0)→(x=0,y=0), (0,1)→(50,0), (1,0)→(0,50), (1,1)→(50,50)
        tile_box = torch.tensor([[3.0, 7.0, 15.0, 25.0]])
        results = _make_full_results([tile_box] * 4, tile_hw=(50, 50))
        reverted = recon.reconstruct_coordinates(results, history)

        assert len(reverted["boxes"]) == 1
        expected = torch.tensor(
            [
                [3.0, 7.0, 15.0, 25.0],
                [53.0, 7.0, 65.0, 25.0],
                [3.0, 57.0, 15.0, 75.0],
                [53.0, 57.0, 65.0, 75.0],
            ]
        )
        _assert_coords(reverted, 0, expected, atol=1e-3)

        out = reverted["masks"][0]  # (4, 100, 100)
        assert out.shape == (4, 100, 100)
        _assert_mask_quadrant(out, 0, slice(0, 50), slice(0, 50), 100, 100)
        _assert_mask_quadrant(out, 1, slice(0, 50), slice(50, 100), 100, 100)
        _assert_mask_quadrant(out, 2, slice(50, 100), slice(0, 50), 100, 100)
        _assert_mask_quadrant(out, 3, slice(50, 100), slice(50, 100), 100, 100)

    def test_empty_detections_per_tile(self, pipeline):
        """Tiles with no detections produce empty results for every field, and all keys are preserved."""
        prep, recon = pipeline
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
        _, history = prep.preprocess(image, steps)

        empty = torch.zeros((0, 4))
        results = _make_full_results([empty] * 4, tile_hw=(50, 50))
        reverted = recon.reconstruct_coordinates(results, history)

        assert set(reverted.keys()) == set(results.keys())
        assert len(reverted["boxes"][0]) == 0
        assert len(reverted["segments"][0]) == 0
        assert len(reverted["points"][0]) == 0
        assert len(reverted["masks"][0]) == 0

    def test_partial_detections_across_tiles(self, pipeline):
        """Only tiles with detections contribute to the merged output."""
        prep, recon = pipeline
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
        _, history = prep.preprocess(image, steps)

        empty = torch.zeros((0, 4))
        det = torch.tensor([[6.0, 9.0, 22.0, 35.0]])
        # Only tile (row=1, col=1) has a detection → offset (x=50, y=50)
        results = _make_full_results([empty, empty, empty, det], tile_hw=(50, 50))
        reverted = recon.reconstruct_coordinates(results, history)

        _assert_coords(reverted, 0, torch.tensor([[56.0, 59.0, 72.0, 85.0]]), atol=1e-3)

        out = reverted["masks"][0]  # (1, 100, 100)
        assert out.shape == (1, 100, 100)
        assert torch.all(out[0, 50:100, 50:100] == 1.0)
        assert torch.all(out[0, :50, :] == 0.0) and torch.all(out[0, 50:, :50] == 0.0)

    def test_multiple_images_tile(self, pipeline):
        """Each original image's tiles are grouped and merged independently."""
        prep, recon = pipeline
        images = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(2)]
        steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
        _, history = prep.preprocess(images, steps)

        # 8 tiles total (4 per image), each with 1 detection
        tile_box = torch.tensor([[4.0, 11.0, 18.0, 28.0]])
        results = _make_full_results([tile_box] * 8, tile_hw=(50, 50))
        reverted = recon.reconstruct_coordinates(results, history)

        assert len(reverted["boxes"]) == 2
        for i in range(2):
            assert len(reverted["boxes"][i]) == 4
            assert len(reverted["segments"][i]) == 4
            assert reverted["masks"][i].shape == (4, 100, 100)

    def test_padding_mode_canvas_matches_padded_size(self, pipeline):
        """When scale_mode='padding' pads the image to fit the tile grid, the mask canvas
        is the padded scale_size rather than the original image size."""
        prep, recon = pipeline
        # 90×90 with tile=50, stride=50: (90-50)%50=40≠0 → scale_size=100 (padded to fit 2×2)
        # is_interp=False → no coord scaling; canvas=(100,100), not (90,90)
        image = np.random.randint(0, 256, (90, 90, 3), dtype=np.uint8)
        steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
        _, history = prep.preprocess(image, steps)

        empty = torch.zeros((0, 4))
        det = torch.tensor([[3.0, 4.0, 12.0, 14.0]])
        # Tile (row=1, col=1): offset (x=50, y=50), no scale factor (padding mode)
        results = _make_full_results([empty, empty, empty, det], tile_hw=(50, 50))
        reverted = recon.reconstruct_coordinates(results, history)

        _assert_coords(reverted, 0, torch.tensor([[53.0, 54.0, 62.0, 64.0]]), atol=1e-3)

        out = reverted["masks"][0]  # canvas is (100, 100), not (90, 90)
        assert out.shape == (1, 100, 100)
        assert torch.all(out[0, 50:100, 50:100] == 1.0)
        assert torch.all(out[0, :50, :] == 0.0) and torch.all(out[0, 50:, :50] == 0.0)

    def test_overlapping_stride_shifts_all_fields(self, pipeline):
        """stride < tile_size → overlapping tiles; offsets are still col*stride / row*stride."""
        prep, recon = pipeline
        # 90×90, tile=60, stride=30 → 2×2 grid; offsets (x,y): (0,0),(30,0),(0,30),(30,30)
        image = np.random.randint(0, 256, (90, 90, 3), dtype=np.uint8)
        steps = [{"type": "tile", "configuration": {"tile_size": 60, "stride": 30}}]
        _, history = prep.preprocess(image, steps)

        tile_box = torch.tensor([[5.0, 8.0, 20.0, 25.0]])
        results = _make_full_results([tile_box] * 4, tile_hw=(60, 60))
        reverted = recon.reconstruct_coordinates(results, history)

        expected = torch.tensor(
            [
                [5.0, 8.0, 20.0, 25.0],  # tile (0,0): no offset
                [35.0, 8.0, 50.0, 25.0],  # tile (0,1): x+30
                [5.0, 38.0, 20.0, 55.0],  # tile (1,0): y+30
                [35.0, 38.0, 50.0, 55.0],  # tile (1,1): x+30, y+30
            ]
        )
        _assert_coords(reverted, 0, expected, atol=1e-3)

        # Tile (1,1) mask (det index 3) is pasted at (paste_y=30, paste_x=30)
        out = reverted["masks"][0]  # (4, 90, 90)
        assert out.shape == (4, 90, 90)
        assert torch.all(out[3, 30:90, 30:90] == 1.0)
        assert torch.all(out[3, :30, :] == 0.0) and torch.all(out[3, :, :30] == 0.0)

    def test_overlapping_stride_three_column_grid_middle_tile(self, pipeline):
        """stride=25, tile=50 on a 100x50 (HxW) image → 1x3 grid; middle tile offset is x=25."""
        prep, recon = pipeline
        image = np.random.randint(0, 256, (50, 100, 3), dtype=np.uint8)
        steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 25}}]
        _, history = prep.preprocess(image, steps)

        empty = torch.zeros((0, 4))
        det = torch.tensor([[2.0, 6.0, 18.0, 30.0]])
        # Tile index 1 is (row=0, col=1) → offset x=25, y=0
        reverted = recon.reconstruct_coordinates(_make_full_results([empty, det, empty]), history)

        _assert_coords(reverted, 0, torch.tensor([[27.0, 6.0, 43.0, 30.0]]), atol=1e-3)


# ---------------------------------------------------------------------------
# Resize + Tile combined
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_resize_then_tile_reverts_both(self, pipeline):
        """Coordinates pass through tile revert then resize revert in order."""
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        steps = [
            {"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": False}},
            {"type": "tile", "configuration": {"tile_size": 50, "stride": 50}},
        ]
        _, history = prep.preprocess(image, steps)

        # Detection in tile (row=1, col=1): box [7, 12, 25, 38]
        # Revert tile (offset x=50, y=50): [57, 62, 75, 88]
        # Revert resize (scale=2):          [114, 124, 150, 176]
        empty = torch.zeros((0, 4))
        det = torch.tensor([[7.0, 12.0, 25.0, 38.0]])
        reverted = recon.reconstruct_coordinates(_make_full_results([empty, empty, empty, det]), history)

        _assert_coords(reverted, 0, torch.tensor([[114.0, 124.0, 150.0, 176.0]]))

    def test_tile_internal_interpolation_scales_coordinates(self, pipeline):
        """scale_mode='interpolation' rescales the image to fit tiles; coordinate revert
        undoes the scale factor (sx=sy=90/120=0.75 here) after applying the tile offset."""
        prep, recon = pipeline
        # 90×90 doesn't divide evenly into 60×60 tiles; tiler upscales to 120×120
        image = np.random.randint(0, 256, (90, 90, 3), dtype=np.uint8)
        steps = [{"type": "tile", "configuration": {"tile_size": 60, "stride": 60, "scale_mode": "interpolation"}}]
        _, history = prep.preprocess(image, steps)

        # Tile (row=1, col=1) offset (x=60, y=60):
        # [4,8,20,20] + (60,60,60,60) = [64,68,80,80] × 0.75 = [48,51,60,60]
        empty = torch.zeros((0, 4))
        det = torch.tensor([[4.0, 8.0, 20.0, 20.0]])
        reverted = recon.reconstruct_coordinates(_make_full_results([empty, empty, empty, det]), history)

        _assert_coords(reverted, 0, torch.tensor([[48.0, 51.0, 60.0, 60.0]]), atol=1e-3)


# ---------------------------------------------------------------------------
# OBB — oriented bounding boxes
# ---------------------------------------------------------------------------


class TestOBB:
    """Oriented bounding boxes have shape (N, 4, 2) and take a separate code path in both
    the resize and tile revert handlers."""

    def test_obb_resize(self, pipeline):
        """Each OBB's 4 corners are independently reverted via revert_to_origin(box, ops)."""
        prep, recon = pipeline
        # 200×200 → 100×100: revert doubles every coordinate
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": False}}]
        _, history = prep.preprocess(image, steps)

        obb = torch.tensor([[[10.0, 15.0], [20.0, 10.0], [25.0, 20.0], [15.0, 25.0]]])  # (1, 4, 2)
        results = {"boxes": [obb], "scores": [torch.ones(1)], "classes": [np.zeros(1, dtype=np.int32)]}
        reverted = recon.reconstruct_coordinates(results, history)

        expected = torch.tensor([[[20.0, 30.0], [40.0, 20.0], [50.0, 40.0], [30.0, 50.0]]])
        assert torch.allclose(reverted["boxes"][0].float(), expected, atol=1.0)

    def test_obb_tile(self, pipeline):
        """OBB corners are shifted by the tile offset broadcast over (N, 4, 2)."""
        prep, recon = pipeline
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]
        _, history = prep.preprocess(image, steps)

        # Tile (row=1, col=1): offset (x=50, y=50); all corners get +50 on each axis
        empty_obb = torch.zeros((0, 4, 2))
        obb = torch.tensor([[[3.0, 5.0], [15.0, 3.0], [18.0, 12.0], [6.0, 14.0]]])  # (1, 4, 2)
        results = {
            "boxes": [empty_obb, empty_obb, empty_obb, obb],
            "scores": [torch.zeros(0)] * 3 + [torch.ones(1)],
            "classes": [np.zeros(0, dtype=np.int32)] * 3 + [np.zeros(1, dtype=np.int32)],
        }
        reverted = recon.reconstruct_coordinates(results, history)

        merged = reverted["boxes"][0]
        assert merged.shape == (1, 4, 2)
        expected = torch.tensor([[[53.0, 55.0], [65.0, 53.0], [68.0, 62.0], [56.0, 64.0]]])
        assert torch.allclose(merged.float(), expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Points variants
# ---------------------------------------------------------------------------


class TestPoints:
    def test_points_shapes_and_visibility(self, pipeline):
        """(N,K,2) without visibility and (N,K,3) with mixed visibility are both reverted
        correctly; xy coords are scaled and the visibility channel is left unchanged."""
        prep, recon = pipeline
        # 200×200 → 100×100: revert doubles every coordinate
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        steps = [{"type": "resize", "configuration": {"width": 100, "height": 100, "preserve_aspect": False}}]
        _, history = prep.preprocess(image, steps)

        boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
        base = {"boxes": [boxes], "scores": [torch.ones(1)], "classes": [np.zeros(1, dtype=np.int32)], "segments": [[]]}

        # (N=1, K=2, 2) — no visibility channel
        pts_no_vis = torch.tensor([[[15.0, 25.0], [35.0, 55.0]]])
        rev_no_vis = recon.reconstruct_coordinates({**base, "points": [pts_no_vis]}, history)
        assert rev_no_vis["points"][0].shape == (1, 2, 2)
        assert torch.allclose(rev_no_vis["points"][0].float(), torch.tensor([[[30.0, 50.0], [70.0, 110.0]]]), atol=1.0)

        # (N=1, K=3, 3) — three keypoints with mixed visibility; vis column must be unchanged
        pts_vis = torch.tensor([[[15.0, 25.0, 1.0], [30.0, 40.0, 0.0], [45.0, 55.0, 1.0]]])
        rev_vis = recon.reconstruct_coordinates({**base, "points": [pts_vis]}, history)
        assert rev_vis["points"][0].shape == (1, 3, 3)
        assert torch.allclose(
            rev_vis["points"][0].float(), torch.tensor([[[30.0, 50.0, 1.0], [60.0, 80.0, 0.0], [90.0, 110.0, 1.0]]]), atol=1.0
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_steps_returns_results_unchanged(self, pipeline):
        _, recon = pipeline
        boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        results = _make_full_results([boxes])
        reverted = recon.reconstruct_coordinates(results, [])
        assert torch.allclose(reverted["boxes"][0], boxes)

    def test_empty_results_returns_empty(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        steps = [{"type": "resize", "configuration": {"width": 50, "height": 50}}]
        _, history = prep.preprocess(image, steps)
        reverted = recon.reconstruct_coordinates({}, history)
        assert reverted == {}
