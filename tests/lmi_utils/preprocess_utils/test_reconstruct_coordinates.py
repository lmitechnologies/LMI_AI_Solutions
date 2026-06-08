import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


@pytest.fixture
def pipeline():
    return Preprocessor(), Reconstructor()


def _make_full_results(boxes_list, tile_hw=None):
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
            xy = torch.stack([cx, cy], dim=-1).unsqueeze(1)
            pts_list.append(torch.cat([xy, torch.ones(len(boxes), 1, 1)], dim=-1))
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
    return [torch.stack([b[[0, 1]], b[[2, 1]], b[[2, 3]], b[[0, 3]]]) for b in boxes]


def _centroids(boxes):
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    xy = torch.stack([cx, cy], dim=-1).unsqueeze(1)
    return torch.cat([xy, torch.ones(len(boxes), 1, 1)], dim=-1)


def _assert_mask_quadrant(canvas, det_idx, y_slice, x_slice, h, w):
    ref = torch.zeros(h, w)
    ref[y_slice, x_slice] = 1.0
    assert torch.all(canvas[det_idx] == ref), f"mask[{det_idx}] placement mismatch"


def _assert_coords(reverted, image_idx, expected_boxes, atol=1.0):
    boxes = reverted["boxes"][image_idx].float()
    assert torch.allclose(boxes, expected_boxes.float(), atol=atol), f"boxes mismatch: {boxes}"
    for seg, exp in zip(reverted["segments"][image_idx], _corners(expected_boxes)):
        assert torch.allclose(seg.float(), exp.float(), atol=atol), "segment mismatch"
    pts = reverted["points"][image_idx].float()
    assert torch.allclose(pts, _centroids(expected_boxes).float(), atol=atol), "points mismatch"


class TestResize:
    def test_uniform_scale(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.resize(width=100, height=100, preserve_aspect=False)])

        boxes = torch.tensor([[10.0, 20.0, 40.0, 45.0]])
        results = _make_full_results([boxes])
        results["masks"] = [torch.ones(1, 100, 100)]
        reverted = recon.reconstruct_coordinates(results, history)
        _assert_coords(reverted, 0, torch.tensor([[20.0, 40.0, 80.0, 90.0]]))
        assert reverted["masks"][0].shape == (1, 200, 200)

    def test_independent_xy_scales(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.resize(width=100, height=100, preserve_aspect=False)])

        boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
        results = _make_full_results([boxes])
        results["masks"] = [torch.ones(1, 100, 100)]
        reverted = recon.reconstruct_coordinates(results, history)
        _assert_coords(reverted, 0, torch.tensor([[20.0, 60.0, 80.0, 180.0]]))
        assert reverted["masks"][0].shape == (1, 300, 200)

    def test_empty_boxes_all_keys_preserved(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.resize(width=100, height=100, preserve_aspect=False)])

        empty = torch.zeros((0, 4))
        results = _make_full_results([empty])
        reverted = recon.reconstruct_coordinates(results, history)

        assert set(reverted.keys()) == set(results.keys())
        assert len(reverted["boxes"][0]) == 0
        assert len(reverted["segments"][0]) == 0
        assert len(reverted["points"][0]) == 0

    def test_multiple_images_resize(self, pipeline):
        prep, recon = pipeline
        images = [np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8) for _ in range(2)]
        _, history = prep.preprocess(images, [steps.resize(width=100, height=100, preserve_aspect=False)])

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
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.resize(width=100, height=100, preserve_aspect=True)])

        boxes = torch.tensor([[35.0, 10.0, 75.0, 90.0]])
        results = _make_full_results([boxes])
        results["masks"] = [torch.ones(1, 100, 100)]
        reverted = recon.reconstruct_coordinates(results, history)

        assert reverted["masks"][0].shape == (1, 200, 100)
        _assert_coords(reverted, 0, torch.tensor([[20.0, 20.0, 100.0, 180.0]]))


class TestBinaryMaskResample:
    """Masks resample via bilinear in _resample_masks, re-thresholded to strict 0/1 (float32)."""

    def test_revert_stays_binary(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.resize(width=64, height=64, preserve_aspect=False)])

        # non-uniform mask: bilinear yields fractional edge values unless re-thresholded
        mask = torch.zeros(1, 64, 64)
        mask[:, 10:40, 12:50] = 1.0
        out = recon.reconstruct_coordinates({"masks": [mask]}, history)["masks"][0]

        assert out.shape == (1, 200, 200)
        assert out.dtype == torch.float32
        assert set(torch.unique(out).tolist()) <= {0.0, 1.0}

    def test_apply_forward_stays_binary(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.resize(width=64, height=64, preserve_aspect=True)])

        mask = torch.zeros(1, 200, 200)
        mask[:, 30:160, 40:175] = 1.0
        out = recon.apply_coordinates({"masks": [mask]}, history)["masks"][0]

        assert set(torch.unique(out).tolist()) <= {0.0, 1.0}

    def test_full_mask_not_eroded(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.resize(width=64, height=64, preserve_aspect=False)])

        out = recon.reconstruct_coordinates({"masks": [torch.ones(1, 64, 64)]}, history)["masks"][0]
        assert torch.all(out == 1.0)  # a fully-on mask stays fully on (no edge erosion)


class TestTile:
    def test_shifts_all_fields_by_tile_offset(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.tile(tile_size=50, stride=50)])

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

        out = reverted["masks"][0]
        assert out.shape == (4, 100, 100)
        _assert_mask_quadrant(out, 0, slice(0, 50), slice(0, 50), 100, 100)
        _assert_mask_quadrant(out, 1, slice(0, 50), slice(50, 100), 100, 100)
        _assert_mask_quadrant(out, 2, slice(50, 100), slice(0, 50), 100, 100)
        _assert_mask_quadrant(out, 3, slice(50, 100), slice(50, 100), 100, 100)

    def test_empty_detections_per_tile(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.tile(tile_size=50, stride=50)])

        empty = torch.zeros((0, 4))
        results = _make_full_results([empty] * 4, tile_hw=(50, 50))
        reverted = recon.reconstruct_coordinates(results, history)

        assert set(reverted.keys()) == set(results.keys())
        assert len(reverted["boxes"][0]) == 0
        assert len(reverted["segments"][0]) == 0
        assert len(reverted["points"][0]) == 0
        assert len(reverted["masks"][0]) == 0

    def test_partial_detections_across_tiles(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.tile(tile_size=50, stride=50)])

        empty = torch.zeros((0, 4))
        det = torch.tensor([[6.0, 9.0, 22.0, 35.0]])
        results = _make_full_results([empty, empty, empty, det], tile_hw=(50, 50))
        reverted = recon.reconstruct_coordinates(results, history)

        _assert_coords(reverted, 0, torch.tensor([[56.0, 59.0, 72.0, 85.0]]), atol=1e-3)

        out = reverted["masks"][0]
        assert out.shape == (1, 100, 100)
        assert torch.all(out[0, 50:100, 50:100] == 1.0)
        assert torch.all(out[0, :50, :] == 0.0) and torch.all(out[0, 50:, :50] == 0.0)

    def test_multiple_images_tile(self, pipeline):
        prep, recon = pipeline
        images = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(2)]
        _, history = prep.preprocess(images, [steps.tile(tile_size=50, stride=50)])

        tile_box = torch.tensor([[4.0, 11.0, 18.0, 28.0]])
        results = _make_full_results([tile_box] * 8, tile_hw=(50, 50))
        reverted = recon.reconstruct_coordinates(results, history)

        assert len(reverted["boxes"]) == 2
        for i in range(2):
            assert len(reverted["boxes"][i]) == 4
            assert len(reverted["segments"][i]) == 4
            assert reverted["masks"][i].shape == (4, 100, 100)

    def test_overlapping_stride_shifts_all_fields(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (90, 90, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.tile(tile_size=60, stride=30)])

        tile_box = torch.tensor([[5.0, 8.0, 20.0, 25.0]])
        results = _make_full_results([tile_box] * 4, tile_hw=(60, 60))
        reverted = recon.reconstruct_coordinates(results, history)

        expected = torch.tensor(
            [
                [5.0, 8.0, 20.0, 25.0],
                [35.0, 8.0, 50.0, 25.0],
                [5.0, 38.0, 20.0, 55.0],
                [35.0, 38.0, 50.0, 55.0],
            ]
        )
        _assert_coords(reverted, 0, expected, atol=1e-3)

        out = reverted["masks"][0]
        assert out.shape == (4, 90, 90)
        assert torch.all(out[3, 30:90, 30:90] == 1.0)
        assert torch.all(out[3, :30, :] == 0.0) and torch.all(out[3, :, :30] == 0.0)

    def test_overlapping_stride_three_column_grid_middle_tile(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (50, 100, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.tile(tile_size=50, stride=25)])

        empty = torch.zeros((0, 4))
        det = torch.tensor([[2.0, 6.0, 18.0, 30.0]])
        reverted = recon.reconstruct_coordinates(_make_full_results([empty, det, empty]), history)

        _assert_coords(reverted, 0, torch.tensor([[27.0, 6.0, 43.0, 30.0]]), atol=1e-3)


class TestPipeline:
    def test_resize_then_tile_reverts_both(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        configs = [
            steps.resize(width=100, height=100, preserve_aspect=False),
            steps.tile(tile_size=50, stride=50),
        ]
        _, history = prep.preprocess(image, configs)

        empty = torch.zeros((0, 4))
        det = torch.tensor([[7.0, 12.0, 25.0, 38.0]])
        reverted = recon.reconstruct_coordinates(_make_full_results([empty, empty, empty, det]), history)

        _assert_coords(reverted, 0, torch.tensor([[114.0, 124.0, 150.0, 176.0]]))

    def test_tile_internal_interpolation_scales_coordinates(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (90, 90, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.tile(tile_size=60, stride=60, scale_mode="interpolation")])

        empty = torch.zeros((0, 4))
        det = torch.tensor([[4.0, 8.0, 20.0, 20.0]])
        reverted = recon.reconstruct_coordinates(_make_full_results([empty, empty, empty, det]), history)

        _assert_coords(reverted, 0, torch.tensor([[48.0, 51.0, 60.0, 60.0]]), atol=1e-3)


class TestOBB:
    def test_obb_resize(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.resize(width=100, height=100, preserve_aspect=False)])

        obb = torch.tensor([[[10.0, 15.0], [20.0, 10.0], [25.0, 20.0], [15.0, 25.0]]])
        results = {"boxes": [obb], "scores": [torch.ones(1)], "classes": [np.zeros(1, dtype=np.int32)]}
        reverted = recon.reconstruct_coordinates(results, history)

        expected = torch.tensor([[[20.0, 30.0], [40.0, 20.0], [50.0, 40.0], [30.0, 50.0]]])
        assert torch.allclose(reverted["boxes"][0].float(), expected, atol=1.0)

    def test_obb_tile(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.tile(tile_size=50, stride=50)])

        empty_obb = torch.zeros((0, 4, 2))
        obb = torch.tensor([[[3.0, 5.0], [15.0, 3.0], [18.0, 12.0], [6.0, 14.0]]])
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


class TestPoints:
    def test_points_shapes_and_visibility(self, pipeline):
        prep, recon = pipeline
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        _, history = prep.preprocess(image, [steps.resize(width=100, height=100, preserve_aspect=False)])

        boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
        base = {"boxes": [boxes], "scores": [torch.ones(1)], "classes": [np.zeros(1, dtype=np.int32)], "segments": [[]]}

        pts_no_vis = torch.tensor([[[15.0, 25.0], [35.0, 55.0]]])
        rev_no_vis = recon.reconstruct_coordinates({**base, "points": [pts_no_vis]}, history)
        assert rev_no_vis["points"][0].shape == (1, 2, 2)
        assert torch.allclose(rev_no_vis["points"][0].float(), torch.tensor([[[30.0, 50.0], [70.0, 110.0]]]), atol=1.0)

        pts_vis = torch.tensor([[[15.0, 25.0, 1.0], [30.0, 40.0, 0.0], [45.0, 55.0, 1.0]]])
        rev_vis = recon.reconstruct_coordinates({**base, "points": [pts_vis]}, history)
        assert rev_vis["points"][0].shape == (1, 3, 3)
        assert torch.allclose(
            rev_vis["points"][0].float(), torch.tensor([[[30.0, 50.0, 1.0], [60.0, 80.0, 0.0], [90.0, 110.0, 1.0]]]), atol=1.0
        )


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
        _, history = prep.preprocess(image, [steps.resize(width=50, height=50)])
        reverted = recon.reconstruct_coordinates({}, history)
        assert reverted == {}
