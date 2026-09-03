"""Fragment merging: truncation flags, explanation clearing, pairing, union."""

import numpy as np
import pytest
import torch

from lmi_utils.postprocess_utils.tile_merge import merge_tile_fragments

# Two tiles side by side: T0 spans x [0, 100), T1 spans x [60, 160). Image is 150x100, so T1's
# right edge falls in the padding and never counts as an interior seam.
_TILE_RC = np.array([[0, 0], [0, 1]])
_ORIGINS = np.array([[0, 0], [0, 60]])
_TILE_SIZE = (100, 100)
_IM_SIZE = (100, 150)


def _result(boxes, scores, classes=None):
    n = len(boxes)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "classes": np.array(classes if classes is not None else [0] * n, dtype=np.int32),
    }


def _merge(boxes, scores, tile_idx, classes=None, containment=0.8, **kwargs):
    return merge_tile_fragments(
        _result(boxes, scores, classes),
        torch.tensor(tile_idx, dtype=torch.long),
        _TILE_RC,
        _ORIGINS,
        _TILE_SIZE,
        _IM_SIZE,
        containment,
        **kwargs,
    )


def test_seam_fragments_union_into_one_box():
    # Cut at x=100 (T0's right edge) / x=60 (T1's left edge): both halves are flagged and pair.
    out = _merge([[40, 20, 100, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1])
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 20.0, 120.0, 50.0]))
    assert out["scores"].item() == pytest.approx(0.9)  # group max, not mean


def test_untruncated_prediction_explains_a_fragment_and_blocks_the_merge():
    # T1 also saw the object whole (0.9); T0's fragment is contained in it, so its flag clears and
    # it must not pair with T1's left-flagged detection behind that correct answer.
    out = _merge(
        [[70, 20, 100, 50], [70, 20, 120, 50], [60, 20, 120, 50]],
        [0.4, 0.9, 0.5],
        [0, 1, 1],
    )
    assert out["boxes"].shape == (3, 4)


def test_without_the_explanation_the_same_fragments_do_merge():
    out = _merge([[70, 20, 100, 50], [60, 20, 120, 50]], [0.4, 0.5], [0, 1])
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([60.0, 20.0, 120.0, 50.0]))


def test_objects_stacked_along_the_seam_do_not_chain():
    # Perpendicular extents don't line up, so the top-left half must not pair with the bottom-right.
    out = _merge(
        [[40, 10, 100, 40], [60, 10, 120, 40], [40, 60, 100, 90], [60, 60, 120, 90]],
        [0.5, 0.5, 0.5, 0.5],
        [0, 1, 0, 1],
    )
    assert out["boxes"].shape == (2, 4)
    ys = sorted(out["boxes"][:, 1].tolist())
    assert ys == [pytest.approx(10.0), pytest.approx(60.0)]


def test_fragments_of_different_classes_do_not_pair():
    out = _merge([[40, 20, 100, 50], [60, 20, 120, 50]], [0.5, 0.5], [0, 1], classes=[0, 1])
    assert out["boxes"].shape == (2, 4)


def test_image_border_is_not_a_seam():
    # Both boxes sit on T0's left edge, which is the image border — nothing continues past it.
    out = _merge([[0, 20, 30, 50], [0, 60, 30, 90]], [0.5, 0.5], [0, 0])
    assert out["boxes"].shape == (2, 4)


def test_masks_union_and_segments_union():
    masks = torch.zeros((2, 100, 150), dtype=torch.uint8)
    masks[0, 20:50, 40:100] = 1
    masks[1, 20:50, 60:120] = 1
    merged = {
        "boxes": torch.tensor([[40.0, 20, 100, 50], [60.0, 20, 120, 50]]),
        "masks": masks,
        "segments": [
            torch.tensor([[40.0, 20], [100.0, 20], [100.0, 50], [40.0, 50]]),
            torch.tensor([[60.0, 20], [120.0, 20], [120.0, 50], [60.0, 50]]),
        ],
        "scores": torch.tensor([0.4, 0.9]),
        "classes": np.array([0, 0], dtype=np.int32),
    }
    out = merge_tile_fragments(merged, torch.tensor([0, 1]), _TILE_RC, _ORIGINS, _TILE_SIZE, _IM_SIZE, 0.8)
    assert out["masks"].shape == (1, 100, 150)
    assert int(out["masks"].sum()) == 30 * 80  # x 40..120, y 20..50
    ring = out["segments"][0]
    assert ring[:, 0].min() == pytest.approx(40.0) and ring[:, 0].max() == pytest.approx(120.0)


def test_fragment_that_stops_short_of_the_seam_is_not_flagged():
    # Known gap: edge contact is a proxy. T0's half ends 8px before the tile edge, so it never
    # looks truncated and its true partner is lost.
    out = _merge([[40, 20, 92, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1])
    assert out["boxes"].shape == (2, 4)


def test_fragment_within_the_edge_tolerance_still_pairs():
    out = _merge([[40, 20, 99, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1])
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 20.0, 120.0, 50.0]))


def test_two_objects_abutting_at_the_seam_do_not_merge():
    # A ends where B begins. Each tile sees one whole and the other cut, so the two cut halves
    # face each other across the seam and would pair — the whole detections explain them instead.
    # This is the defence that is absent at zero overlap, which is why merging refuses it.
    out = _merge(
        [[20, 20, 80, 50], [80, 20, 100, 50], [60, 20, 80, 50], [80, 20, 140, 50]],
        [0.9, 0.5, 0.5, 0.9],
        [0, 0, 1, 1],
    )
    assert out["boxes"].shape == (4, 4)


# A 2x2 block of 100px tiles at stride 60 over a 150x150 image.
_BLOCK_RC = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
_BLOCK_ORIGINS = np.array([[0, 0], [0, 60], [60, 0], [60, 60]])


def test_object_spanning_a_2x2_block_closes_diagonally():
    out = merge_tile_fragments(
        _result(
            [[40, 40, 100, 100], [60, 40, 120, 100], [40, 60, 100, 120], [60, 60, 120, 120]],
            [0.3, 0.4, 0.5, 0.6],
        ),
        torch.tensor([0, 1, 2, 3]),
        _BLOCK_RC,
        _BLOCK_ORIGINS,
        (100, 100),
        (150, 150),
        0.8,
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 40.0, 120.0, 120.0]))
    assert out["scores"].item() == pytest.approx(0.6)


def _grid(rows: int, cols: int, stride: int = 60):
    rc = np.array([[r, c] for r in range(rows) for c in range(cols)])
    return rc, rc * stride


def test_object_spanning_three_tiles_vertically():
    # One column of 3 tiles at y = 0, 60, 120 over a 100x210 image; the middle tile sees only the
    # object's interior, with neither its top nor its bottom edge visible.
    rc, origins = _grid(3, 1)
    out = merge_tile_fragments(
        _result([[30, 40, 70, 100], [30, 60, 70, 160], [30, 120, 70, 200]], [0.3, 0.5, 0.4]),
        torch.tensor([0, 1, 2]),
        rc,
        origins,
        (100, 100),
        (210, 100),
        0.8,
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([30.0, 40.0, 70.0, 200.0]))
    assert out["scores"].item() == pytest.approx(0.5)


def test_object_spanning_a_3x2_block_of_six_tiles():
    # 2 rows x 3 cols at stride 60 over a 150x210 image. Every fragment pairs with its row and
    # column neighbours; union-find has to fold all six into one group.
    rc, origins = _grid(2, 3)
    out = merge_tile_fragments(
        _result(
            [
                [40, 40, 100, 100],
                [60, 40, 160, 100],
                [120, 40, 200, 100],
                [40, 60, 100, 140],
                [60, 60, 160, 140],
                [120, 60, 200, 140],
            ],
            [0.3, 0.35, 0.4, 0.45, 0.7, 0.5],
        ),
        torch.tensor([0, 1, 2, 3, 4, 5]),
        rc,
        origins,
        (100, 100),
        (150, 210),
        0.8,
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 40.0, 200.0, 140.0]))
    assert out["scores"].item() == pytest.approx(0.7)


def test_masks_union_across_a_three_tile_chain():
    rc, origins = _grid(1, 3)
    masks = torch.zeros((3, 100, 210), dtype=torch.uint8)
    masks[0, 30:60, 40:100] = 1
    masks[1, 30:60, 60:160] = 1
    masks[2, 30:60, 120:200] = 1
    merged = {
        "boxes": torch.tensor([[40.0, 30, 100, 60], [60.0, 30, 160, 60], [120.0, 30, 200, 60]]),
        "masks": masks,
        "scores": torch.tensor([0.3, 0.4, 0.5]),
        "classes": np.zeros(3, dtype=np.int32),
    }
    out = merge_tile_fragments(merged, torch.tensor([0, 1, 2]), rc, origins, (100, 100), (100, 210), 0.8)
    assert out["masks"].shape == (1, 100, 210)
    assert int(out["masks"].sum()) == 30 * 160  # x 40..200, y 30..60


def test_fragment_below_the_score_floor_is_discarded_before_pairing():
    out = _merge([[40, 20, 100, 50], [60, 20, 120, 50]], [0.001, 0.9], [0, 1])
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([60.0, 20.0, 120.0, 50.0]))


def test_single_instance_and_empty_input_pass_through():
    empty = {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "classes": np.zeros(0, dtype=np.int32)}
    out = merge_tile_fragments(empty, torch.zeros(0, dtype=torch.long), _TILE_RC, _ORIGINS, _TILE_SIZE, _IM_SIZE, 0.8)
    assert out["boxes"].shape == (0, 4)
    one = _merge([[40, 20, 100, 50]], [0.9], [0])
    assert one["boxes"].shape == (1, 4)


def test_results_without_scores_still_merge():
    merged = {"boxes": torch.tensor([[40.0, 20, 100, 50], [60.0, 20, 120, 50]]), "classes": np.zeros(2, dtype=np.int32)}
    out = merge_tile_fragments(merged, torch.tensor([0, 1]), _TILE_RC, _ORIGINS, _TILE_SIZE, _IM_SIZE, 0.8)
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 20.0, 120.0, 50.0]))


def test_a_wider_edge_tolerance_recovers_the_fragment_it_missed():
    # The 8px-short half from the test above, with a tolerance wide enough to reach it.
    out = _merge([[40, 20, 92, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1], edge_tolerance=10)
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 20.0, 120.0, 50.0]))


def test_a_zero_edge_tolerance_demands_an_exact_edge_hit():
    assert _merge([[40, 20, 99, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1], edge_tolerance=0)["boxes"].shape == (2, 4)
    assert _merge([[40, 20, 100, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1], edge_tolerance=0)["boxes"].shape == (1, 4)
