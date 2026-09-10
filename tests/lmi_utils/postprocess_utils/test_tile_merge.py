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


def test_covered_fragment_joins_the_whole_detection_and_keeps_its_extent():
    # T1 also saw the object whole (0.9); T0's fragment is contained in it, so it joins that group
    # instead of pairing with T1's left-flagged detection, and the whole detection's box is kept.
    out = _merge(
        [[70, 20, 100, 50], [70, 20, 120, 50], [60, 20, 120, 50]],
        [0.4, 0.9, 0.5],
        [0, 1, 1],
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([70.0, 20.0, 120.0, 50.0]))
    assert out["scores"].item() == pytest.approx(0.9)


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


def test_a_sliver_inside_two_objects_does_not_weld_them():
    # Both T0 detections are whole and are different objects; a T1 sliver at the x=60 seam sits inside
    # each. Linking it to both would join them into one group, and the group keeps only one box.
    out = _merge(
        [[20, 30, 80, 60], [55, 45, 95, 85], [60, 46, 75, 56]],
        [0.9, 0.8, 0.3],
        [0, 0, 1],
    )
    assert out["boxes"].shape == (2, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([20.0, 30.0, 80.0, 60.0]))
    assert torch.allclose(out["boxes"][1], torch.tensor([55.0, 45.0, 95.0, 85.0]))


def test_a_sliver_inside_one_object_seen_from_two_tiles_still_merges():
    # Now the two coverers are one object detected twice, and the T1 piece is that object's part past the
    # x=60 seam, so it keeps both links and everything collapses to a single box.
    out = _merge(
        [[20, 30, 80, 60], [20, 30, 80, 60], [60, 30, 80, 60]],
        [0.9, 0.8, 0.3],
        [0, 0, 1],
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([20.0, 30.0, 80.0, 60.0]))


def test_a_neighbour_box_that_contains_a_fragment_does_not_claim_it():
    # Q and S are whole in T0 and look cut at T1's left edge (x=60). P's cut box in T0 contains both T1 copies,
    # but trimmed to the shared strip x 60-100 it is far larger than either, so it links to neither.
    p, q, s = [30, 10, 100, 70], [61, 20, 75, 40], [61, 45, 75, 65]
    out = _merge([p, q, s, q, s], [0.4, 0.9, 0.8, 0.5, 0.5], [0, 0, 0, 1, 1])
    assert sorted(map(tuple, out["boxes"].tolist())) == sorted(tuple(map(float, b)) for b in (p, q, s))


def test_two_whole_objects_joined_through_a_cut_coverer_both_survive():
    # X spans the seam and is cut in both tiles. Its T0 piece covers Q's T1 copy and its T1 piece covers S's T0
    # copy, and both links agree in the shared strip, so Q and S land in X's group. Both whole objects must remain.
    q, s = [62, 20, 90, 60], [70, 20, 125, 60]
    out = _merge(
        [q, [55, 20, 100, 60], [70, 20, 100, 60], q, [60, 20, 110, 60], s],
        [0.9, 0.5, 0.5, 0.5, 0.5, 0.8],
        [0, 0, 0, 1, 1, 1],
    )
    kept = set(map(tuple, out["boxes"].tolist()))
    assert tuple(map(float, q)) in kept
    assert tuple(map(float, s)) in kept


def test_a_leftover_fragment_inside_a_whole_detection_is_dropped():
    # A is whole in T1. Its T0 piece is cut at x=100 but, from an irregular shape, spans only y 40-60 there,
    # so it does not agree with A in the shared strip and stays unlinked. It sits inside A, so it is dropped.
    a = [65, 20, 120, 80]
    out = _merge([[65, 40, 100, 60], a], [0.4, 0.9], [0, 1])
    assert out["boxes"].tolist() == [list(map(float, a))]


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
    # face each other across the seam and would pair — the whole detections claim them instead.
    # This is the defence that is absent at zero overlap, which is why merging refuses it.
    out = _merge(
        [[20, 20, 80, 50], [80, 20, 100, 50], [60, 20, 80, 50], [80, 20, 140, 50]],
        [0.9, 0.5, 0.5, 0.9],
        [0, 0, 1, 1],
    )
    assert out["boxes"].shape == (2, 4)
    assert sorted(out["boxes"][:, 0].tolist()) == [20.0, 80.0]
    assert sorted(out["boxes"][:, 2].tolist()) == [80.0, 140.0]


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


def test_object_wider_than_a_tile_across_three_overlapping_rows():
    # 3 rows x 2 cols at stride 60 over a 210x150 image. The object (x 20..140, y 80..130) is wider
    # than a tile, so no piece is uncut: rows 0 and 2 hold a top and a bottom half, row 1 holds the
    # full height cut left/right. The halves are not cut at row 1's edges, so seam pairing alone
    # leaves three groups and NMS then keeps the higher-scoring halves over the whole.
    rc, origins = _grid(3, 2)
    out = merge_tile_fragments(
        _result(
            [
                [20, 80, 100, 100],
                [60, 80, 140, 100],
                [20, 80, 100, 130],
                [60, 80, 140, 130],
                [20, 120, 100, 130],
                [60, 120, 140, 130],
            ],
            [0.5, 0.55, 0.4, 0.45, 0.6, 0.65],
        ),
        torch.tensor([0, 1, 2, 3, 4, 5]),
        rc,
        origins,
        (100, 100),
        (210, 150),
        0.8,
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([20.0, 80.0, 140.0, 130.0]))
    assert out["scores"].item() == pytest.approx(0.65)


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
