"""Fragment merging: cut flags, joins between tiles, union, dropping fragments another tile saw in full."""

import numpy as np
import pytest
import torch

from lmi_utils.postprocess_utils import tile_merge
from lmi_utils.postprocess_utils.tile_merge import merge_tile_fragments

# Two tiles side by side: T0 spans x [0, 100), T1 spans x [60, 160). Image is 150x100, so T1's
# right edge falls in the padding and never counts as an interior seam.
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
    # each. Joining it to both would put them in one group, and the group keeps only one box.
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
    # x=60 seam, so it keeps both joins and everything collapses to a single box.
    out = _merge(
        [[20, 30, 80, 60], [20, 30, 80, 60], [60, 30, 80, 60]],
        [0.9, 0.8, 0.3],
        [0, 0, 1],
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([20.0, 30.0, 80.0, 60.0]))


def test_a_neighbour_box_that_contains_a_fragment_does_not_claim_it():
    # Q and S are whole in T0 and look cut at T1's left edge (x=60). P's cut box in T0 contains both T1 copies,
    # but trimmed to the shared strip x 60-100 it is far larger than either, so it joins to neither.
    p, q, s = [30, 10, 100, 70], [61, 20, 75, 40], [61, 45, 75, 65]
    out = _merge([p, q, s, q, s], [0.4, 0.9, 0.8, 0.5, 0.5], [0, 0, 0, 1, 1])
    assert sorted(map(tuple, out["boxes"].tolist())) == sorted(tuple(map(float, b)) for b in (p, q, s))


def test_two_whole_objects_joined_through_a_cut_coverer_both_survive():
    # X spans the seam and is cut in both tiles. Its T0 piece covers Q's T1 copy and its T1 piece covers S's T0
    # copy, and both joins agree in the shared strip, so Q and S land in X's group. Both whole objects must remain.
    q, s = [62, 20, 90, 60], [70, 20, 125, 60]
    out = _merge(
        [q, [55, 20, 100, 60], [70, 20, 100, 60], q, [60, 20, 110, 60], s],
        [0.9, 0.5, 0.5, 0.5, 0.5, 0.8],
        [0, 0, 0, 1, 1, 1],
    )
    kept = set(map(tuple, out["boxes"].tolist()))
    assert tuple(map(float, q)) in kept
    assert tuple(map(float, s)) in kept


def test_a_leftover_fragment_the_other_tile_saw_in_full_is_dropped():
    # A is whole in T1. Its T0 piece is cut at x=100 but, from an irregular shape, spans only y 40-60 there,
    # so it does not agree with A in the shared strip and stays unjoined. T1 saw all of it, so it is dropped.
    a = [65, 20, 120, 80]
    out = _merge([[65, 40, 100, 60], a], [0.4, 0.9], [0, 1])
    assert out["boxes"].tolist() == [list(map(float, a))]


def test_a_noisy_sliver_sticking_out_of_its_whole_detection_is_dropped():
    # The T0 sliver at x=100 reaches past A in y, so A holds under 0.8 of it and the boxes disagree. T1 saw it in full.
    a = [70, 25, 120, 45]
    out = _merge([[90, 20, 100, 55], a], [0.6, 0.9], [0, 1])
    assert out["boxes"].tolist() == [list(map(float, a))]


def test_an_unmatched_sliver_the_other_tile_saw_in_full_is_dropped():
    out = _merge([[90, 20, 100, 50], [10, 20, 40, 50]], [0.6, 0.9], [0, 0])
    assert out["boxes"].tolist() == [[10.0, 20.0, 40.0, 50.0]]


def test_a_fragment_no_tile_saw_in_full_is_kept():
    # Reaches past T1's left edge (x=60), so T1 did not see all of it.
    out = _merge([[30, 20, 100, 50], [10, 60, 40, 90]], [0.6, 0.9], [0, 0])
    assert out["boxes"].shape == (2, 4)


def test_a_fragment_cut_by_both_tiles_is_kept():
    # Spans the whole overlap: cut at T0's right edge and touching T1's left edge, so neither tile saw it whole.
    out = _merge([[61, 20, 100, 50], [10, 60, 40, 90]], [0.6, 0.9], [0, 0])
    assert out["boxes"].shape == (2, 4)


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
    out = merge_tile_fragments(merged, torch.tensor([0, 1]), _ORIGINS, _TILE_SIZE, _IM_SIZE, 0.8)
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
_BLOCK_ORIGINS = np.array([[0, 0], [0, 60], [60, 0], [60, 60]])


def test_object_spanning_a_2x2_block_closes_diagonally():
    out = merge_tile_fragments(
        _result(
            [[40, 40, 100, 100], [60, 40, 120, 100], [40, 60, 100, 120], [60, 60, 120, 120]],
            [0.3, 0.4, 0.5, 0.6],
        ),
        torch.tensor([0, 1, 2, 3]),
        _BLOCK_ORIGINS,
        (100, 100),
        (150, 150),
        0.8,
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 40.0, 120.0, 120.0]))
    assert out["scores"].item() == pytest.approx(0.6)


def _grid(rows: int, cols: int, stride: int = 60):
    return np.array([[r, c] for r in range(rows) for c in range(cols)]) * stride


def test_object_spanning_three_tiles_vertically():
    # One column of 3 tiles at y = 0, 60, 120 over a 100x210 image; the middle tile sees only the
    # object's interior, with neither its top nor its bottom edge visible.
    origins = _grid(3, 1)
    out = merge_tile_fragments(
        _result([[30, 40, 70, 100], [30, 60, 70, 160], [30, 120, 70, 200]], [0.3, 0.5, 0.4]),
        torch.tensor([0, 1, 2]),
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
    # full height cut left/right. The halves are not cut at row 1's edges, so only their joins to
    # row 1's pieces join the three rows into one object.
    origins = _grid(3, 2)
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
        origins,
        (100, 100),
        (210, 150),
        0.8,
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([20.0, 80.0, 140.0, 130.0]))
    assert out["scores"].item() == pytest.approx(0.65)


def test_object_spanning_a_3x2_block_of_six_tiles():
    # 2 rows x 3 cols at stride 60 over a 150x210 image. Every fragment joins to its row and
    # column neighbours; union-find has to fold all six into one group.
    origins = _grid(2, 3)
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
        origins,
        (100, 100),
        (150, 210),
        0.8,
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 40.0, 200.0, 140.0]))
    assert out["scores"].item() == pytest.approx(0.7)


def test_masks_union_across_a_three_tile_chain():
    # the best-scoring member is the last row, so the output row is not the first member's
    origins = _grid(1, 3)
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
    out = merge_tile_fragments(merged, torch.tensor([0, 1, 2]), origins, (100, 100), (100, 210), 0.8)
    assert out["masks"].shape == (1, 100, 210)
    assert int(out["masks"].sum()) == 30 * 160  # x 40..200, y 30..60


def test_a_low_score_fragment_still_pairs():
    # removing weak views is the caller's score threshold, not merging's job
    out = _merge([[40, 20, 100, 50], [60, 20, 120, 50]], [0.001, 0.9], [0, 1])
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 20.0, 120.0, 50.0]))


def test_a_low_score_whole_detection_is_kept():
    out = _merge([[10, 20, 40, 50], [10, 60, 40, 90]], [0.001, 0.9], [0, 0])
    assert out["boxes"].shape == (2, 4)


def test_single_instance_and_empty_input_pass_through():
    empty = {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "classes": np.zeros(0, dtype=np.int32)}
    out = merge_tile_fragments(empty, torch.zeros(0, dtype=torch.long), _ORIGINS, _TILE_SIZE, _IM_SIZE, 0.8)
    assert out["boxes"].shape == (0, 4)
    one = _merge([[10, 20, 40, 50]], [0.9], [0])
    assert one["boxes"].shape == (1, 4)


def test_a_lone_sliver_the_other_tile_saw_in_full_is_dropped():
    # same sliver as the two-detection case above, with nothing else in the image
    assert _merge([[90, 20, 100, 50]], [0.6], [0])["boxes"].shape == (0, 4)


def test_merge_origin_marks_a_lone_fragment():
    out = _merge([[30, 20, 100, 50]], [0.6], [0], report_origin=True)
    assert out["merge_origin"].tolist() == [tile_merge.ORIGIN_FRAGMENT]


def test_results_without_scores_still_merge():
    merged = {"boxes": torch.tensor([[40.0, 20, 100, 50], [60.0, 20, 120, 50]]), "classes": np.zeros(2, dtype=np.int32)}
    out = merge_tile_fragments(merged, torch.tensor([0, 1]), _ORIGINS, _TILE_SIZE, _IM_SIZE, 0.8)
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 20.0, 120.0, 50.0]))


def test_a_wider_edge_tolerance_recovers_the_fragment_it_missed():
    # The 8px-short half from the test above, with a tolerance wide enough to reach it.
    out = _merge([[40, 20, 92, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1], edge_tolerance=10)
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 20.0, 120.0, 50.0]))


def test_a_zero_edge_tolerance_still_joins_within_the_join_margin():
    assert _merge([[40, 20, 99, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1], edge_tolerance=0)["boxes"].shape == (1, 4)


def test_a_zero_edge_tolerance_drops_only_a_sliver_that_hits_the_edge_exactly():
    whole = [10, 60, 40, 90]
    assert _merge([[90, 20, 99, 50], whole], [0.6, 0.9], [0, 0], edge_tolerance=0)["boxes"].shape == (2, 4)
    assert _merge([[90, 20, 100, 50], whole], [0.6, 0.9], [0, 0], edge_tolerance=0)["boxes"].shape == (1, 4)


def test_a_whole_object_near_the_seam_is_not_dropped_as_a_fragment():
    # 4px short of T0's right edge: inside the join margin, outside the edge tolerance. T1 saw that area but found nothing.
    out = _merge([[70, 20, 96, 50], [10, 60, 40, 90]], [0.6, 0.9], [0, 0])
    assert out["boxes"].shape == (2, 4)


def test_merge_origin_is_absent_unless_asked():
    out = _merge([[40, 20, 100, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1])
    assert "merge_origin" not in out


def test_merge_origin_marks_a_union_of_fragments():
    out = _merge([[40, 20, 100, 50], [60, 20, 120, 50]], [0.4, 0.9], [0, 1], report_origin=True)
    assert out["merge_origin"].tolist() == [tile_merge.ORIGIN_UNION]


def test_merge_origin_marks_a_whole_detection_that_absorbed_fragments():
    out = _merge([[70, 20, 100, 50], [70, 20, 120, 50], [60, 20, 120, 50]], [0.4, 0.9, 0.5], [0, 1, 1], report_origin=True)
    assert out["merge_origin"].tolist() == [tile_merge.ORIGIN_WHOLE_GROUPED]


def test_merge_origin_separates_a_kept_fragment_from_a_whole_detection():
    # first box reaches past T1's left edge so no tile saw it whole; the second is whole in T0
    out = _merge([[30, 20, 100, 50], [10, 60, 40, 90]], [0.6, 0.9], [0, 0], report_origin=True)
    assert out["merge_origin"].tolist() == [tile_merge.ORIGIN_FRAGMENT, tile_merge.ORIGIN_WHOLE]


def _split(boxes, groups, whole, joins):
    """_split_distinct_whole with the pair scan merge_tile_fragments would have handed it."""
    return tile_merge._split_distinct_whole({"boxes": boxes}, boxes, groups, whole, joins, tile_merge.intersecting_pairs(boxes))


def test_split_distinct_whole_treats_a_chain_of_overlaps_as_one_object():
    # A holds 0.96 of B and B holds 0.96 of C, but A and C only reach 0.92: a chain is still one object
    boxes = torch.tensor([[0.0, 0, 100, 10], [4.0, 0, 104, 10], [8.0, 0, 108, 10]])
    whole = torch.ones(3, dtype=torch.bool)
    assert _split(boxes, [[0, 1, 2]], whole, tile_merge._NO_PAIRS) == [[0, 1, 2]]


def test_split_distinct_whole_still_separates_two_distinct_objects():
    boxes = torch.tensor([[0.0, 0, 10, 10], [100.0, 0, 110, 10]])
    whole = torch.ones(2, dtype=torch.bool)
    assert _split(boxes, [[0, 1]], whole, tile_merge._NO_PAIRS) == [[0], [1]]


def _joined(pairs):
    """The (L, 2) join list that _split_distinct_whole takes."""
    return torch.tensor(pairs, dtype=torch.long)


def test_split_distinct_whole_gives_each_fragment_to_the_object_it_joined():
    # fragment 3 reaches object 1 only through fragment 2
    boxes = torch.tensor([[0.0, 0, 10, 10], [100.0, 0, 110, 10], [95.0, 0, 105, 10], [90.0, 0, 100, 10]])
    whole = torch.tensor([True, True, False, False])
    joined = _joined([(0, 1), (1, 2), (2, 3)])
    assert _split(boxes, [[0, 1, 2, 3]], whole, joined) == [[0], [1, 2, 3]]


def test_split_distinct_whole_gives_a_fragment_joined_to_both_objects_to_the_first():
    boxes = torch.tensor([[0.0, 0, 10, 10], [100.0, 0, 110, 10], [5.0, 0, 105, 10]])
    whole = torch.tensor([True, True, False])
    joined = _joined([(0, 2), (1, 2)])
    assert _split(boxes, [[0, 1, 2]], whole, joined) == [[0, 2], [1]]


def test_merge_origin_marks_the_second_object_that_absorbed_a_fragment():
    # W0 and W1 are different whole objects in T1. C joins both; C2 joins only W1, so W1 absorbed a fragment too.
    w0, w1, c, c2 = [70, 20, 115, 50], [70, 20, 105, 70], [70, 20, 100, 50], [70, 25, 100, 70]
    out = _merge([w0, w1, c, c2], [0.9, 0.8, 0.7, 0.7], [1, 1, 0, 0], report_origin=True)
    assert out["boxes"].tolist() == [list(map(float, w0)), list(map(float, w1))]
    assert out["merge_origin"].tolist() == [tile_merge.ORIGIN_WHOLE_GROUPED, tile_merge.ORIGIN_WHOLE_GROUPED]


# Two tiles side by side with no overlap at all: T0 spans x [0, 100), T1 spans x [100, 200).
# Their shared area is the line x=100, so the agreement test runs in a band straddling it.
_NO_OVERLAP_ORIGINS = np.array([[0, 0], [0, 100]])
_NO_OVERLAP_IM_SIZE = (100, 200)


def _no_overlap(boxes, scores, tile_idx, classes=None, containment=0.8, **kwargs):
    return merge_tile_fragments(
        _result(boxes, scores, classes),
        torch.tensor(tile_idx, dtype=torch.long),
        _NO_OVERLAP_ORIGINS,
        _TILE_SIZE,
        _NO_OVERLAP_IM_SIZE,
        containment,
        **kwargs,
    )


def test_no_overlap_tiles_union_a_cut_object():
    out = _no_overlap([[40, 20, 100, 50], [100, 21, 160, 50]], [0.4, 0.9], [0, 1])
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([40.0, 20.0, 160.0, 50.0]))


def test_no_overlap_tiles_keep_pieces_that_do_not_line_up_along_the_seam():
    out = _no_overlap([[40, 20, 100, 50], [100, 60, 160, 90]], [0.4, 0.9], [0, 1])
    assert out["boxes"].shape == (2, 4)


def test_no_overlap_tiles_leave_a_piece_that_stops_short_of_the_seam():
    # 10px short is well outside the band, so nothing pairs with it.
    out = _no_overlap([[40, 20, 90, 50], [100, 20, 160, 50]], [0.4, 0.9], [0, 1])
    assert out["boxes"].shape == (2, 4)


def test_no_overlap_tiles_pair_within_the_edge_tolerance():
    out = _no_overlap([[40, 20, 99, 50], [100, 20, 160, 50]], [0.4, 0.9], [0, 1])
    assert out["boxes"].shape == (1, 4)


def test_no_overlap_tiles_do_not_pair_different_classes():
    out = _no_overlap([[40, 20, 100, 50], [100, 20, 160, 50]], [0.4, 0.9], [0, 1], classes=[0, 1])
    assert out["boxes"].shape == (2, 4)


def test_no_overlap_tiles_mark_the_union_and_keep_a_lone_fragment():
    out = _no_overlap([[40, 20, 100, 50], [100, 20, 160, 50], [100, 70, 140, 90]], [0.4, 0.9, 0.5], [0, 1, 1], report_origin=True)
    assert out["merge_origin"].tolist() == [tile_merge.ORIGIN_UNION, tile_merge.ORIGIN_FRAGMENT]


def test_no_overlap_tiles_close_a_2x2_block():
    # One object through the corner of four 100px tiles with no overlap, over a 200x200 image.
    out = merge_tile_fragments(
        _result([[60, 60, 100, 100], [100, 60, 140, 100], [60, 100, 100, 140], [100, 100, 140, 140]], [0.5] * 4),
        torch.tensor([0, 1, 2, 3]),
        np.array([[0, 0], [0, 100], [100, 0], [100, 100]]),
        _TILE_SIZE,
        (200, 200),
        0.8,
    )
    assert out["boxes"].shape == (1, 4)
    assert torch.allclose(out["boxes"][0], torch.tensor([60.0, 60.0, 140.0, 140.0]))


def _slanted_masks(gap):
    """A bar crossing the seam at an angle, as two mask halves. Their boxes barely overlap in y; their
    crossings at x=100 line up exactly. ``gap`` shifts the right half away from the shared rows."""
    left, right = torch.zeros(100, 200, dtype=torch.bool), torch.zeros(100, 200, dtype=torch.bool)
    for x in range(40, 100):
        y = 20 + (x - 40) // 2
        left[y : y + 8, x] = True
    for x in range(100, 160):
        y = 50 + gap + (x - 100) // 2
        right[y : y + 8, x] = True
    return torch.stack([left, right])


def test_no_overlap_tiles_join_a_slanted_object_by_its_mask():
    # Box y extents are 20..57 and 50..87 - a 1D IoU of 0.1, far under AGREEMENT_IOU. Only the masks
    # meeting at the seam show they are one object.
    res = {"masks": _slanted_masks(0), "scores": torch.tensor([0.4, 0.9]), "classes": np.array([0, 0], np.int32)}
    out = merge_tile_fragments(res, torch.tensor([0, 1]), _NO_OVERLAP_ORIGINS, _TILE_SIZE, _NO_OVERLAP_IM_SIZE, 0.8)
    assert len(out["masks"]) == 1


def test_no_overlap_tiles_keep_slanted_pieces_that_miss_each_other_at_the_seam():
    res = {"masks": _slanted_masks(20), "scores": torch.tensor([0.4, 0.9]), "classes": np.array([0, 0], np.int32)}
    out = merge_tile_fragments(res, torch.tensor([0, 1]), _NO_OVERLAP_ORIGINS, _TILE_SIZE, _NO_OVERLAP_IM_SIZE, 0.8)
    assert len(out["masks"]) == 2


def test_a_mask_that_stops_short_of_the_seam_still_does_not_pair():
    masks = _slanted_masks(0)
    masks[0, :, 90:] = False  # the left half now ends 10px before the seam
    res = {"masks": masks, "scores": torch.tensor([0.4, 0.9]), "classes": np.array([0, 0], np.int32)}
    out = merge_tile_fragments(res, torch.tensor([0, 1]), _NO_OVERLAP_ORIGINS, _TILE_SIZE, _NO_OVERLAP_IM_SIZE, 0.8)
    assert len(out["masks"]) == 2


def test_no_overlap_tiles_compare_a_mask_that_ends_at_the_band_edge():
    # The left mask is whole (it stops 3px short of the seam) but still reaches the right piece's
    # grown box, so the band slice of it has no width at all.
    masks = torch.zeros(2, 100, 200, dtype=torch.bool)
    masks[0, 20:50, 40:97] = True
    masks[1, 20:50, 100:160] = True
    res = {"masks": masks, "scores": torch.tensor([0.4, 0.9]), "classes": np.array([0, 0], np.int32)}
    out = merge_tile_fragments(res, torch.tensor([0, 1]), _NO_OVERLAP_ORIGINS, _TILE_SIZE, _NO_OVERLAP_IM_SIZE, 0.8)
    assert len(out["masks"]) == 2
