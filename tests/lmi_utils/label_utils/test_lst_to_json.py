import json

import cv2
import numpy as np
import pytest

from lmi_utils.label_utils.lst_to_json import collect_results, get_annotations_from_json

W, H = 1000, 1000


def box(region_id, label, x1, y1, x2, y2, rotation=0):
    value = {"x": x1 / W * 100, "y": y1 / H * 100, "width": (x2 - x1) / W * 100, "height": (y2 - y1) / H * 100}
    value.update(rotation=rotation, rectanglelabels=[label])
    return {"id": region_id, "type": "rectanglelabels", "value": value, "original_width": W, "original_height": H}


def keypoint(region_id, label, x, y, parent=None):
    result = {"id": region_id, "type": "keypointlabels", "original_width": W, "original_height": H}
    result["value"] = {"x": x / W * 100, "y": y / H * 100, "width": 0.5, "keypointlabels": [label]}
    if parent is not None:
        result["parentID"] = parent
    return result


def link(results):
    """The converted annotations by Label Studio label, and the linking errors."""
    out = []
    _, errors = collect_results(results, out, 0, set(), [], "img.png")
    return {a.label_id: a for a in out}, errors


def test_a_nested_keypoint_links_to_its_parent_even_inside_another_box():
    by_label, errors = link([box("a", "outer", 0, 0, 900, 900), box("b", "inner", 100, 100, 300, 300), keypoint("k", "tip", 200, 200, "b")])
    assert errors == []
    assert by_label["tip"].bounding_box_id == by_label["inner"].id


def test_an_unnested_keypoint_links_to_the_one_box_containing_it():
    by_label, errors = link([box("a", "cap", 100, 100, 500, 300), box("b", "crimp", 100, 400, 500, 600), keypoint("k", "cap_l", 120, 290)])
    assert errors == []
    assert by_label["cap_l"].bounding_box_id == by_label["cap"].id


def test_containment_uses_unrounded_bounds():
    # truncated to whole pixels the crimp box would start at y=359 and contain the keypoint too
    by_label, errors = link(
        [box("a", "cap", 100, 200, 900, 359.2), box("b", "crimp", 100, 359.6, 900, 600), keypoint("k", "cap_l", 120, 359.1)]
    )
    assert errors == []
    assert by_label["cap_l"].bounding_box_id == by_label["cap"].id


@pytest.mark.parametrize(
    "results",
    [
        [box("a", "cap", 0, 0, 500, 500), box("b", "crimp", 0, 400, 500, 900), keypoint("k", "cap_l", 100, 450)],
        [box("a", "cap", 0, 0, 500, 500), keypoint("k", "cap_l", 700, 700)],
        [box("a", "cap", 0, 0, 500, 500), keypoint("p", "other", 10, 10), keypoint("k", "cap_l", 100, 100, "p")],
        [box("a", "cap", 0, 0, 500, 500), keypoint("k", "cap_l", 100, 100, "missing")],
        [box("a", "cap", 0, 0, 500, 500, rotation=10), keypoint("k", "cap_l", 100, 100, "a")],
        [box("a", "cap", 0, 0, 500, 500), box("b", "string", 600, 600, 900, 900, rotation=10), keypoint("k", "cap_l", 100, 100)],
    ],
    ids=["in two boxes", "in no box", "parent is a keypoint", "parent not in the list", "rotated parent", "beside a rotated box"],
)
def test_a_keypoint_that_cannot_be_linked_is_reported_and_left_unlinked(results):
    by_label, errors = link(results)
    assert len(errors) == 1
    assert by_label["cap_l"].bounding_box_id is None


def test_keypoints_repeating_a_name_on_one_box_are_reported_and_unlinked():
    out = []
    results = [box("a", "cap", 0, 0, 500, 500), keypoint("k1", "cap_l", 10, 10, "a"), keypoint("k2", "cap_l", 20, 20, "a")]
    _, errors = collect_results(results, out, 0, set(), [], "img.png")
    assert len(errors) == 1
    assert [a.bounding_box_id for a in out if a.label_id == "cap_l"] == [None, None]


def test_a_keypoint_nested_under_a_shared_region_id_is_reported():
    by_label, errors = link([box("a", "cap", 0, 0, 500, 500), box("a", "crimp", 0, 600, 500, 900), keypoint("k", "cap_l", 100, 100, "a")])
    assert len(errors) == 1
    assert by_label["cap_l"].bounding_box_id is None


def write_project(tmp_path, task):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    cv2.imwrite(str(imgs / "img.png"), np.zeros((H, W, 3), dtype=np.uint8))
    path = tmp_path / "lst.json"
    path.write_text(json.dumps([{"data": {"image": "gs://bucket/img.png"}, **task}]))
    return str(path), str(imgs)


UNLINKABLE = [box("a", "cap", 0, 0, 500, 500), keypoint("k", "cap_l", 700, 700), keypoint("j", "cap_r", 800, 800)]


def test_conversion_fails_listing_every_unlinkable_keypoint(tmp_path, caplog):
    path, imgs = write_project(tmp_path, {"annotations": [{"result": UNLINKABLE}]})
    with pytest.raises(ValueError, match="2 keypoint"):
        get_annotations_from_json(path, imgs)
    assert "cap_l" in caplog.text and "cap_r" in caplog.text


def test_keep_unlinked_keypoints_writes_them_unlinked(tmp_path):
    path, imgs = write_project(tmp_path, {"annotations": [{"result": UNLINKABLE}]})
    [file], _ = get_annotations_from_json(path, imgs, keep_unlinked_keypoints=True)
    assert [a.bounding_box_id for a in file.annotations if a.label_id in ("cap_l", "cap_r")] == [None, None]


def test_unlinkable_prediction_keypoints_do_not_fail_the_conversion(tmp_path):
    path, imgs = write_project(tmp_path, {"predictions": [{"result": UNLINKABLE}]})
    [file], _ = get_annotations_from_json(path, imgs)
    assert len(file.predictions) == 3


def test_a_parent_is_looked_up_in_its_own_result_list(tmp_path):
    # the prediction's box shares the region id, but an annotation keypoint can only nest under an annotation region
    annotations = [box("a", "cap", 0, 0, 500, 500), keypoint("k", "cap_l", 100, 100, "p")]
    path, imgs = write_project(
        tmp_path, {"annotations": [{"result": annotations}], "predictions": [{"result": [box("p", "cap", 0, 0, 500, 500)]}]}
    )
    with pytest.raises(ValueError, match="1 keypoint"):
        get_annotations_from_json(path, imgs)
