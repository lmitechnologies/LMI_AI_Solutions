import json

import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.representations import (
    AnnotationType,
    Box,
    BoxAnnotation,
    Dataset,
    FileAnnotations,
    KeypointAnnotation,
    Label,
    Point2d,
    Polygon,
    PolygonAnnotation,
)
from lmi_utils.label_utils.json_to_factory import convert_json_to_factory


def convert(tmp_path, paths):
    images = tmp_path / "images"
    for path in paths:
        (images / path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(images / path), np.zeros((4, 4, 3), dtype=np.uint8))
    files = [FileAnnotations(id=str(i), path=p, height=4, width=4) for i, p in enumerate(paths)]
    convert_json_to_factory(Dataset(labels=[], files=files), images, tmp_path / "out")
    return sorted(str(p.relative_to(tmp_path / "out")) for p in (tmp_path / "out").rglob("*.label.json"))


@pytest.mark.parametrize(
    "paths, expected",
    [
        (["a/image.png", "b/image.png"], ["a/annotation.label.json", "b/annotation.label.json"]),
        # Factory groups a mixed dataset by path without extension, which pairs image.label.json with image.png
        (["a/image.png", "flat/b.png"], ["a/image.label.json", "flat/b.label.json"]),
    ],
    ids=["item folders", "mixed"],
)
def test_annotation_files_pair_with_their_images_on_import(tmp_path, paths, expected):
    assert convert(tmp_path, paths) == expected


def convert_labeled(tmp_path, labels, annotations):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "a.png"), np.zeros((100, 100, 3), dtype=np.uint8))
    dataset = Dataset(labels=labels, files=[FileAnnotations(id="0", path="a.png", height=100, width=100, annotations=annotations)])
    convert_json_to_factory(dataset, images, tmp_path / "out")
    meta = tmp_path / "out" / ".meta.json"
    return json.loads(meta.read_text())["labelDictionary"] if meta.exists() else None


def box(box_id, label, x_min=0, y_min=0, x_max=100, y_max=100, angle=0):
    return BoxAnnotation(box_id, label, Box(x_min, y_min, x_max, y_max, angle))


def keypoint(name, x, y, box_id="b"):
    return KeypointAnnotation(f"{box_id}-{name}", name, Point2d(x, y), bounding_box_id=box_id)


def test_a_pose_label_carries_its_declared_points_where_they_sit_on_average(tmp_path):
    labels = [Label(id="tiger", color="#112233", annotation_type=AnnotationType.BOX, keypoints=["tail", "nose"]), Label(id="nose")]
    annotations = [box("b", "tiger"), keypoint("nose", 20, 40), box("c", "tiger", 0, 0, 50, 50), keypoint("nose", 20, 20, "c")]
    assert convert_labeled(tmp_path, labels, annotations) == {
        "tiger": {
            "id": "tiger",
            "annotationType": "Box",
            "order": 0,
            "color": "#112233",
            "points": [{"name": "tail", "x": 0.5, "y": 0.5}, {"name": "nose", "x": 0.3, "y": 0.4}],
        }
    }


def test_points_without_a_declared_layout_are_the_held_names_sorted(tmp_path):
    labels = [
        Label(id="part", annotation_type="Box"),
        Label(id="tip", annotation_type="Keypoint"),
        Label(id="base", annotation_type="Keypoint"),
    ]
    annotations = [box("b", "part"), keypoint("tip", 50, 50), keypoint("base", 50, 50)]
    assert [p["name"] for p in convert_labeled(tmp_path, labels, annotations)["part"]["points"]] == ["base", "tip"]


def test_shape_labels_are_typed_ordered_and_oriented(tmp_path):
    annotations = [
        PolygonAnnotation("p", "dent", Polygon([[0, 0], [10, 0], [10, 10]])),
        box("b", "scratch", angle=30),
    ]
    dictionary = convert_labeled(tmp_path, [Label(id="scratch", annotation_type="Box")], annotations)
    assert [(e["id"], e["annotationType"], e["order"], e.get("oriented")) for e in dictionary.values()] == [
        ("scratch", "Box", 0, True),
        ("dent", "Polygon", 1, None),
    ]


def test_an_unlinked_keypoint_links_to_the_one_box_it_lies_in(tmp_path):
    convert_labeled(tmp_path, [], [box("b", "part", 0, 0, 50, 50), box("c", "part", 60, 60, 90, 90), keypoint("tip", 10, 10, None)])
    annotations = json.loads((tmp_path / "out" / "a.label.json").read_text())["annotations"]
    assert [a["bounding_box_id"] for a in annotations if a["type"] == "Keypoint"] == ["b"]


@pytest.mark.parametrize(
    "keypoint_box, match",
    [(None, "not assigned to any box"), ("missing", "not found")],
    ids=["outside every box", "missing box"],
)
def test_a_keypoint_that_cannot_be_linked_is_refused_before_anything_is_written(tmp_path, keypoint_box, match):
    with pytest.raises(ValueError, match=match):
        convert_labeled(tmp_path, [], [box("b", "part", 0, 0, 50, 50), keypoint("tip", 70, 70, keypoint_box)])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "labels, annotations",
    [
        ([Label(id="dent", annotation_type="Box")], [PolygonAnnotation("p", "dent", Polygon([[0, 0], [10, 0], [10, 10]]))]),
        ([], [box("b", "dent"), PolygonAnnotation("p", "dent", Polygon([[0, 0], [10, 0], [10, 10]]))]),
        ([Label(id="dent", annotation_type="rectanglelabels")], [box("b", "dent")]),
    ],
    ids=["stated type differs from drawn", "drawn two ways", "unknown type"],
)
def test_a_label_without_one_factory_type_is_refused(tmp_path, labels, annotations):
    with pytest.raises(ValueError, match="label 'dent'"):
        convert_labeled(tmp_path, labels, annotations)


def test_an_image_factory_does_not_import_is_refused(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "a.bmp"), np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="a.bmp"):
        convert_json_to_factory(Dataset(labels=[], files=[FileAnnotations("0", "a.bmp", 4, 4)]), images, tmp_path / "out")


@pytest.mark.parametrize("name", [" part", "x" * 101])
def test_a_label_name_factory_refuses_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match="characters"):
        convert_labeled(tmp_path, [], [box("b", name)])


def test_a_point_outside_the_declared_layout_is_refused(tmp_path):
    with pytest.raises(ValueError, match="declares keypoints"):
        convert_labeled(tmp_path, [Label(id="part", keypoints=["tip"])], [box("b", "part"), keypoint("base", 1, 1)])
