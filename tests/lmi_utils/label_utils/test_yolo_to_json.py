import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from lmi_utils.dataset_utils.representations import AnnotationType, Box, Dataset
from lmi_utils.label_utils.json_to_yolo import convert_to_yolo
from lmi_utils.label_utils.yolo_to_json import convert_yolo_to_json, main

WIDTH, HEIGHT = 40, 20


def write_yolo(root, config, labels, splits=("train",)):
    """A YOLO dataset of one image per split, each labeled with `labels` rows."""
    for split in splits:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        cv2.imwrite(str(root / "images" / split / "a.png"), np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        (root / "labels" / split / "a.txt").write_text("\n".join(labels) + "\n")
    config = {"train": "images/train", **config}
    (root / "data.yaml").write_text(yaml.safe_dump(config))
    return root / "data.yaml"


def convert(path, **kwargs):
    return Dataset.load(str(convert_yolo_to_json(path, **kwargs)))


def shapes(dataset, kind):
    return [a for f in dataset.files for a in f.annotations if a.type == kind]


def test_pose_rows_become_boxes_with_linked_keypoints(tmp_path):
    path = write_yolo(tmp_path, {"names": {0: "tiger"}, "kpt_shape": [3, 2]}, ["0 0.5 0.5 0.5 0.5 0.25 0.5 0 0 0.75 0.5"])
    dataset = convert(path)

    [box] = shapes(dataset, AnnotationType.BOX)
    assert (box.label_id, box.value.x_min, box.value.y_min, box.value.x_max, box.value.y_max) == ("tiger", 10, 5, 30, 15)
    keypoints = shapes(dataset, AnnotationType.KEYPOINT)
    # the zeroed slot is unlabeled, not a point at the origin
    assert [(k.label_id, k.value.x, k.value.y, k.value.visibility, k.parent_id) for k in keypoints] == [
        ("point-0", 10, 10, None, box.id),
        ("point-2", 30, 10, None, box.id),
    ]
    assert dataset.labels[0].keypoints == ["point-0", "point-1", "point-2"]
    assert [(label.id, label.annotation_type) for label in dataset.labels[1:]] == [
        (name, AnnotationType.KEYPOINT.value) for name in ("point-0", "point-1", "point-2")
    ]


def test_default_keypoint_names_sort_in_slot_order(tmp_path):
    path = write_yolo(tmp_path, {"names": ["a"], "kpt_shape": [11, 2]}, [])
    assert convert(path).labels[0].keypoints == sorted(f"point-{slot:02d}" for slot in range(11))


def test_visibility_zero_drops_a_keypoint_and_others_keep_theirs(tmp_path):
    path = write_yolo(tmp_path, {"names": ["a"], "kpt_shape": [2, 3]}, ["0 0.5 0.5 1 1 0.5 0.5 0 0.25 0.25 1"])
    [keypoint] = shapes(convert(path), AnnotationType.KEYPOINT)
    assert (keypoint.label_id, keypoint.value.visibility) == ("point-1", 1)


def test_keypoint_names_are_given_on_the_command_line(tmp_path):
    path = write_yolo(tmp_path, {"names": ["a"], "kpt_shape": [2, 2]}, [])
    assert convert(path, keypoint_names=["nose", "tail"]).labels[0].keypoints == ["nose", "tail"]


def test_kpt_names_give_each_class_its_own_layout(tmp_path):
    config = {"names": ["big", "small"], "kpt_shape": [2, 3], "kpt_names": {0: ["x", "y"], 1: ["z"]}}
    path = write_yolo(tmp_path, config, ["1 0.5 0.5 1 1 0.5 0.5 2 0 0 0"])
    dataset = convert(path)
    assert [label.keypoints for label in dataset.labels[:2]] == [["x", "y"], ["z"]]
    assert [k.label_id for k in shapes(dataset, AnnotationType.KEYPOINT)] == ["z"]


def test_a_labeled_slot_beyond_the_class_layout_is_refused(tmp_path):
    config = {"names": ["a"], "kpt_shape": [2, 3], "kpt_names": {0: ["x"]}}
    path = write_yolo(tmp_path, config, ["0 0.5 0.5 1 1 0.5 0.5 2 0.5 0.5 2"])
    with pytest.raises(ValueError, match="slot 1"):
        convert_yolo_to_json(path)


def test_kpt_names_keyed_by_class_name_are_refused(tmp_path):
    path = write_yolo(tmp_path, {"names": ["a"], "kpt_shape": [1, 2], "kpt_names": {"a": ["x"]}}, [])
    with pytest.raises(ValueError, match="class index"):
        convert_yolo_to_json(path)


def test_detect_and_segment_rows(tmp_path):
    path = write_yolo(tmp_path, {"names": ["box", "outline"]}, ["0 0.5 0.5 0.5 0.5", "1 0 0 1 0 1 1"])
    dataset = convert(path)
    [polygon] = shapes(dataset, AnnotationType.POLYGON)
    assert polygon.value.points == [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT]]
    assert [(label.id, label.annotation_type) for label in dataset.labels] == [("box", "Box"), ("outline", "Polygon")]


def test_splits_keep_their_paths_and_a_shared_split_is_read_once(tmp_path):
    path = write_yolo(tmp_path, {"names": ["a"], "val": "images/val", "test": "images/train"}, ["0 0.5 0.5 1 1"], ("train", "val"))
    dataset = convert(path)
    assert [f.path for f in dataset.files] == ["images/train/a.png", "images/val/a.png"]
    assert len({f.id for f in dataset.files}) == 2


def test_a_stale_path_resolves_against_the_yaml_directory(tmp_path):
    path = write_yolo(tmp_path, {"names": ["a"], "path": "moved-away"}, [])
    assert convert(path).files[0].path == "images/train/a.png"


def test_a_missing_split_is_refused(tmp_path):
    path = write_yolo(tmp_path, {"names": ["a"], "val": "images/val"}, [])
    with pytest.raises(FileNotFoundError, match="val"):
        convert_yolo_to_json(path)


def test_pose_round_trips_through_json_to_yolo(tmp_path):
    row = "0 0.5 0.5 0.5 0.5 0.25 0.5 2 0 0 0 0.75 0.5 1"
    path = write_yolo(tmp_path / "in", {"names": ["a"], "kpt_shape": [3, 3]}, [row])
    output = convert_yolo_to_json(path)
    args = {"path_train_json": str(output), "path_val_json": "labels.json", "path_out": str(tmp_path / "out")}
    convert_to_yolo({**args, "path_train_imgs": str(tmp_path / "in"), "path_val_imgs": None, "target_classes": "all"})

    [written] = (tmp_path / "out" / "labels" / "train").glob("*.txt")
    assert [float(v) for v in written.read_text().split()] == pytest.approx([float(v) for v in row.split()])


def test_obb_rows_become_turned_boxes_that_round_trip_through_json_to_yolo(tmp_path):
    # a 10 x 5 pixel box turned 30 degrees clockwise about its top-left corner at (10, 5)
    corners = Box(10, 5, 20, 10, 30)._corners()
    row = "0 " + " ".join(f"{x / WIDTH} {y / HEIGHT}" for x, y in corners)
    output = convert_yolo_to_json(write_yolo(tmp_path / "in", {"names": ["part"]}, [row]), obb=True)

    [box] = shapes(Dataset.load(str(output)), AnnotationType.BOX)
    assert [box.value.x_max - box.value.x_min, box.value.y_max - box.value.y_min, box.value.angle] == pytest.approx([10, 5, 30])
    args = {"path_train_json": str(output), "path_val_json": "labels.json", "path_out": str(tmp_path / "out"), "obb": True}
    convert_to_yolo({**args, "path_train_imgs": str(tmp_path / "in"), "path_val_imgs": None, "target_classes": "all"})
    [written] = (tmp_path / "out" / "labels" / "train").glob("*.txt")
    values = [float(v) for v in written.read_text().split()[1:]]
    expected = sorted((x / WIDTH, y / HEIGHT) for x, y in corners)
    # json_to_yolo writes 4 decimals
    assert np.ravel(sorted(zip(values[::2], values[1::2]))) == pytest.approx(np.ravel(expected), abs=1e-4)


def test_an_obb_row_of_the_wrong_length_is_refused(tmp_path):
    with pytest.raises(ValueError, match="oriented box"):
        convert_yolo_to_json(write_yolo(tmp_path, {"names": ["part"]}, ["0 0.5 0.5 0.5 0.5"]), obb=True)


def test_obb_with_keypoints_is_refused(tmp_path):
    with pytest.raises(ValueError, match="kpt_shape"):
        convert_yolo_to_json(write_yolo(tmp_path, {"names": ["a"], "kpt_shape": [1, 2]}, []), obb=True)


def test_an_image_directory_replaces_the_splits_and_becomes_the_root(tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.png"), np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    (folder / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    path = tmp_path / "classes.yaml"
    path.write_text(yaml.safe_dump({"names": ["a"]}))

    output = convert_yolo_to_json(path, images=folder)
    assert output == folder / "labels.json"
    [file] = Dataset.load(str(output)).files
    assert (file.path, len(file.annotations)) == ("a.png", 1)


def test_a_yaml_without_splits_needs_an_image_directory(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text(yaml.safe_dump({"names": ["a"]}))
    with pytest.raises(ValueError, match="splits"):
        convert_yolo_to_json(path)


def test_a_dataset_given_relative_to_the_working_directory_finds_its_labels(tmp_path, monkeypatch):
    write_yolo(tmp_path, {"names": ["a"]}, ["0 0.5 0.5 0.5 0.5"])
    monkeypatch.chdir(tmp_path)
    assert len(shapes(convert(Path("data.yaml")), AnnotationType.BOX)) == 1


def test_labels_beside_images_under_an_images_directory_are_found(tmp_path):
    folder = tmp_path / "images" / "batch"
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "a.png"), np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    (folder / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (tmp_path / "classes.yaml").write_text(yaml.safe_dump({"names": ["a"]}))
    assert len(shapes(convert(tmp_path / "classes.yaml", images=folder), AnnotationType.BOX)) == 1


def test_a_class_written_as_a_float_is_read(tmp_path):
    assert len(shapes(convert(write_yolo(tmp_path, {"names": ["a"]}, ["0.0 0.5 0.5 0.5 0.5"])), AnnotationType.BOX)) == 1


@pytest.mark.parametrize("row, match", [("0.5 0.5 0.5 0.5 0.5", "class index"), ("0 0.5 x 0.5 0.5", "could not convert")])
def test_a_bad_row_is_refused_with_its_file_and_line(tmp_path, row, match):
    with pytest.raises(ValueError, match=rf"a\.txt:1: .*{match}"):
        convert_yolo_to_json(write_yolo(tmp_path, {"names": ["a"]}, [row]))


def test_command_line_keypoint_names_are_trimmed(tmp_path, monkeypatch):
    path = write_yolo(tmp_path, {"names": ["a"], "kpt_shape": [2, 2]}, [])
    monkeypatch.setattr(sys, "argv", ["yolo_to_json", "-y", str(path), "--keypoint_names", "nose, tail"])
    main()
    assert Dataset.load(str(tmp_path / "labels.json")).labels[0].keypoints == ["nose", "tail"]
