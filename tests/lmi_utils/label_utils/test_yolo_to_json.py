import json

import cv2
import numpy as np
import pytest
import yaml

from lmi_utils.dataset_utils.representations import Dataset
from lmi_utils.label_utils.json_to_factory import convert_json_to_factory
from lmi_utils.label_utils.yolo_to_json import convert_yolo_to_json

# The uniform slot space a two-class pose model trains in: bolt owns the first three slots, tab the last two.
# Slot 1 and 2 mirror each other, as do 3 and 4, so each class's flip closes over the slots it owns.
KPT_NAMES = {
    "bolt": ["head", "left-flange", "right-flange", "__unused_3", "__unused_4"],
    "tab": ["__unused_0", "__unused_1", "__unused_2", "left-edge", "right-edge"],
}
FLIP_IDX = [0, 2, 1, 4, 3]


def _write_split(root, split, rows):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "image.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    (label_dir / "image.txt").write_text("\n".join(rows))


def _write_dataset(tmp_path, rows, names=None, kpt_shape=(5, 3), flip_idx=FLIP_IDX, kpt_names=KPT_NAMES, splits=("train",)):
    root = tmp_path / "yolo"
    root.mkdir()
    for split in splits:
        _write_split(root, split, rows)
    config = {"path": str(root), "train": "images/train", "val": "images/val", "names": names or {0: "bolt", 1: "tab"}}
    if kpt_shape:
        config["kpt_shape"] = list(kpt_shape)
    if flip_idx:
        config["flip_idx"] = list(flip_idx)
    if kpt_names:
        config["kpt_names"] = kpt_names
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(yaml.dump(config, sort_keys=False))
    return dataset_yaml


def _convert(tmp_path, dataset_yaml, **kwargs):
    """Run the whole pipeline: YOLO to the dataset json, then the json to a Factory dataset directory.

    The json is reloaded from disk rather than passed along in memory, so every assertion below also covers the
    declaration surviving the intermediate file.
    """
    root = dataset_yaml.parent
    convert_yolo_to_json(dataset_yaml, **kwargs)
    output_dir = tmp_path / "factory"
    convert_json_to_factory(Dataset.load(str(root / "labels.json")), root, output_dir)
    return output_dir


def _schema(output_dir):
    return json.loads((output_dir / ".meta.json").read_text())["annotationSchema"]


def _annotations(output_dir, split="train"):
    return json.loads((output_dir / "images" / split / "image.label.json").read_text())["annotations"]


# A bolt filling the image, observing its head and left flange; its right flange and both padded slots are zeroed.
BOLT_ROW = "0 0.5 0.5 1.0 1.0 0.5 0.2 2 0.2 0.8 1 0 0 0 0 0 0 0 0 0"
# A tab in the lower right, observing both of its own slots and none of bolt's.
TAB_ROW = "1 0.75 0.75 0.5 0.5 0 0 0 0 0 0 0 0 0 0.6 0.7 2 0.9 0.7 2"


def test_per_class_layouts_drop_the_slots_a_class_does_not_own(tmp_path):
    output_dir = _convert(tmp_path, _write_dataset(tmp_path, [BOLT_ROW, TAB_ROW]))

    assert _schema(output_dir)["classes"] == {
        "bolt": {"keypoints": ["head", "left-flange", "right-flange"], "horizontalFlip": [0, 2, 1]},
        # The global flip is translated into each class's own order, so tab's pair mirrors at local 0 and 1.
        "tab": {"keypoints": ["left-edge", "right-edge"], "horizontalFlip": [1, 0]},
    }


def test_ultralytics_numeric_kpt_name_keys_are_resolved_by_class_index(tmp_path):
    # Ultralytics' built-in pose yamls use numeric keys here, independently of the class names in `names`.
    kpt_names = {
        0: ["head", "left-flange", "right-flange", "__unused_3", "__unused_4"],
        1: ["__unused_0", "__unused_1", "__unused_2", "left-edge", "right-edge"],
    }
    output_dir = _convert(tmp_path, _write_dataset(tmp_path, [BOLT_ROW, TAB_ROW], kpt_names=kpt_names))

    assert _schema(output_dir)["classes"] == {
        "bolt": {"keypoints": ["head", "left-flange", "right-flange"], "horizontalFlip": [0, 2, 1]},
        "tab": {"keypoints": ["left-edge", "right-edge"], "horizontalFlip": [1, 0]},
    }


def test_conflicting_index_and_class_name_layouts_are_rejected(tmp_path):
    kpt_names = {
        0: ["head", "left-flange", "right-flange", "__unused_3", "__unused_4"],
        "bolt": ["head", "right-flange", "left-flange", "__unused_3", "__unused_4"],
        "tab": KPT_NAMES["tab"],
    }
    dataset_yaml = _write_dataset(tmp_path, [BOLT_ROW, TAB_ROW], kpt_names=kpt_names)

    with pytest.raises(ValueError, match="disagree for class 'bolt'"):
        _convert(tmp_path, dataset_yaml)


def test_rows_become_boxes_with_their_observed_keypoints_linked(tmp_path):
    output_dir = _convert(tmp_path, _write_dataset(tmp_path, [BOLT_ROW, TAB_ROW]))

    annotations = _annotations(output_dir)
    boxes = [annotation for annotation in annotations if annotation["type"] == "Box"]
    assert [box["label_id"] for box in boxes] == ["bolt", "tab"]
    assert boxes[0]["value"] == {"x_min": 0.0, "y_min": 0.0, "x_max": 20.0, "y_max": 20.0, "angle": 0.0}

    keypoints = [annotation for annotation in annotations if annotation["type"] == "Keypoint"]
    # The zeroed slots -- bolt's unobserved right flange and each class's padding -- produce no observation.
    assert [(kp["label_id"], kp["bounding_box_id"]) for kp in keypoints] == [
        ("head", boxes[0]["id"]),
        ("left-flange", boxes[0]["id"]),
        ("left-edge", boxes[1]["id"]),
        ("right-edge", boxes[1]["id"]),
    ]
    assert keypoints[0]["value"] == {"x": 10.0, "y": 4.0, "visibility": 2}


def test_a_file_declaring_no_kpt_names_gives_every_class_the_whole_layout(tmp_path):
    # A generic YOLO pose file states one layout for the model, which is the only sound reading of it.
    dataset_yaml = _write_dataset(
        tmp_path, ["0 0.5 0.5 1.0 1.0 0.5 0.2 2 0.2 0.8 1"], names={0: "bolt"}, kpt_shape=(2, 3), flip_idx=[1, 0], kpt_names=None
    )
    output_dir = _convert(tmp_path, dataset_yaml, keypoint_names=["left-flange", "right-flange"])

    assert _schema(output_dir)["classes"] == {"bolt": {"keypoints": ["left-flange", "right-flange"], "horizontalFlip": [1, 0]}}


def test_unnamed_keypoints_are_named_positionally(tmp_path):
    dataset_yaml = _write_dataset(
        tmp_path, ["0 0.5 0.5 1.0 1.0 0.5 0.2 2 0.2 0.8 1"], names={0: "bolt"}, kpt_shape=(2, 3), flip_idx=None, kpt_names=None
    )
    output_dir = _convert(tmp_path, dataset_yaml)

    assert _schema(output_dir)["classes"]["bolt"] == {"keypoints": ["point-0", "point-1"], "horizontalFlip": None}


def test_a_flip_leaving_the_class_declares_no_flip(tmp_path):
    # Slot 1 mirrors to slot 3, which bolt does not own. A partial mapping would drop the keypoint under a flip
    # instead of mirroring it, so the class declares none.
    flip_idx = [0, 3, 4, 1, 2]
    output_dir = _convert(tmp_path, _write_dataset(tmp_path, [BOLT_ROW, TAB_ROW], flip_idx=flip_idx))

    assert _schema(output_dir)["classes"]["bolt"]["horizontalFlip"] is None


def test_two_dimensional_keypoints_carry_no_visibility(tmp_path):
    # The declaration is normalized to the standard 3; only the point values keep their source fidelity
    dataset_yaml = _write_dataset(
        tmp_path, ["0 0.5 0.5 1.0 1.0 0.5 0.2 0.2 0.8"], names={0: "bolt"}, kpt_shape=(2, 2), flip_idx=None, kpt_names=None
    )
    output_dir = _convert(tmp_path, dataset_yaml)

    assert _schema(output_dir)["coordinateDimensions"] == 3
    keypoints = [annotation for annotation in _annotations(output_dir) if annotation["type"] == "Keypoint"]
    assert [kp["value"] for kp in keypoints] == [{"x": 10.0, "y": 4.0}, {"x": 4.0, "y": 16.0}]


def test_every_declared_split_is_converted_and_keeps_its_own_items(tmp_path):
    dataset_yaml = _write_dataset(tmp_path, [BOLT_ROW], splits=("train", "val"))
    output_dir = _convert(tmp_path, dataset_yaml)

    assert (output_dir / "images" / "train" / "image.png").exists()
    assert (output_dir / "images" / "val" / "image.png").exists()
    assert len(_annotations(output_dir, "val")) == 3


def test_a_row_of_the_wrong_width_is_rejected(tmp_path):
    dataset_yaml = _write_dataset(tmp_path, ["0 0.5 0.5 1.0 1.0 0.5 0.2 2"])

    with pytest.raises(ValueError, match="has 7 values for the 19"):
        _convert(tmp_path, dataset_yaml)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kpt_shape", [5, 4], "coordinate dimensions must be 2 or 3"),
        ("flip_idx", [0, 1, 1, 3, 4], "must be a permutation"),
        ("flip_idx", [1, 2, 0, 4, 3], "is not an involution"),
    ],
)
def test_malformed_pose_contracts_are_rejected_before_json_is_written(tmp_path, field, value, message):
    dataset_yaml = _write_dataset(tmp_path, [BOLT_ROW, TAB_ROW])
    config = yaml.safe_load(dataset_yaml.read_text())
    config[field] = value
    dataset_yaml.write_text(yaml.dump(config, sort_keys=False))

    with pytest.raises(ValueError, match=message):
        convert_yolo_to_json(dataset_yaml)
    assert not (dataset_yaml.parent / "labels.json").exists()


def test_a_detection_dataset_converts_without_a_schema(tmp_path):
    dataset_yaml = _write_dataset(tmp_path, ["0 0.5 0.5 1.0 1.0"], kpt_shape=None, flip_idx=None, kpt_names=None)
    output_dir = _convert(tmp_path, dataset_yaml)

    assert not (output_dir / ".meta.json").exists()
    assert [annotation["type"] for annotation in _annotations(output_dir)] == ["Box"]


def test_a_stale_yaml_path_falls_back_to_the_dataset_root(tmp_path):
    # `path` records where the dataset was written; unpacking it elsewhere must not silently convert nothing.
    dataset_yaml = _write_dataset(tmp_path, [BOLT_ROW])
    config = yaml.safe_load(dataset_yaml.read_text())
    config["path"] = "/nonexistent/build/output"
    dataset_yaml.write_text(yaml.dump(config, sort_keys=False))

    assert len(_annotations(_convert(tmp_path, dataset_yaml))) == 3
