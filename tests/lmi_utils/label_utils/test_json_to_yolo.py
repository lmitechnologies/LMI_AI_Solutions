import cv2
import numpy as np
import pytest
import yaml

from lmi_utils.dataset_utils.representations import Box, BoxAnnotation, Dataset, FileAnnotations, KeypointAnnotation, Label, Point2d
from lmi_utils.label_utils.json_to_yolo import convert_to_yolo


def _write_pose_dataset(path, n_kpts):
    path.mkdir()
    cv2.imwrite(str(path / "image.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    layout = [f"point-{index}" for index in range(n_kpts)]
    annotations = [BoxAnnotation("box", "person", Box(0, 0, 20, 20))]
    if layout:
        annotations.append(KeypointAnnotation("point", layout[0], Point2d(10, 10), bounding_box_id="box"))
    dataset = Dataset(
        labels=[Label(id="person", keypoints=layout)],
        files=[FileAnnotations("file", "image.png", 20, 20, annotations)],
    )
    dataset.save(str(path / "labels.json"))


def _args(train_path, output_path, val_path=None):
    return {
        "path_train_json": str(train_path / "labels.json"),
        "path_val_json": str((val_path or train_path) / "labels.json"),
        "path_out": str(output_path),
        "path_train_imgs": str(train_path),
        "path_val_imgs": str(val_path or train_path),
        "target_classes": "all",
    }


# alpha sits top-left, beta bottom-right, so a written row is identifiable by its normalized center.
_ALPHA_BOX = ("alpha", Box(0, 0, 10, 10))
_BETA_BOX = ("beta", Box(10, 10, 20, 20))


def _write_detection_dataset(path, boxes):
    """A two-class dataset whose annotation order is `boxes`, so callers can force a given observation order."""
    path.mkdir()
    cv2.imwrite(str(path / "image.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    annotations = [BoxAnnotation(f"box-{index}", label_id, box) for index, (label_id, box) in enumerate(boxes)]
    dataset = Dataset(
        labels=[Label(id="alpha"), Label(id="beta")],
        files=[FileAnnotations("file", "image.png", 20, 20, annotations)],
    )
    dataset.save(str(path / "labels.json"))


def _val_rows_by_center(output_path):
    """The single val label file's rows keyed by rounded normalized center x, with the class index kept."""
    val_dir = output_path / "labels" / "val"
    (txt_file,) = list(val_dir.glob("*.txt"))
    rows = {}
    for line in txt_file.read_text().splitlines():
        parts = line.split()
        rows[round(float(parts[1]), 2)] = int(parts[0])
    return rows


def test_convert_to_yolo_indexes_val_against_the_shared_class_map(tmp_path):
    # Train observes alpha then beta; val observes them in the opposite order. Left to derive its own map the
    # val split would index beta as 0 -- disagreeing with train and with the model's class list. A provided
    # class map must govern both splits so beta keeps index 1 on val.
    train_path = tmp_path / "train"
    val_path = tmp_path / "val"
    _write_detection_dataset(train_path, [_ALPHA_BOX, _BETA_BOX])
    _write_detection_dataset(val_path, [_BETA_BOX, _ALPHA_BOX])

    class_map_path = tmp_path / "class_map.yaml"
    with open(class_map_path, "w") as stream:
        yaml.safe_dump({"names": {0: "alpha", 1: "beta"}}, stream)

    output_path = tmp_path / "output"
    args = _args(train_path, output_path, val_path)
    args["path_dataset_yaml"] = str(class_map_path)
    convert_to_yolo(args)

    val_rows = _val_rows_by_center(output_path)
    assert val_rows[0.25] == 0  # alpha, top-left
    assert val_rows[0.75] == 1  # beta, bottom-right -- the map's index, not val's observation order


def test_convert_to_yolo_writes_three_dimensional_keypoint_shape(tmp_path):
    train_path = tmp_path / "train"
    _write_pose_dataset(train_path, 2)

    convert_to_yolo(_args(train_path, tmp_path / "output"))

    with open(tmp_path / "output" / "dataset.yaml") as stream:
        dataset_yaml = yaml.safe_load(stream)
    assert dataset_yaml["kpt_shape"] == [2, 3]


def test_convert_to_yolo_rejects_train_val_keypoint_mismatch(tmp_path):
    train_path = tmp_path / "train"
    val_path = tmp_path / "val"
    _write_pose_dataset(train_path, 2)
    _write_pose_dataset(val_path, 3)

    with pytest.raises(ValueError, match="different keypoint counts"):
        convert_to_yolo(_args(train_path, tmp_path / "output", val_path))
