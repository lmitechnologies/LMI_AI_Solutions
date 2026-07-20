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
