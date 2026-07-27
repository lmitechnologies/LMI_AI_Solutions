import json

import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.representations import Box, BoxAnnotation, Dataset, FileAnnotations, KeypointAnnotation, Label, Point2d
from lmi_utils.label_utils.json_to_factory import build_pose_schema, convert_json_to_factory, scaffold_pose_schema

# A two-sided class, so a flip mapping has something to swap and the unobserved slot stays declared.
BOLT_KEYPOINTS = ["head", "left-flange", "right-flange"]


def _image_dir(tmp_path):
    path = tmp_path / "images"
    path.mkdir()
    cv2.imwrite(str(path / "image.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    return path


def _pose_dataset(label, annotations=None):
    default = [
        BoxAnnotation("box", "bolt", Box(0, 0, 20, 20)),
        KeypointAnnotation("kp-head", "head", Point2d(10, 4, visibility=2), bounding_box_id="box"),
        KeypointAnnotation("kp-left", "left-flange", Point2d(4, 16), bounding_box_id="box"),
    ]
    return Dataset(
        labels=[label],
        files=[FileAnnotations("file", "image.png", 20, 20, default if annotations is None else annotations)],
    )


def _convert(tmp_path, dataset):
    output_dir = tmp_path / "factory"
    convert_json_to_factory(dataset, _image_dir(tmp_path), output_dir)
    return output_dir


def _meta(output_dir):
    return json.loads((output_dir / ".meta.json").read_text())


def test_declared_schema_reaches_the_root_meta(tmp_path):
    # The layout, its flip and its skeleton are declarations: they must survive conversion whole, including the
    # slot no annotation in this dataset observes.
    label = Label(id="bolt", keypoints=BOLT_KEYPOINTS, horizontal_flip=[0, 2, 1], skeleton=[[0, 1], [0, 2]])
    output_dir = _convert(tmp_path, _pose_dataset(label))

    assert _meta(output_dir)["annotationSchema"] == {
        "type": "Pose",
        "version": 1,
        "coordinateDimensions": 3,
        "classes": {
            "bolt": {"keypoints": BOLT_KEYPOINTS, "horizontalFlip": [0, 2, 1], "skeleton": [[0, 1], [0, 2]]},
        },
    }


def test_a_class_with_no_declared_flip_declares_none(tmp_path):
    # Importable, but training may not flip it horizontally -- which is the point of stating it explicitly.
    output_dir = _convert(tmp_path, _pose_dataset(Label(id="bolt", keypoints=BOLT_KEYPOINTS)))

    assert _meta(output_dir)["annotationSchema"]["classes"]["bolt"]["horizontalFlip"] is None


def test_a_flip_may_be_declared_by_keypoint_name():
    schema = build_pose_schema([Label(id="bolt", keypoints=BOLT_KEYPOINTS, horizontal_flip=["head", "right-flange", "left-flange"])])

    assert schema["classes"]["bolt"]["horizontalFlip"] == [0, 2, 1]


def test_a_flip_that_is_not_an_involution_is_rejected():
    # Flipping an image twice has to restore every keypoint, so a rotation of the slots is not a mirror.
    with pytest.raises(ValueError, match="not an involution"):
        build_pose_schema([Label(id="bolt", keypoints=BOLT_KEYPOINTS, horizontal_flip=[1, 2, 0])])


def test_a_flip_naming_an_undeclared_keypoint_is_rejected():
    with pytest.raises(ValueError, match="not one of the keypoints"):
        build_pose_schema([Label(id="bolt", keypoints=BOLT_KEYPOINTS, horizontal_flip=["head", "elbow", "left-flange"])])


def test_a_skeleton_edge_outside_the_layout_is_rejected():
    with pytest.raises(ValueError, match="outside its 3 keypoint slots"):
        build_pose_schema([Label(id="bolt", keypoints=BOLT_KEYPOINTS, skeleton=[[0, 3]])])


def test_a_scaffold_groups_observed_keypoints_under_their_box_class():
    # A starting point for a project whose keypoint layout was never declared, such as one built by hand in
    # Label Studio. The flip stays null: no annotation says which slot mirrors which.
    dataset = _pose_dataset(Label(id="bolt"))

    assert scaffold_pose_schema(dataset)["classes"] == {"bolt": {"keypoints": ["head", "left-flange"], "horizontalFlip": None}}


def test_a_scaffold_skips_keypoints_it_cannot_attribute():
    annotations = [
        BoxAnnotation("box", "bolt", Box(0, 0, 10, 10)),
        BoxAnnotation("box2", "bolt", Box(5, 5, 20, 20)),
        KeypointAnnotation("kp-head", "head", Point2d(2, 2)),
        # Inside both boxes, so which class owns the slot is a guess.
        KeypointAnnotation("kp-left", "left-flange", Point2d(7, 7)),
        # Inside neither.
        KeypointAnnotation("kp-right", "right-flange", Point2d(18, 2)),
    ]
    dataset = _pose_dataset(Label(id="bolt"), annotations)

    assert scaffold_pose_schema(dataset)["classes"]["bolt"]["keypoints"] == ["head"]


def test_every_split_directory_declares_the_schema(tmp_path):
    # Factory matches `.meta.json` at the exact root of the folder it imports, so a split nested below the output
    # root needs its own copy or its keypoints arrive with no declared layout.
    image_dir = tmp_path / "images"
    (image_dir / "train").mkdir(parents=True)
    (image_dir / "val").mkdir(parents=True)
    for split in ("train", "val"):
        cv2.imwrite(str(image_dir / split / "image.png"), np.zeros((20, 20, 3), dtype=np.uint8))

    label = Label(id="bolt", keypoints=BOLT_KEYPOINTS)
    dataset = Dataset(
        labels=[label],
        files=[
            FileAnnotations(
                split,
                f"{split}/image.png",
                20,
                20,
                [
                    BoxAnnotation("box", "bolt", Box(0, 0, 20, 20)),
                    KeypointAnnotation("kp-head", "head", Point2d(10, 4, visibility=2), bounding_box_id="box"),
                ],
            )
            for split in ("train", "val")
        ],
    )
    output_dir = tmp_path / "factory"
    convert_json_to_factory(dataset, image_dir, output_dir)

    schema = _meta(output_dir)["annotationSchema"]
    for split in ("train", "val"):
        assert _meta(output_dir / split)["annotationSchema"] == schema


def test_a_dataset_declaring_no_keypoints_writes_no_schema(tmp_path):
    dataset = Dataset(
        labels=[Label(id="bolt")],
        files=[FileAnnotations("file", "image.png", 20, 20, [BoxAnnotation("box", "bolt", Box(0, 0, 20, 20))])],
    )
    output_dir = _convert(tmp_path, dataset)

    assert not (output_dir / ".meta.json").exists()


def test_keypoints_reach_factory_linked_and_without_unset_fields(tmp_path):
    output_dir = _convert(tmp_path, _pose_dataset(Label(id="bolt", keypoints=BOLT_KEYPOINTS)))

    annotations = json.loads((output_dir / "image.label.json").read_text())["annotations"]
    by_id = {annotation["id"]: annotation for annotation in annotations}
    assert by_id["kp-head"]["bounding_box_id"] == "box"
    assert by_id["kp-head"]["value"] == {"x": 10.0, "y": 4.0, "visibility": 2}
    # An unset visibility means visible; Factory rejects an explicit null, so it must be absent instead.
    assert by_id["kp-left"]["value"] == {"x": 4.0, "y": 16.0}
    assert "link" not in by_id["box"]


def test_an_unlinked_keypoint_is_resolved_by_containment(tmp_path):
    annotations = [
        BoxAnnotation("box", "bolt", Box(0, 0, 20, 20)),
        KeypointAnnotation("kp-head", "head", Point2d(10, 4)),
    ]
    output_dir = _convert(tmp_path, _pose_dataset(Label(id="bolt", keypoints=BOLT_KEYPOINTS), annotations))

    written = json.loads((output_dir / "image.label.json").read_text())["annotations"]
    assert [a["bounding_box_id"] for a in written if a["type"] == "Keypoint"] == ["box"]


def test_a_keypoint_owned_by_no_box_is_rejected(tmp_path):
    annotations = [
        BoxAnnotation("box", "bolt", Box(0, 0, 5, 5)),
        KeypointAnnotation("kp-head", "head", Point2d(18, 18)),
    ]
    with pytest.raises(ValueError, match="not assigned to any box"):
        _convert(tmp_path, _pose_dataset(Label(id="bolt", keypoints=BOLT_KEYPOINTS), annotations))
