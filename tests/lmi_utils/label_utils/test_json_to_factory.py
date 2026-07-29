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
    label = Label(
        id="bolt", keypoint_ids=BOLT_KEYPOINTS, horizontal_flip_pairs=[["left-flange", "right-flange"]], skeleton=[[0, 1], [0, 2]]
    )
    output_dir = _convert(tmp_path, _pose_dataset(label))

    assert _meta(output_dir)["annotationSchema"] == {
        "type": "Pose",
        "version": 1,
        "coordinateDimensions": 3,
        "keypoints": {name: {"name": name} for name in BOLT_KEYPOINTS},
        "classes": {
            "bolt": {
                "name": "bolt",
                "keypointIds": BOLT_KEYPOINTS,
                "horizontalFlipPairs": [["left-flange", "right-flange"]],
                "skeleton": [[0, 1], [0, 2]],
            },
        },
    }


def test_a_class_with_no_declared_flip_declares_none(tmp_path):
    # Importable, but training may not flip it horizontally -- which is the point of stating it explicitly.
    output_dir = _convert(tmp_path, _pose_dataset(Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS)))

    assert _meta(output_dir)["annotationSchema"]["classes"]["bolt"]["horizontalFlipPairs"] is None


def test_a_symmetry_that_swaps_nothing_is_not_the_same_as_declaring_none():
    # An empty list is a statement: this class mirrors onto itself. None says the symmetry is unknown.
    schema = build_pose_schema([Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS, horizontal_flip_pairs=[])])

    assert schema["classes"]["bolt"]["horizontalFlipPairs"] == []


def test_a_keypoint_paired_with_itself_is_rejected():
    with pytest.raises(ValueError, match="with itself"):
        build_pose_schema([Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS, horizontal_flip_pairs=[["head", "head"]])])


def test_a_keypoint_used_in_two_pairs_is_rejected():
    with pytest.raises(ValueError, match="more than one horizontal flip pair"):
        build_pose_schema(
            [Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS, horizontal_flip_pairs=[["head", "left-flange"], ["head", "right-flange"]])]
        )


def test_a_flip_naming_an_undeclared_keypoint_is_rejected():
    with pytest.raises(ValueError, match="which it does not declare"):
        build_pose_schema([Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS, horizontal_flip_pairs=[["head", "elbow"]])])


def test_a_skeleton_edge_outside_the_layout_is_rejected():
    with pytest.raises(ValueError, match="outside its 3 keypoint slots"):
        build_pose_schema([Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS, skeleton=[[0, 3]])])


def test_a_scaffold_groups_observed_keypoints_under_their_box_class():
    # A starting point for a project whose keypoint layout was never declared, such as one built by hand in
    # Label Studio. The flip stays null: no annotation says which slot mirrors which.
    dataset = _pose_dataset(Label(id="bolt"))

    assert scaffold_pose_schema(dataset)["classes"] == {
        "bolt": {"name": "bolt", "keypointIds": ["head", "left-flange"], "horizontalFlipPairs": None}
    }


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

    assert scaffold_pose_schema(dataset)["classes"]["bolt"]["keypointIds"] == ["head"]


def test_every_split_directory_declares_the_schema(tmp_path):
    # Factory matches `.meta.json` at the exact root of the folder it imports, so a split nested below the output
    # root needs its own copy or its keypoints arrive with no declared layout.
    image_dir = tmp_path / "images"
    (image_dir / "train").mkdir(parents=True)
    (image_dir / "val").mkdir(parents=True)
    for split in ("train", "val"):
        cv2.imwrite(str(image_dir / split / "image.png"), np.zeros((20, 20, 3), dtype=np.uint8))

    label = Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS)
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
    output_dir = _convert(tmp_path, _pose_dataset(Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS)))

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
    output_dir = _convert(tmp_path, _pose_dataset(Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS), annotations))

    written = json.loads((output_dir / "image.label.json").read_text())["annotations"]
    assert [a["bounding_box_id"] for a in written if a["type"] == "Keypoint"] == ["box"]


def test_a_keypoint_owned_by_no_box_is_rejected(tmp_path):
    annotations = [
        BoxAnnotation("box", "bolt", Box(0, 0, 5, 5)),
        KeypointAnnotation("kp-head", "head", Point2d(18, 18)),
    ]
    with pytest.raises(ValueError, match="not assigned to any box"):
        _convert(tmp_path, _pose_dataset(Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS), annotations))


def test_a_class_and_its_keypoints_are_named_separately_from_their_ids():
    # What a person reads is the name; what every annotation, class map and model contract stores is the id.
    # Renaming later must not strand any of them, which is only possible while the two are separate fields.
    schema = build_pose_schema(
        [Label(id="bolt", name="Hex bolt", keypoint_ids=["head", "left-flange"])],
        {"head": "Head", "left-flange": "Left flange"},
    )

    assert schema["classes"]["bolt"]["name"] == "Hex bolt"
    assert schema["classes"]["bolt"]["keypointIds"] == ["head", "left-flange"]
    assert schema["keypoints"] == {"head": {"name": "Head"}, "left-flange": {"name": "Left flange"}}


def test_a_keypoint_two_classes_share_is_defined_once():
    # Two classes each declaring `head` mean one keypoint, which is why the vocabulary is a schema-level record
    # rather than a list per class.
    schema = build_pose_schema(
        [
            Label(id="bolt", keypoint_ids=["head", "left-flange"]),
            Label(id="screw", keypoint_ids=["head", "tip"]),
        ]
    )

    assert sorted(schema["keypoints"]) == ["head", "left-flange", "tip"]


def test_an_unnamed_class_or_keypoint_shows_as_its_id():
    schema = build_pose_schema([Label(id="bolt", keypoint_ids=["head"])])

    assert schema["classes"]["bolt"]["name"] == "bolt"
    assert schema["keypoints"]["head"] == {"name": "head"}


def test_an_id_a_source_name_cannot_produce_is_rejected():
    # Nothing in AIS invents ids; a caller building labels by hand has to derive them the same way a converter
    # does, or Factory would reject the dataset later with less to go on.
    with pytest.raises(ValueError, match="not a valid identifier"):
        build_pose_schema([Label(id="hex bolt", keypoint_ids=["head"])])


def test_two_classes_with_one_name_are_rejected():
    # Label Studio resolves a region's label by its alias or its displayed value, so a repeated name is ambiguous
    # to the annotation editor itself.
    with pytest.raises(ValueError, match="are both named 'Bolt'"):
        build_pose_schema(
            [
                Label(id="bolt_a", name="Bolt", keypoint_ids=["head"]),
                Label(id="bolt_b", name="Bolt", keypoint_ids=["head"]),
            ]
        )


def test_a_name_that_is_another_entrys_id_is_rejected():
    with pytest.raises(ValueError, match="another class's identifier"):
        build_pose_schema(
            [
                Label(id="bolt", name="Hex bolt", keypoint_ids=["head"]),
                Label(id="screw", name="bolt", keypoint_ids=["head"]),
            ]
        )


def test_a_name_label_studio_would_read_as_a_substitution_is_rejected():
    with pytest.raises(ValueError, match="task-data substitution"):
        build_pose_schema([Label(id="bolt", name="$bolt", keypoint_ids=["head"])])


def test_a_supplied_schema_is_checked_before_it_is_written(tmp_path):
    # A hand-authored schema passed through reaches Factory unchanged, so it is worth rejecting here, where the
    # message can name every problem at once.
    dataset = _pose_dataset(Label(id="bolt", keypoint_ids=BOLT_KEYPOINTS))
    schema = {
        "type": "Pose",
        "version": 1,
        "coordinateDimensions": 3,
        "keypoints": {"head": {"name": "Head"}},
        "classes": {"bolt": {"name": "Bolt", "keypointIds": ["head", "left-flange"], "horizontalFlipPairs": None}},
    }

    with pytest.raises(ValueError, match="does not define"):
        convert_json_to_factory(dataset, _image_dir(tmp_path), tmp_path / "factory", annotation_schema=schema)
