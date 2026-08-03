import json
import logging

import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.pose_identifiers import derive_pose_id, pose_label_alias
from lmi_utils.dataset_utils.representations import AnnotationType, Dataset
from lmi_utils.label_utils.json_to_factory import convert_json_to_factory
from lmi_utils.label_utils.lst_to_json import get_annotations_from_json, load_pose_schema, main

WIDTH = HEIGHT = 20
# Label Studio stores every coordinate as a percentage of the image.
PERCENT = 100 / WIDTH

# The class and one keypoint are named differently from their ids, so every assertion below also covers the
# identity split: what the annotator reads never reaches an annotation.
POSE_SCHEMA = {
    "type": "Pose",
    "version": 1,
    "coordinateDimensions": 3,
    "keypoints": {
        "head": {"name": "Head"},
        "left-flange": {"name": "left-flange"},
        "right-flange": {"name": "right-flange"},
    },
    "classes": {
        "bolt": {
            "name": "Hex bolt",
            "keypointIds": ["head", "left-flange", "right-flange"],
            "horizontalFlipPairs": [["left-flange", "right-flange"]],
            "skeleton": [[0, 1], [0, 2]],
        },
    },
}


def _region(region_id, result_type, value):
    return {
        "id": region_id,
        "original_width": WIDTH,
        "original_height": HEIGHT,
        "image_rotation": 0,
        "from_name": result_type.replace("labels", ""),
        "to_name": "image",
        "type": result_type,
        "value": value,
    }


def _box(region_id, label_id, x_min, y_min, x_max, y_max):
    value = {
        "x": x_min * PERCENT,
        "y": y_min * PERCENT,
        "width": (x_max - x_min) * PERCENT,
        "height": (y_max - y_min) * PERCENT,
        "rectanglelabels": [pose_label_alias(label_id)],
    }
    return _region(region_id, "rectanglelabels", value)


def _keypoint(region_id, label_id, x, y, parent_id=None):
    # A Label Studio keypoint carries no visibility; an unset one reaches Factory as visible.
    region = _region(region_id, "keypointlabels", {"x": x * PERCENT, "y": y * PERCENT, "keypointlabels": [pose_label_alias(label_id)]})
    if parent_id is not None:
        region["parentID"] = parent_id
    return region


def _relation(from_id, to_id):
    return {"type": "relation", "from_id": from_id, "to_id": to_id, "direction": "right"}


def _write_export(tmp_path, results, name="image.png", url=None, image_name=None):
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_dir / (image_name or name)), np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    export = [{"data": {"image": url or f"/data/upload/1/{name}"}, "annotations": [{"result": results}]}]
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export))
    return export_path, image_dir


def _write_schema(tmp_path, schema=POSE_SCHEMA):
    meta_path = tmp_path / ".meta.json"
    meta_path.write_text(json.dumps({"annotationSchema": schema}))
    return meta_path


def _convert(tmp_path, results, pose_schema=None, unlinked_keypoints="error", **export_kwargs):
    """Run the whole pipeline: the Label Studio export to the dataset json, then that json to a Factory dataset.

    The json is reloaded from disk rather than passed along in memory, so every assertion below also covers the
    declaration surviving the intermediate file.
    """
    export_path, image_dir = _write_export(tmp_path, results, **export_kwargs)
    files, labels, keypoint_names = get_annotations_from_json(str(export_path), str(image_dir), pose_schema=pose_schema)
    json_path = tmp_path / "labels.json"
    Dataset(
        labels=labels,
        files=files,
        coordinate_dimensions=(pose_schema or {}).get("coordinateDimensions"),
        keypoints=keypoint_names,
    ).save(str(json_path))

    output_dir = tmp_path / "factory"
    convert_json_to_factory(Dataset.load(str(json_path)), image_dir, output_dir, unlinked_keypoints=unlinked_keypoints)
    return output_dir


def _schema(output_dir):
    return json.loads((output_dir / ".meta.json").read_text())["annotationSchema"]


def _annotations(output_dir):
    return json.loads((output_dir / "image.label.json").read_text())["annotations"]


# One bolt filling the image, with two of its three declared slots observed.
ONE_BOLT = [
    _box("box1", "bolt", 0, 0, WIDTH, HEIGHT),
    _keypoint("kp1", "head", 10, 4),
    _keypoint("kp2", "left-flange", 4, 16),
]

# Two bolts whose boxes overlap, so containment alone cannot say which one owns a keypoint in the overlap.
TWO_BOLTS = [
    _box("box1", "bolt", 0, 0, 12, HEIGHT),
    _box("box2", "bolt", 8, 0, WIDTH, HEIGHT),
    _keypoint("kp1", "head", 10, 4),
    _keypoint("kp2", "head", 10, 16),
    _relation("kp1", "box1"),
    _relation("box2", "kp2"),
]


def test_the_declared_schema_reaches_factory(tmp_path):
    # The export states no layout, so the schema is supplied alongside it and must arrive whole -- including the
    # right flange, which no annotation here observes.
    output_dir = _convert(tmp_path, ONE_BOLT, pose_schema=POSE_SCHEMA)

    assert _schema(output_dir) == POSE_SCHEMA


def test_declared_keypoints_are_slots_rather_than_classes(tmp_path):
    output_dir = _convert(tmp_path, ONE_BOLT, pose_schema=POSE_SCHEMA)

    # A keypoint label names a slot of its class; only the box class is a class of the dataset.
    assert list(_schema(output_dir)["classes"]) == ["bolt"]
    keypoints = [annotation for annotation in _annotations(output_dir) if annotation["type"] == "Keypoint"]
    assert [kp["label_id"] for kp in keypoints] == ["head", "left-flange"]
    assert keypoints[0]["value"] == {"x": 10.0, "y": 4.0}


def test_relations_decide_which_box_owns_a_keypoint(tmp_path):
    output_dir = _convert(tmp_path, TWO_BOLTS, pose_schema=POSE_SCHEMA)

    annotations = _annotations(output_dir)
    boxes = [annotation for annotation in annotations if annotation["type"] == "Box"]
    keypoints = [annotation for annotation in annotations if annotation["type"] == "Keypoint"]
    # The second relation is drawn box to keypoint; either direction links the pair.
    assert [kp["bounding_box_id"] for kp in keypoints] == [boxes[0]["id"], boxes[1]["id"]]


def test_native_parent_ids_decide_which_box_owns_a_keypoint(tmp_path):
    # Regions are deliberately out of order: parentID is resolved after the complete result is indexed.
    results = [
        _keypoint("kp1", "head", 10, 4, parent_id="box1"),
        _box("box1", "bolt", 0, 0, 12, HEIGHT),
        _box("box2", "bolt", 8, 0, WIDTH, HEIGHT),
        _keypoint("kp2", "head", 10, 16, parent_id="box2"),
        # A conflicting legacy relation cannot override native ownership.
        _relation("kp1", "box2"),
    ]
    output_dir = _convert(tmp_path, results, pose_schema=POSE_SCHEMA)

    annotations = _annotations(output_dir)
    boxes = [annotation for annotation in annotations if annotation["type"] == "Box"]
    keypoints = [annotation for annotation in annotations if annotation["type"] == "Keypoint"]
    assert [kp["bounding_box_id"] for kp in keypoints] == [boxes[0]["id"], boxes[1]["id"]]


def test_an_invalid_native_parent_id_is_rejected(tmp_path):
    results = ONE_BOLT + [_keypoint("kp3", "right-flange", 10, 10, parent_id="missing")]

    with pytest.raises(ValueError, match="parentID 'missing'.*does not exist"):
        _convert(tmp_path, results, pose_schema=POSE_SCHEMA)


def test_a_keypoint_in_two_boxes_and_no_relation_is_rejected(tmp_path):
    # Without the relations the overlap is genuinely ambiguous, and a guess would mislabel an instance.
    results = [result for result in TWO_BOLTS if result["type"] != "relation"]

    with pytest.raises(ValueError, match="contained by multiple boxes"):
        _convert(tmp_path, results, pose_schema=POSE_SCHEMA)


def test_an_unrelated_keypoint_falls_back_to_containment(tmp_path):
    output_dir = _convert(tmp_path, ONE_BOLT, pose_schema=POSE_SCHEMA)

    boxes = [annotation for annotation in _annotations(output_dir) if annotation["type"] == "Box"]
    keypoints = [annotation for annotation in _annotations(output_dir) if annotation["type"] == "Keypoint"]
    assert {kp["bounding_box_id"] for kp in keypoints} == {boxes[0]["id"]}


def test_a_keypoint_outside_the_declared_vocabulary_is_rejected(tmp_path):
    results = ONE_BOLT + [_keypoint("kp3", "elbow", 6, 6)]

    with pytest.raises(ValueError, match=r"\['elbow'\] are not declared"):
        _convert(tmp_path, results, pose_schema=POSE_SCHEMA)


def test_a_schema_may_be_read_from_a_dataset_directory(tmp_path):
    _write_schema(tmp_path)

    assert load_pose_schema(tmp_path) == POSE_SCHEMA
    assert load_pose_schema(tmp_path / ".meta.json") == POSE_SCHEMA


def test_a_two_dimensional_declaration_is_normalized_to_three(tmp_path):
    schema = json.loads(json.dumps(POSE_SCHEMA))
    schema["coordinateDimensions"] = 2
    output_dir = _convert(tmp_path, ONE_BOLT, pose_schema=schema)

    assert _schema(output_dir)["coordinateDimensions"] == 3


def test_a_partly_observed_instance_keeps_the_declared_layout(tmp_path):
    # Real projects skip slots an image does not show; the layout is a declaration, so it stays three wide.
    results = [_box("box1", "bolt", 0, 0, WIDTH, HEIGHT), _keypoint("kp1", "head", 10, 4)]
    output_dir = _convert(tmp_path, results, pose_schema=POSE_SCHEMA)

    assert _schema(output_dir)["classes"]["bolt"]["keypointIds"] == ["head", "left-flange", "right-flange"]
    assert len([a for a in _annotations(output_dir) if a["type"] == "Keypoint"]) == 1


# A project Factory created identifies its images by an API URL whose tail is a fixed word, not a file name.
ITEM_ID = "019f8d13-00b5-749b-b1e4-71ee316236a8"
ITEM_URL = f"/api/v1/annotation-projects/019f8d12-fff9-739f-a13b-48396d74479b/items/{ITEM_ID}/image"


def test_an_api_url_resolves_by_the_item_it_names(tmp_path):
    output_dir = _convert(tmp_path, ONE_BOLT, pose_schema=POSE_SCHEMA, url=ITEM_URL, image_name=f"{ITEM_ID}.png")

    assert (output_dir / f"{ITEM_ID}.label.json").exists()


def test_an_image_the_task_was_not_annotated_on_is_rejected(tmp_path):
    # Coordinates are percentages of the size the export recorded, so the wrong image rescales the whole task.
    export_path, image_dir = _write_export(tmp_path, ONE_BOLT)
    cv2.imwrite(str(image_dir / "image.png"), np.zeros((HEIGHT * 2, WIDTH, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="annotated on a 20x20 image"):
        get_annotations_from_json(str(export_path), str(image_dir), pose_schema=POSE_SCHEMA)


def test_a_keypoint_no_box_owns_is_rejected_by_default(tmp_path):
    results = [_box("box1", "bolt", 0, 0, 5, 5), _keypoint("kp1", "head", 18, 18)]

    with pytest.raises(ValueError, match="not assigned to any box"):
        _convert(tmp_path, results, pose_schema=POSE_SCHEMA)


def test_a_keypoint_no_box_owns_may_be_dropped_or_kept(tmp_path):
    # Label Studio lets a keypoint stand alone, but a pose model trains on box-linked keypoints only.
    results = [_box("box1", "bolt", 0, 0, 5, 5), _keypoint("kp1", "head", 3, 3), _keypoint("kp2", "head", 18, 18)]

    dropped = _convert(tmp_path, results, pose_schema=POSE_SCHEMA, unlinked_keypoints="drop")
    assert len([a for a in _annotations(dropped) if a["type"] == "Keypoint"]) == 1

    kept = _convert(tmp_path, results, pose_schema=POSE_SCHEMA, unlinked_keypoints="keep")
    box = next(a for a in _annotations(kept) if a["type"] == "Box")
    keypoints = [a for a in _annotations(kept) if a["type"] == "Keypoint"]
    assert [a.get("bounding_box_id") for a in keypoints] == [box["id"], None]


def test_an_export_without_a_schema_converts_as_before(tmp_path):
    # A detection project has no layout to declare, and its labels stay whatever the export named.
    output_dir = _convert(tmp_path, [_box("box1", "bolt", 0, 0, WIDTH, HEIGHT)])

    assert not (output_dir / ".meta.json").exists()
    assert [annotation["type"] for annotation in _annotations(output_dir)] == ["Box"]


def test_containment_links_keypoints_in_the_exported_json(tmp_path):
    # Drawing a keypoint in Label Studio makes a top-level region, so an export states ownership only through
    # geometry. The link is resolved here rather than left to whatever reads the json next.
    export_path, image_dir = _write_export(tmp_path, ONE_BOLT)

    files, _, _ = get_annotations_from_json(str(export_path), str(image_dir), pose_schema=POSE_SCHEMA)

    box = next(a for a in files[0].annotations if a.type == AnnotationType.BOX)
    keypoints = [a for a in files[0].annotations if a.type == AnnotationType.KEYPOINT]
    assert [kp.bounding_box_id for kp in keypoints] == [box.id, box.id]


def test_a_keypoint_geometry_cannot_place_is_left_unlinked(tmp_path, caplog):
    # A Label Studio project may hold standalone or overlapping keypoints; neither stops the export being read.
    results = [
        _box("box1", "bolt", 0, 0, 5, 5),
        _box("box2", "bolt", 3, 3, 12, 12),
        _keypoint("kp1", "head", 4, 4),
        _keypoint("kp2", "head", 18, 18),
    ]
    export_path, image_dir = _write_export(tmp_path, results)

    with caplog.at_level(logging.WARNING):
        files, _, _ = get_annotations_from_json(str(export_path), str(image_dir), pose_schema=POSE_SCHEMA)

    keypoints = [a for a in files[0].annotations if a.type == AnnotationType.KEYPOINT]
    assert [kp.bounding_box_id for kp in keypoints] == [None, None]
    assert "contained by multiple boxes" in caplog.text
    assert "not assigned to any box" in caplog.text


def test_a_region_names_an_id_rather_than_what_the_annotator_read(tmp_path):
    # The schema calls the class "Hex bolt" and the keypoint "Head". Neither name reaches an annotation, which is
    # what lets either be renamed later without stranding the work already done.
    output_dir = _convert(tmp_path, ONE_BOLT, pose_schema=POSE_SCHEMA)

    assert {annotation["label_id"] for annotation in _annotations(output_dir)} == {"bolt", "head", "left-flange"}


def test_renaming_a_class_leaves_every_annotation_where_it_was(tmp_path):
    renamed = json.loads(json.dumps(POSE_SCHEMA))
    renamed["classes"]["bolt"]["name"] = "Carriage bolt"
    renamed["keypoints"]["head"]["name"] = "Cap"

    before = _annotations(_convert(tmp_path / "before", ONE_BOLT, pose_schema=POSE_SCHEMA))
    after = _annotations(_convert(tmp_path / "after", ONE_BOLT, pose_schema=renamed))

    assert [annotation["label_id"] for annotation in before] == [annotation["label_id"] for annotation in after]


def test_a_project_factory_did_not_generate_keeps_its_own_label_values(tmp_path):
    # Only Factory's own configuration puts labels under the alias prefix. A project built by hand names its
    # labels however the annotator did, and those values pass through as they stand.
    results = [_region("box1", "rectanglelabels", {"x": 0, "y": 0, "width": 100, "height": 100, "rectanglelabels": ["bolt"]})]
    output_dir = _convert(tmp_path, results)

    assert [annotation["label_id"] for annotation in _annotations(output_dir)] == ["bolt"]


def test_a_hand_built_project_resolves_display_names_to_declared_ids(tmp_path):
    results = [
        _region(
            "box1",
            "rectanglelabels",
            {"x": 0, "y": 0, "width": 100, "height": 100, "rectanglelabels": ["Hex bolt"]},
        ),
        _region("kp1", "keypointlabels", {"x": 50, "y": 25, "keypointlabels": ["Head"]}),
    ]
    output_dir = _convert(tmp_path, results, pose_schema=POSE_SCHEMA)

    assert [annotation["label_id"] for annotation in _annotations(output_dir)] == ["bolt", "head"]


def test_directory_input_only_skips_exact_reserved_output_names(tmp_path):
    export_path, image_dir = _write_export(tmp_path, ONE_BOLT)
    export_path.rename(tmp_path / "project-1-labels.json")

    files, _, _ = get_annotations_from_json(str(tmp_path), str(image_dir), pose_schema=POSE_SCHEMA)

    assert [file.path for file in files] == ["image.png"]


def test_a_factory_dataset_json_is_not_mistaken_for_a_label_studio_export(tmp_path):
    export_path, image_dir = _write_export(tmp_path, ONE_BOLT)
    export_path.write_text(json.dumps({"labels": [], "files": []}))

    with pytest.raises(ValueError, match="expected a top-level array of tasks, got dict"):
        get_annotations_from_json(str(export_path), str(image_dir))


def test_a_directory_with_only_reserved_outputs_has_no_export(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (tmp_path / "labels.json").write_text("[]")
    (tmp_path / "preds.json").write_text("[]")

    with pytest.raises(FileNotFoundError, match="No Label Studio JSON exports"):
        get_annotations_from_json(str(tmp_path), str(image_dir))


def test_scaffold_cli_applies_latest_ids_to_its_dataset_json(tmp_path, monkeypatch):
    results = [
        _region(
            "box1",
            "rectanglelabels",
            {"x": 0, "y": 0, "width": 100, "height": 100, "rectanglelabels": ["Hex bolt"]},
        ),
        _region("kp1", "keypointlabels", {"x": 50, "y": 25, "keypointlabels": ["Left eye"]}),
    ]
    export_path, image_dir = _write_export(tmp_path, results)
    output_path = tmp_path / "labels.json"
    schema_path = tmp_path / "schema.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "lst_to_json",
            "-i",
            str(export_path),
            "-imgs",
            str(image_dir),
            "-of",
            str(output_path),
            "--scaffold_schema",
            str(schema_path),
        ],
    )

    main()

    dataset = json.loads(output_path.read_text())
    schema = json.loads(schema_path.read_text())["annotationSchema"]
    class_id = derive_pose_id("Hex bolt")
    keypoint_id = derive_pose_id("Left eye")
    assert dataset["labels"][0]["id"] == class_id
    assert dataset["labels"][0]["keypoint_ids"] == [keypoint_id]
    assert dataset["keypoints"] == {keypoint_id: "Left eye"}
    assert {annotation["label_id"] for annotation in dataset["files"][0]["annotations"]} == {class_id, keypoint_id}
    assert schema["classes"][class_id]["keypointIds"] == [keypoint_id]
