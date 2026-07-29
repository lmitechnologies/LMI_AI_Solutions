import json

import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.representations import Dataset
from lmi_utils.label_utils.coco_to_json import convert_coco_to_json
from lmi_utils.label_utils.json_to_factory import convert_json_to_factory

# One fixed point and one swapped pair, so a supplied flip has something to mirror.
KEYPOINTS = ["head", "left-flange", "right-flange"]
# Instance keypoints as COCO writes them: x, y, visibility, with 0 marking a slot this instance does not observe.
OBSERVED = [10, 4, 2, 4, 16, 1, 0, 0, 0]


def _coco(tmp_path, keypoints=KEYPOINTS, skeleton=None, instance_keypoints=OBSERVED, category_name="bolt"):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "image.png"), np.zeros((20, 20, 3), dtype=np.uint8))

    category = {"id": 1, "name": category_name, "supercategory": ""}
    if keypoints:
        category["keypoints"] = keypoints
    if skeleton:
        category["skeleton"] = skeleton
    annotation = {"id": 7, "image_id": 1, "category_id": 1, "segmentation": [], "area": 400, "bbox": [0, 0, 20, 20], "iscrowd": 0}
    if instance_keypoints:
        annotation["keypoints"] = instance_keypoints
        annotation["num_keypoints"] = sum(1 for index in range(2, len(instance_keypoints), 3) if instance_keypoints[index])

    coco_file = tmp_path / "labels.json"
    coco_file.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 20, "height": 20, "file_name": "image.png"}],
                "annotations": [annotation],
                "categories": [category],
            }
        )
    )
    return coco_file, image_dir


def _convert(tmp_path, coco_file, image_dir, **kwargs):
    """Run the whole pipeline: COCO to the dataset json, then the json to a Factory dataset directory.

    The json is reloaded from disk rather than passed along in memory, so every assertion below also covers the
    declaration surviving the intermediate file.
    """
    json_path = image_dir / "labels.json"
    convert_coco_to_json(coco_file, json_path, **kwargs)
    output_dir = tmp_path / "factory"
    convert_json_to_factory(Dataset.load(str(json_path)), image_dir, output_dir)
    return output_dir


def _schema(output_dir):
    return json.loads((output_dir / ".meta.json").read_text())["annotationSchema"]


def _annotations(output_dir):
    return json.loads((output_dir / "image.label.json").read_text())["annotations"]


def test_categories_become_declared_pose_classes(tmp_path):
    coco_file, image_dir = _coco(tmp_path, skeleton=[[1, 2], [1, 3]])
    output_dir = _convert(tmp_path, coco_file, image_dir)

    assert _schema(output_dir) == {
        "type": "Pose",
        "version": 1,
        "coordinateDimensions": 3,
        "keypoints": {name: {"name": name} for name in KEYPOINTS},
        "classes": {
            # COCO carries no flip symmetry, and a skeleton it writes one-based reaches Factory zero-based.
            "bolt": {"name": "bolt", "keypointIds": KEYPOINTS, "horizontalFlipPairs": None, "skeleton": [[0, 1], [0, 2]]},
        },
    }


def test_a_zero_based_skeleton_is_kept_when_declared(tmp_path):
    coco_file, image_dir = _coco(tmp_path, skeleton=[[0, 1]])
    output_dir = _convert(tmp_path, coco_file, image_dir, skeleton_base=0)

    assert _schema(output_dir)["classes"]["bolt"]["skeleton"] == [[0, 1]]


def test_a_supplied_flip_enables_mirroring(tmp_path):
    coco_file, image_dir = _coco(tmp_path)
    flip_map = tmp_path / "flip.json"
    flip_map.write_text(json.dumps({"bolt": [["left-flange", "right-flange"]]}))
    output_dir = _convert(tmp_path, coco_file, image_dir, flip_map_file=flip_map)

    assert _schema(output_dir)["classes"]["bolt"]["horizontalFlipPairs"] == [["left-flange", "right-flange"]]


def test_category_names_may_be_mapped_to_factory_class_ids(tmp_path):
    coco_file, image_dir = _coco(tmp_path, category_name="Bolt Assembly")
    class_map = tmp_path / "classes.json"
    class_map.write_text(json.dumps({"Bolt Assembly": "bolt"}))
    output_dir = _convert(tmp_path, coco_file, image_dir, class_map_file=class_map)

    assert list(_schema(output_dir)["classes"]) == ["bolt"]
    assert {annotation["label_id"] for annotation in _annotations(output_dir) if annotation["type"] == "Box"} == {"bolt"}


def test_observed_keypoints_are_linked_and_unobserved_slots_are_dropped(tmp_path):
    coco_file, image_dir = _coco(tmp_path)
    output_dir = _convert(tmp_path, coco_file, image_dir)

    keypoints = [annotation for annotation in _annotations(output_dir) if annotation["type"] == "Keypoint"]
    assert [annotation["label_id"] for annotation in keypoints] == ["head", "left-flange"]
    assert {annotation["bounding_box_id"] for annotation in keypoints} == {"7"}
    assert keypoints[0]["value"] == {"x": 10.0, "y": 4.0, "visibility": 2}
    # The slot the instance does not observe stays declared in the schema rather than becoming a point at the origin.
    assert _schema(output_dir)["classes"]["bolt"]["keypointIds"] == KEYPOINTS


def test_an_instance_disagreeing_with_its_category_layout_is_rejected(tmp_path):
    coco_file, image_dir = _coco(tmp_path, instance_keypoints=[10, 4, 2, 4, 16, 1])

    with pytest.raises(ValueError, match="has 2 keypoints for the 3 declared"):
        _convert(tmp_path, coco_file, image_dir)


def test_a_flip_for_a_class_without_keypoints_is_rejected(tmp_path):
    coco_file, image_dir = _coco(tmp_path, keypoints=None, instance_keypoints=None)
    flip_map = tmp_path / "flip.json"
    flip_map.write_text(json.dumps({"bolt": [["head", "left-flange"]]}))

    with pytest.raises(ValueError, match="declares no keypoints"):
        _convert(tmp_path, coco_file, image_dir, flip_map_file=flip_map)


def test_a_detection_only_dataset_converts_without_a_schema(tmp_path):
    coco_file, image_dir = _coco(tmp_path, keypoints=None, instance_keypoints=None)
    output_dir = _convert(tmp_path, coco_file, image_dir)

    assert not (output_dir / ".meta.json").exists()
    assert [annotation["type"] for annotation in _annotations(output_dir)] == ["Box"]


def test_segmentation_converts_instances_of_keypointless_classes_to_polygons(tmp_path):
    coco_file, image_dir = _coco(tmp_path, keypoints=None, instance_keypoints=None)
    coco = json.loads(coco_file.read_text())
    coco["annotations"][0]["segmentation"] = [[0, 0, 10, 0, 10, 10]]
    coco_file.write_text(json.dumps(coco))
    output_dir = _convert(tmp_path, coco_file, image_dir, segmentation=True)

    (polygon,) = _annotations(output_dir)
    assert polygon["type"] == "Polygon"
    assert polygon["value"]["points"] == [[0, 0], [10, 0], [10, 10]]
