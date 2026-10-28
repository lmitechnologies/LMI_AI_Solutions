import json
import os

import cv2
import numpy as np

from lmi_utils.dataset_utils.representations import Box, BoxAnnotation, Dataset, FileAnnotations, Label
from lmi_utils.label_utils.json_to_coco import convert_to_json
from lmi_utils.label_utils.merge_coco_json import merge_datasets


def make_coco_dir(root, label):
    """Run json_to_coco on two images holding one `label` box each; return the train COCO json path."""
    imgs_dir = root / "imgs"
    imgs_dir.mkdir(parents=True)
    files = []
    for i in range(2):
        cv2.imwrite(str(imgs_dir / f"img{i}.png"), np.zeros((100, 200, 3), dtype=np.uint8))
        box = Box(x_min=10, y_min=10, x_max=50, y_max=50, angle=0)
        annot = BoxAnnotation(id=f"a{i}", label_id=label, confidence=1.0, value=box)
        files.append(FileAnnotations(id=str(i), path=f"img{i}.png", height=100, width=200, annotations=[annot], predictions=[]))
    labels_path = root / "labels.json"
    Dataset(labels=[Label(id=label)], files=files).save(str(labels_path))
    convert_to_json(
        {
            "path_train_json": str(labels_path),
            "path_val_json": str(labels_path),
            "path_train_imgs": str(imgs_dir),
            "path_val_imgs": str(imgs_dir),
            "path_out": str(root / "coco"),
            "target_classes": "all",
        }
    )
    return str(root / "coco" / "train" / "_annotations.coco.json")


def test_merge_json_to_coco_outputs(tmp_path):
    a = make_coco_dir(tmp_path / "a", "scratch")
    b = make_coco_dir(tmp_path / "b", "dent")
    out = tmp_path / "merged"

    merge_datasets([a, b], str(out))

    with open(out / "_annotations.coco.json") as f:
        merged = json.load(f)
    names = [img["file_name"] for img in merged["images"]]
    assert names == ["id1_img0.png", "id2_img1.png", "id3_img0.png", "id4_img1.png"]
    assert {f for f in os.listdir(out) if f.endswith(".png")} == set(names)
    # numbered from 1, as json_to_coco does by default
    cat_ids = {c["name"]: c["id"] for c in merged["categories"]}
    assert cat_ids == {"scratch": 1, "dent": 2}
    img_cat = {ann["image_id"]: ann["category_id"] for ann in merged["annotations"]}
    assert img_cat == {1: cat_ids["scratch"], 2: cat_ids["scratch"], 3: cat_ids["dent"], 4: cat_ids["dent"]}
