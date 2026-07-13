import os

import cv2
import numpy as np

from lmi_utils.dataset_utils.representations import Box, BoxAnnotation, Dataset, FileAnnotations, Label
from lmi_utils.label_utils.json_to_coco import convert_to_json, create_coco_dataset


def build_dataset():
    """Two annotated files and one background (annotation-free) file."""
    files = [
        FileAnnotations(
            id=str(i),
            path=f"img{i}.png",
            height=100,
            width=200,
            annotations=[
                BoxAnnotation(
                    id=f"a{i}",
                    label_id="defect",
                    confidence=1.0,
                    value=Box(x_min=10, y_min=10, x_max=50, y_max=50, angle=0),
                )
            ],
            predictions=[],
        )
        for i in range(2)
    ]
    files.append(FileAnnotations(id="2", path="img2.png", height=100, width=200, annotations=[], predictions=[]))
    return Dataset(labels=[Label(id="defect")], files=files)


def test_background_files_skipped_by_default():
    dataset, coco, fnames, _ = create_coco_dataset(build_dataset())
    assert fnames == {"img0.png", "img1.png"}
    assert {img.file_name for img in coco.images} == {"id0_img0.png", "id1_img1.png"}
    assert len(coco.annotations) == 2
    assert {f.path for f in dataset.files} == {"img0.png", "img1.png"}


def test_background_files_kept_with_background():
    dataset, coco, fnames, _ = create_coco_dataset(build_dataset(), background=True)
    assert fnames == {"img0.png", "img1.png", "img2.png"}
    assert {img.file_name for img in coco.images} == {"id0_img0.png", "id1_img1.png", "id2_img2.png"}
    # the background image contributes no annotations
    assert len(coco.annotations) == 2
    annotated_image_ids = {ann.image_id for ann in coco.annotations}
    bg_image = next(img for img in coco.images if img.file_name == "id2_img2.png")
    assert bg_image.id not in annotated_image_ids
    assert {f.path for f in dataset.files} == {"img0.png", "img1.png", "img2.png"}


def test_convert_to_json_bg_copies_background_images(tmp_path):
    imgs_dir = tmp_path / "imgs"
    imgs_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(imgs_dir / f"img{i}.png"), np.zeros((100, 200, 3), dtype=np.uint8))
    annotations_path = tmp_path / "annotations.json"
    build_dataset().save(str(annotations_path))

    out_dir = tmp_path / "out"
    convert_to_json(
        {
            "path_train_json": str(annotations_path),
            "path_val_json": str(annotations_path),
            "path_train_imgs": str(imgs_dir),
            "path_val_imgs": str(imgs_dir),
            "path_out": str(out_dir),
            "target_classes": "all",
            "bg": True,
        }
    )

    for split in ("train", "valid"):
        copied = {f for f in os.listdir(out_dir / split) if f.endswith(".png")}
        assert copied == {"id0_img0.png", "id1_img1.png", "id2_img2.png"}
