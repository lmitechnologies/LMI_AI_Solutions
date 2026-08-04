import os

import cv2
import numpy as np

from lmi_utils.dataset_utils.representations import (
    Box,
    BoxAnnotation,
    Dataset,
    FileAnnotations,
    Label,
    Mask,
    MaskAnnotation,
)
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


def build_mask_dataset(blobs):
    """One file holding one mask annotation, with a filled rectangle per (y0, y1, x0, x1) blob."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    for y0, y1, x0, x1 in blobs:
        mask[y0:y1, x0:x1] = 1
    file = FileAnnotations(
        id="0",
        path="img0.png",
        height=100,
        width=200,
        annotations=[MaskAnnotation(id="a0", label_id="defect", confidence=1.0, value=Mask(mask=mask))],
        predictions=[],
    )
    return Dataset(labels=[Label(id="defect")], files=[file])


def test_disconnected_mask_is_one_annotation_by_default():
    _, coco, _, _ = create_coco_dataset(build_mask_dataset([(10, 30, 10, 30), (60, 80, 65, 85)]))
    assert len(coco.annotations) == 1
    # the union of both regions, so the box covers the gap between them
    assert [round(v) for v in coco.annotations[0].bbox] == [10, 10, 74, 69]


def test_split_regions_gives_each_region_its_own_annotation():
    _, coco, _, _ = create_coco_dataset(build_mask_dataset([(10, 30, 10, 30), (60, 80, 65, 85)]), split_regions=True)
    assert len(coco.annotations) == 2
    boxes = sorted([round(v) for v in ann.bbox] for ann in coco.annotations)
    assert boxes == [[10, 10, 19, 19], [65, 60, 19, 19]]


def test_connected_mask_is_one_annotation_either_way():
    for split_regions in (False, True):
        _, coco, _, _ = create_coco_dataset(build_mask_dataset([(10, 30, 10, 30)]), split_regions=split_regions)
        assert len(coco.annotations) == 1
        assert [round(v) for v in coco.annotations[0].bbox] == [10, 10, 19, 19]


def test_split_regions_carry_their_own_segmentation():
    _, coco, _, _ = create_coco_dataset(build_mask_dataset([(10, 30, 10, 30), (60, 80, 65, 85)]), split_regions=True)
    for annotation in coco.annotations:
        xs = annotation.segmentation[0][0::2]
        ys = annotation.segmentation[0][1::2]
        # each outline stays inside its own box rather than spanning both regions
        assert min(xs) == annotation.bbox[0] and min(ys) == annotation.bbox[1]
        assert max(xs) - min(xs) == annotation.bbox[2]
        assert max(ys) - min(ys) == annotation.bbox[3]
