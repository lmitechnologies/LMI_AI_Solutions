import argparse
import logging
import os

from lmi_utils.dataset_utils.coco_dataset import CocoAnnotation, CocoCategory, CocoDataset, CocoImage
from lmi_utils.dataset_utils.file_utils import copy_images_in_folder, load_and_update
from lmi_utils.dataset_utils.representations import BoxAnnotation, Dataset, MaskAnnotation, PolygonAnnotation

logger = logging.getLogger(__name__)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_train_json", default="labels.json", help="[optional] the path of a json file for train")
    ap.add_argument("--path_val_json", default="labels.json", help="[optional] the path of a json file for val")
    ap.add_argument("--path_out", "-o", required=True, help="the output path for dataset")
    ap.add_argument("--path_train_imgs", "-ti", required=True, help="the output path for train images")
    ap.add_argument("--path_val_imgs", "-vi", required=False, help="the output path for val images")
    ap.add_argument("--target_classes", default="all", help="[optional] the comma separated target classes, default=all")
    ap.add_argument("--bg", action="store_true", help="save images with no labels, where yolo models treat them as background")
    ap.add_argument("--merge_box", action="store_true", help="merge multiple instances of same class boxes into one. Brush labels only!")
    ap.add_argument("--idx0", action="store_true", help="start index from 0 instead of 1")

    args = vars(ap.parse_args())
    return args


def get_coco_annotation(annotation, **kwargs):
    """Convert a dataset annotation to COCO format."""
    if isinstance(annotation, MaskAnnotation):
        bbox = annotation.value.to_box(**kwargs)
        return annotation.value.to_coco(**kwargs), bbox.to_coco(**kwargs)
    elif isinstance(annotation, PolygonAnnotation):
        bbox = annotation.value.to_box(**kwargs)
        return annotation.value.to_coco(**kwargs), bbox.to_coco(**kwargs)
    elif isinstance(annotation, BoxAnnotation):
        poly = annotation.value.to_polygon(**kwargs)
        segm = poly.to_coco(**kwargs)
        return segm, annotation.value.to_coco(**kwargs)
    else:
        raise ValueError(f"Unsupported annotation type: {type(annotation)}")


def create_coco_dataset(dataset: Dataset, is_crowd: bool = False, target_classes: list = None, **kwargs) -> CocoDataset:
    """
    Create a COCO dataset from a given dataset and image path.

    Args:
        dataset (Dataset): The dataset to convert.
        path_imgs (str): The path to the images directory.

    Returns:
        CocoDataset: A COCO dataset with updated file dimensions.
    """
    if target_classes is None:
        target_classes = [c.id for c in dataset.labels]
    coco_dataset = CocoDataset()
    file_id_map = {}
    annotation_id = 0
    # add categories (only target classes)
    labels = [label for label in dataset.labels if label.id in target_classes]
    for label_id, label in enumerate(labels):
        coco_dataset.add_category(
            CocoCategory(
                id=label_id + (0 if kwargs.get("idx0", False) is True else 1),
                name=label.id,
                supercategory="",
            )
        )

    fnames = set()
    # add images and annotations
    for file_id, file in enumerate(dataset.files):
        filtered_annotations = [ann for ann in file.annotations if ann.label_id in target_classes]
        file_id_map[os.path.basename(file.path)] = file.id
        if len(filtered_annotations) == 0:
            logger.warning(f"Skipping file {file.path} as it has no annotations for target classes")
            continue

        logger.info(f"Processing file {file_id + 1}/{len(dataset.files)}: {file.path}")
        image_id = file_id + 1
        # update the image name if id is not part of the image
        out_name = os.path.basename(file.path)
        if f"id{file.id}_" not in out_name:
            out_name = f"id{file.id}_{out_name}"
        coco_dataset.add_image(
            CocoImage(
                id=image_id,
                file_name=out_name,
                height=file.height,
                width=file.width,
            )
        )
        added_annotations = 0
        for annotation in filtered_annotations:
            try:
                # both segmentation and bbox are required for COCO format
                segmentation, bbox = get_coco_annotation(annotation, h=file.height, w=file.width, **kwargs)
                if bbox[2] <= 0 or bbox[3] <= 0:
                    logger.warning(f"Skipping annotation {annotation.id} for file {file.path} as bbox is invalid: {bbox}")
                    continue
                annotation_id += 1
                coco_dataset.add_annotation(
                    CocoAnnotation(
                        id=annotation_id,
                        image_id=image_id,
                        category_id=coco_dataset.get_category_by_name(annotation.label_id).id,
                        segmentation=segmentation,
                        bbox=bbox,
                        area=0,
                        iscrowd=is_crowd,
                    )
                )
                added_annotations += 1
            except Exception as e:
                logger.error(f"Error processing  (annotation could be invalid) {annotation.id} for file {file.path}: {e}")
                continue
        if added_annotations > 0:
            fnames.add(os.path.basename(file.path))
        else:
            # remove the image if no annotations were added
            logger.warning(f"No valid annotations found for file {file.path}, removing image from dataset")
            for coco_image in coco_dataset.images:
                if coco_image.file_name == out_name:
                    coco_dataset.images.remove(coco_image)
                    # removing annotations for this image
                    break
    dataset.files = [file for file in dataset.files if os.path.basename(file.path) in fnames]

    return dataset, coco_dataset, fnames, file_id_map


def convert_to_json(args):
    path_train_imgs = args["path_train_imgs"]
    path_val_imgs = args["path_val_imgs"] if args.get("path_val_imgs") else path_train_imgs
    path_train_json = (
        args["path_train_json"] if args["path_train_json"] != "labels.json" else os.path.join(path_train_imgs, args["path_train_json"])
    )
    path_val_json = args["path_val_json"] if args["path_val_json"] != "labels.json" else os.path.join(path_val_imgs, args["path_val_json"])
    path_out = args["path_out"]
    background = args.get("bg", False)
    if background:
        logger.warning("Background is not supported for COCO format at the moment")
    merge_box = args.get("merge_box", False)

    if not os.path.exists(path_train_json):
        raise FileNotFoundError(f"Train annotations file {path_train_json} does not exist")

    if not os.path.exists(path_val_json):
        logger.warning(f"Validation annotations file {path_val_json} does not exist, using train annotations instead")
        path_val_json = path_train_json
        path_val_imgs = path_train_imgs

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    logger.info(f"Train images path: {path_train_imgs}")
    logger.info(f"Validation images path: {path_val_imgs}")
    logger.info(f"Train annotations path: {path_train_json}")
    logger.info(f"Validation annotations path: {path_val_json}")
    use_train_for_val = path_train_imgs == path_val_imgs and path_train_json == path_val_json
    logger.info(f"Using train dataset for validation: {use_train_for_val}")
    # load datasets
    train_dataset = load_and_update(annotations_path=path_train_json, path_imgs=path_train_imgs)
    if use_train_for_val:
        logger.info("Using train dataset for validation")
        val_dataset = train_dataset
    else:
        val_dataset = load_and_update(annotations_path=path_val_json, path_imgs=path_val_imgs)

    # filter target classes
    target_classes = args["target_classes"]
    if target_classes != "all":
        target_classes = [c.strip() for c in target_classes.split(",")]
    else:
        target_classes = [c.id for c in train_dataset.labels]
    logger.info(f"Target classes: {target_classes}")

    # create coco datasets
    train_ais_dataset, coco_train_dataset, train_files, train_file_id_map = create_coco_dataset(
        train_dataset, is_crowd=False, target_classes=target_classes, merge_boxes=merge_box, idx0=args.get("idx0", False)
    )
    if use_train_for_val:
        logger.info("Creating validation dataset from train dataset")
        val_ais_dataset = train_ais_dataset
        coco_val_dataset = coco_train_dataset
        val_files = train_files
        val_file_id_map = train_file_id_map
    else:
        logger.info("Creating validation dataset from val dataset")
        val_ais_dataset, coco_val_dataset, val_files, val_file_id_map = create_coco_dataset(
            val_dataset, is_crowd=False, target_classes=target_classes, merge_boxes=merge_box, idx0=args.get("idx0", False)
        )

    # save ais datasets
    os.makedirs(os.path.join(path_out, "train"), exist_ok=True)
    os.makedirs(os.path.join(path_out, "valid"), exist_ok=True)
    train_ais_dataset.save(os.path.join(path_out, "train", "ais.train.dataset.json"))
    val_ais_dataset.save(os.path.join(path_out, "valid", "ais.val.dataset.json"))

    # save coco json annotations
    coco_train_dataset.save_to_json(file_path=os.path.join(path_out, "train", "_annotations.coco.json"))
    coco_val_dataset.save_to_json(file_path=os.path.join(path_out, "valid", "_annotations.coco.json"))

    # save datasets
    copy_images_in_folder(path_train_imgs, os.path.join(path_out, "train"), train_files, train_file_id_map)
    copy_images_in_folder(path_val_imgs, os.path.join(path_out, "valid"), val_files, val_file_id_map)


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    convert_to_json(args)


if __name__ == "__main__":
    main()
