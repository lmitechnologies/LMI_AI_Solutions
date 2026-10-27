import argparse
import logging
import os
import re
import shutil
from typing import Dict, List

from lmi_utils.dataset_utils.coco_dataset import CocoAnnotation, CocoCategory, CocoDataset, CocoImage, CocoLicense

logger = logging.getLogger(__name__)


def _image_dir(json_path: str) -> str:
    # csv_to_coco puts images in images/; json_to_coco puts them beside the json.
    json_dir = os.path.dirname(json_path)
    images_dir = os.path.join(json_dir, "images")
    return images_dir if os.path.isdir(images_dir) else json_dir


def merge_datasets(input_paths: List[str], output_path: str, indx_start: int = 1) -> None:
    """
    Merges multiple COCO JSON files into one dataset in the json_to_coco layout:
    <output_path>/_annotations.coco.json with the images beside it.
    Handles ID conflicts by re-indexing images, annotations, and categories; category ids start at indx_start.
    """
    merged = CocoDataset()
    merged.info.description = "Merged Dataset"
    merged.info.contributor = "Merged via Script"

    # Global Maps to unify data across datasets
    # Map Name -> New ID
    global_category_map: Dict[str, int] = {}

    # Map (Name, URL) -> New ID to avoid duplicate licenses
    global_license_map: Dict[tuple, int] = {}

    # Global counters for new unique IDs
    # We start at 1 to stay safe with COCO conventions
    current_cat_id = indx_start
    current_img_id = 1
    current_ann_id = 1
    current_lic_id = 1

    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Starting merge of {len(input_paths)} datasets...")

    for path in input_paths:
        logger.info(f"Processing: {path}")
        ds = CocoDataset.load_from_json(path)
        image_dir = _image_dir(path)

        # --- 1. Merge Licenses ---
        local_lic_map = {}  # old_id -> new_id
        for lic in ds.licenses:
            key = (lic.name, lic.url)
            if key not in global_license_map:
                new_lic = CocoLicense(id=current_lic_id, name=lic.name, url=lic.url)
                merged.add_license(new_lic)
                global_license_map[key] = current_lic_id
                current_lic_id += 1
            local_lic_map[lic.id] = global_license_map[key]

        # --- 2. Merge Categories (Unify by Name) ---
        local_cat_map = {}  # old_id -> new_id
        for cat in ds.categories:
            if cat.name not in global_category_map:
                new_cat = CocoCategory(id=current_cat_id, name=cat.name, supercategory=cat.supercategory)
                merged.add_category(new_cat)
                global_category_map[cat.name] = current_cat_id
                current_cat_id += 1
            local_cat_map[cat.id] = global_category_map[cat.name]

        # --- 3. Merge Images (Remap IDs) ---
        local_img_map = {}  # old_id -> new_id
        for img in ds.images:
            # Handle license mapping (default to 0 if not found)
            new_lic_id = local_lic_map.get(img.license, 0)
            new_name = f"id{current_img_id}_" + re.sub(r"^id\d+_", "", img.file_name)
            shutil.copy(os.path.join(image_dir, img.file_name), os.path.join(output_path, new_name))

            new_img = CocoImage(
                id=current_img_id,
                width=img.width,
                height=img.height,
                file_name=new_name,
                license=new_lic_id,
                flickr_url=img.flickr_url,
                coco_url=img.coco_url,
                date_captured=img.date_captured,
            )
            merged.add_image(new_img)
            local_img_map[img.id] = current_img_id
            current_img_id += 1

        # --- 4. Merge Annotations (Remap IDs and References) ---
        for ann in ds.annotations:
            # Sanity Check: Ensure referenced image/category actually exist in this dataset
            if ann.image_id not in local_img_map:
                logger.warning(f"Skipping annotation {ann.id} (Image {ann.image_id} not found)")
                continue
            if ann.category_id not in local_cat_map:
                logger.warning(f"Skipping annotation {ann.id} (Category {ann.category_id} not found)")
                continue

            new_ann = CocoAnnotation(
                id=current_ann_id,
                image_id=local_img_map[ann.image_id],
                category_id=local_cat_map[ann.category_id],
                segmentation=ann.segmentation,
                area=ann.area,
                bbox=ann.bbox,
                iscrowd=ann.iscrowd,
            )
            merged.add_annotation(new_ann)
            current_ann_id += 1

    # Save
    logger.info(f"Saving merged dataset to {output_path}...")
    merged.save_to_json(os.path.join(output_path, "_annotations.coco.json"))

    # Print stats
    stats = merged.get_statistics()
    logger.info("Merge Complete.")
    logger.info(f"Total Images: {stats['num_images']}")
    logger.info(f"Total Annotations: {stats['num_annotations']}")
    logger.info(f"Total Categories: {stats['num_categories']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Merge multiple COCO datasets.")
    parser.add_argument("-i", "--inputs", nargs="+", required=True, help="List of input COCO JSON files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output directory for the merged dataset")
    parser.add_argument("--idx0", action="store_true", help="start category ids from 0 instead of 1, as json_to_coco --idx0")

    args = parser.parse_args()
    merge_datasets(args.inputs, args.output, indx_start=0 if args.idx0 else 1)
