import argparse
import logging
import os
import shutil
from typing import Dict, List

from lmi_utils.dataset_utils.coco_dataset import CocoAnnotation, CocoCategory, CocoDataset, CocoImage, CocoLicense

logger = logging.getLogger(__name__)


def merge_datasets(input_paths: List[str], output_path: str, indx_start: int = 0) -> None:
    """
    Merges multiple COCO JSON files into a single dataset.
    Handles ID conflicts by re-indexing images, annotations, and categories.
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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create a directory for images output if it doesn't exist
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))

    logger.info(f"Starting merge of {len(input_paths)} datasets...")

    for path in input_paths:
        logger.info(f"Processing: {path}")
        try:
            ds = CocoDataset.load_from_json(path)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            continue

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
            full_image_path = os.path.join(os.path.dirname(path), "images", img.file_name)
            new_name = img.file_name.replace(img.file_name, f"id{current_img_id}_" + img.file_name)
            try:
                shutil.copy(full_image_path, os.path.join(output_path, "images", new_name))
            except Exception as e:
                logger.error(f"Error copying image {full_image_path}: {e}")
                exit(1)

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
    merged.save_to_json(os.path.join(output_path, "merged.json"))

    # Print stats
    stats = merged.get_statistics()
    logger.info("Merge Complete.")
    logger.info(f"Total Images: {stats['num_images']}")
    logger.info(f"Total Annotations: {stats['num_annotations']}")
    logger.info(f"Total Categories: {stats['num_categories']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Merge multiple COCO datasets.")
    parser.add_argument("-i", "--inputs", nargs="+", help="List of input COCO JSON files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output path for the merged JSON file")

    args = parser.parse_args()
    merge_datasets(args.inputs, args.output)
