"""Convert a classification dataset laid out as one folder per class to a Factory dataset."""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from lmi_utils.dataset_utils.file_utils import IMG_FORMATS, update_file_dimensions
from lmi_utils.dataset_utils.representations import Dataset, FileAnnotations
from lmi_utils.label_utils.json_to_factory import DATASET_META_FILE, FACTORY_IMAGE_SUFFIXES, check_name, raise_problems

logger = logging.getLogger(__name__)

# The top-level folders of an Ultralytics classification dataset, each holding the class folders
SPLITS = {"train", "val", "test"}
IMAGE_SUFFIXES = {f".{ext}" for ext in IMG_FORMATS}
CLASSIFICATION = "Classification"


def item_key(relative_path: Path) -> str:
    """The item Factory makes of a file: its path with every extension stripped (stem in the SDK's path-utils)."""
    name = relative_path.name
    dot = name.find(".", 1 if name.startswith(".") else 0)
    return (relative_path.parent / (name[:dot] if dot >= 0 else name)).as_posix()


def _subfolders(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def class_folders(input_dir: Path) -> Dict[str, List[Path]]:
    """Each class name and its folders: the top-level folders, or the folders in each split when the top level is splits."""
    top = _subfolders(input_dir)
    if top and all(folder.name in SPLITS for folder in top):
        folders = [folder for split in top for folder in _subfolders(split)]
    else:
        folders = top
    classes = {}
    for folder in folders:
        classes.setdefault(folder.name, []).append(folder)
    return dict(sorted(classes.items()))


def _images(folder: Path) -> List[Path]:
    """The images in a folder at any depth, leaving out hidden files and folders."""
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )


def convert_class_folders_to_factory(
    input_dir: Path, output_dir: Path, link: bool = False, per_class: Optional[int] = None, seed: int = 0
) -> Dict[str, int]:
    """Write a Factory dataset with one Classification annotation per image, named after the image's class folder.

    Args:
        input_dir: the class folders, or train/val/test folders holding them; images may sit at any depth in a class folder.
        output_dir: the Factory dataset folder to create; it must not exist or be empty.
        link: hard-link the images rather than copy them; output_dir must be on the same filesystem.
        per_class: keep a random sample of at most this many images of each class.
        seed: the sample's random seed.

    Returns:
        The number of images written for each class.

    Raises:
        ValueError: listing every problem found, before anything is written
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} is not empty")
    classes = class_folders(input_dir)
    if not classes:
        raise ValueError(f"{input_dir} has no class folders")

    problems = []
    images = {}
    for name, folders in classes.items():
        check_name(name, f"class '{name}'", problems)
        found = [image for folder in folders for image in _images(folder)]
        problems += [
            f"{image}: Factory skips {image.suffix} images" for image in found if image.suffix.lower() not in FACTORY_IMAGE_SUFFIXES
        ]
        if not found:
            logger.warning(f"Leaving out class '{name}': its folders hold no images")
            continue
        if per_class is not None:
            found = sorted(random.Random(seed).sample(found, min(per_class, len(found))))
        images[name] = found

    class_dirs = {folder for folders in classes.values() for folder in folders}
    problems += [f"{image}: not in a class folder" for image in _images(input_dir) if not class_dirs & set(image.parents)]

    owners = {}
    for image in (image for found in images.values() for image in found):
        key = item_key(image.relative_to(input_dir))
        if key in owners:
            problems.append(f"{owners[key]} and {image} would be one Factory item, '{key}'")
        owners[key] = image
    raise_problems(problems, "Cannot convert these images")

    files = [
        FileAnnotations(id=str(i), path=str(image.relative_to(input_dir)), height=None, width=None)
        for i, image in enumerate(owners.values())
    ]
    update_file_dimensions(Dataset(labels=[], files=files), str(input_dir))
    sizes = {file.path: (file.width, file.height) for file in files}

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, found in images.items():
        for image in found:
            relative = image.relative_to(input_dir)
            out_image = output_dir / relative
            out_image.parent.mkdir(parents=True, exist_ok=True)
            if link:
                os.link(image, out_image)
            else:
                shutil.copy(image, out_image)
            width, height = sizes[str(relative)]
            annotations = {"width": width, "height": height, "annotations": [{"id": "1", "type": CLASSIFICATION, "label_id": name}]}
            (output_dir / f"{item_key(relative)}.label.json").write_text(json.dumps(annotations, indent=4))

    label_dictionary = {name: {"id": name, "annotationType": CLASSIFICATION, "order": i} for i, name in enumerate(images)}
    (output_dir / DATASET_META_FILE).write_text(json.dumps({"labelDictionary": label_dictionary}, indent=4))
    counts = {name: len(found) for name, found in images.items()}
    logger.info(f"Converted {sum(counts.values())} images in {len(counts)} classes to {output_dir}: {counts}")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Convert a classification dataset laid out as one folder per class to a Factory dataset.")
    parser.add_argument(
        "--input_dir", "-i", type=Path, required=True, help="Folder of class folders, or of train/val/test folders holding them."
    )
    parser.add_argument("--output_dir", "-o", type=Path, required=True, help="Path to the output Factory dataset directory.")
    parser.add_argument("--link", action="store_true", help="hard-link the images rather than copy them (same filesystem only)")
    parser.add_argument("--per_class", type=int, default=None, help="keep a random sample of at most this many images of each class")
    parser.add_argument("--seed", type=int, default=0, help="random seed for --per_class")
    args = parser.parse_args()
    convert_class_folders_to_factory(args.input_dir, args.output_dir, args.link, args.per_class, args.seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
