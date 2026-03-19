import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2

from lmi_utils.dataset_utils.representations import Annotation, Dataset

logger = logging.getLogger(__name__)

"""
factory dataset format:
for each image:
{
    "id": "2azypeya",
    "width": 640,
    "height": 480,
    "annotations": [
        {
            "id": "abcd1234",
            "type": "Polygon",
            "label_id": "bicycle",
            "value": {
                "points": [
                    [100, 100],
                    [200, 100],
                    [200, 200],
                    [100, 200]
                ]
            }
        }
    ],
    "predictions": [
        {
            "id": "8oe97l4x",
            "type": "Box",
            "label_id": "bicycle",
            "value": {
                "x_min": 81,
                "y_min": 7,
                "x_max": 109,
                "y_max": 133,
                "angle": 0
            }
        },
    ]

"""


def to_dict(annot: Annotation):
    """convert it to a dictionary."""
    return json.loads(annot.to_json())


def convert_json_to_factory(dataset: Dataset, image_dir: Path, output_dir: Path):
    """
    Convert a JSON file to a Factory dataset.

    Args:
        dataset (Dataset): Dataset object representing the input JSON file.
        image_dir (Path): Path to the directory containing the images.
        output_dir (Path): Path to the output Factory dataset directory.
    """
    fname_to_list = {}
    for file in dataset.files:
        image_path = image_dir / file.path
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image file {image_path}.")

        h0, w0 = image.shape[:2]
        annots = {
            "width": w0,
            "height": h0,
            "annotations": [],
            "predictions": [],
        }
        for annotation in file.annotations:
            annots["annotations"].append(to_dict(annotation))
        for prediction in file.predictions:
            annots["predictions"].append(to_dict(prediction))

        rel_path = image_path.relative_to(image_dir)
        outname = rel_path.with_suffix(".label.json")
        fname_to_list[outname] = annots

    # write annotation files
    for fname, annots in fname_to_list.items():
        out_path = output_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(annots, f, indent=4)

    # copy images to output_dir
    for file in dataset.files:
        image_path = image_dir / file.path
        rel_path = image_path.relative_to(image_dir)
        out_image_path = output_dir / rel_path
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, out_image_path)

    logger.info(f"Successfully converted {len(dataset.files)} files and saved to {output_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Convert a JSON file to a Factory dataset.")
    parser.add_argument("--image_dir", "-i", type=Path, required=True, help="Path to the directory containing the images.")
    parser.add_argument(
        "--json_file", "-j", type=Path, default=None, help="Path to the input JSON file. Default is <image_dir>/labels.json"
    )
    parser.add_argument("--output_dir", "-o", type=Path, required=True, help="Path to the output Factory dataset directory.")
    args = parser.parse_args()

    if args.json_file is None:
        args.json_file = args.image_dir / "labels.json"

    dataset = Dataset.load(str(args.json_file))
    convert_json_to_factory(dataset, args.image_dir, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
