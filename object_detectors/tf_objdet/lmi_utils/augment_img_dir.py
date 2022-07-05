import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path, PurePath
import cv2
from tqdm import tqdm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input_directory",
        required=True,
        help="path to directory of input images",
    )
    ap.add_argument(
        "-o", "--output_directory", required=True, help="where to write output images"
    )
    ap.add_argument(
        "-a",
        "--augmentation_config",
        required=True,
        help="path to Python file detailing augmentation method.",
    )
    args = vars(ap.parse_args())
    img_dir = args["input_directory"]
    output_dir = args["output_directory"]
    aug_config = args["augmentation_config"]
    img_paths = [f for f in Path(img_dir).iterdir()]
    if not Path(output_dir).is_dir():
        raise ValueError(f"The directory {output_dir} does not exist")

    aug_config = __import__(Path(aug_config).stem)
    TOTAL = aug_config.COUNT if aug_config.COUNT else len(img_paths)
    remaining = TOTAL

    for img_path in tqdm(img_paths, total=len(img_paths)):
        img = cv2.imread(img_path.as_posix())
        if img is None:
            raise ValueError("Encountered non-image file. Aborting...")
        # Ceiling division
        # Goal is to distribute augmented images as evenly as possible across
        # original images. All should have same representation except for 1,
        # which will have less
        cur_count = min(-(-TOTAL // len(img_paths)), remaining)
        remaining -= cur_count
        for i in range(cur_count):
            aug_img = aug_config.augment(img)
            aug_img_path = PurePath(output_dir, img_path.stem + f"-{i}.png").as_posix()
            cv2.imwrite(aug_img_path, aug_img)