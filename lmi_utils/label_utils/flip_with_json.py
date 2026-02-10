# built-in packages
import logging
from pathlib import Path

import cv2
import numpy as np

# LMI packages
from dataset_utils.representations import Dataset
from system_utils.path_utils import get_relative_paths

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def flip_imgs_with_json(path_imgs, path_json, flip, path_out, recursive):
    """
    flip images and its annotations with a json file
    Arguments:
        path_imgs(Path): the image folder
        path_json(Path): the path of json annotation file
        flip(list): a list of two elements, [flip_x, flip_y], where flip_x and flip_y are boolean values
        path_out(Path): the output folder
        recursive(bool): whether to search images recursively
    """

    dataset = Dataset.load(path_json)
    paths_in_dataset = {p.path: p for p in dataset.files}
    files = get_relative_paths(path_imgs, recursive)
    files = [Path(f) for f in files]

    flipx, flipy = flip

    for f in files:
        f = path_imgs / f
        relative_path = f.parent.relative_to(path_imgs)
        im = cv2.imread(f)
        if im is None:
            logger.warning(f"Could not read image: {f}")
            continue

        h, w = im.shape[:2]

        # skip images that are not in the dataset
        key = (relative_path / f.name).as_posix()
        if key not in paths_in_dataset:
            logger.warning(f"Image {key} is not in the dataset, skip")
            continue

        file_annot = paths_in_dataset[key]

        # flip image
        im2 = im.copy()
        if flipx:
            im2 = np.flip(im2, 1)
        if flipy:
            im2 = np.flip(im2, 0)

        # write images
        tx = "x" if flipx else ""
        ty = "y" if flipy else ""
        out_name = f.stem + f"_flip_{tx}{ty}" + f.suffix
        out_path = path_out / str(relative_path) / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"write to {out_path}")
        cv2.imwrite(out_path, im2)

        # flip shapes in-place
        for annot in file_annot.annotations:
            annot.value.flip(flipx=flipx, flipy=flipy, h=h, w=w)

        # update file path using linux format
        file_annot.path = (relative_path / out_name).as_posix()

    dataset.save(path_out / "labels.json")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--path_imgs", "-i", required=True, type=Path, help="the path to images")
    ap.add_argument(
        "--path_json",
        default=Path("labels.json"),
        type=Path,
        help='[optional] the path of a json file that corresponds to path_imgs, default="labels.json" in path_imgs',
    )
    ap.add_argument("--path_out", "-o", required=True, type=Path, help="the path to resized images")
    ap.add_argument("--flip_lr", action="store_true", help="flip images horizontally")
    ap.add_argument("--flip_ud", action="store_true", help="flip images vertically")
    ap.add_argument("--recursive", action="store_true", help="search images recursively")
    args = vars(ap.parse_args())

    path_imgs = args["path_imgs"]
    path_out = args["path_out"]
    path_json = args["path_json"] if args["path_json"] != Path("labels.json") else path_imgs / "labels.json"

    # check if annotation exists
    if not path_json.is_file():
        raise Exception(f"cannot find file: {path_json}")

    # check if input and output paths are different
    assert path_imgs != path_out, "input and output path must be different"

    # check if at least one flip direction is specified
    if not args["flip_lr"] and not args["flip_ud"]:
        raise Exception("at least one flip direction must be specified")

    flip = [args["flip_lr"], args["flip_ud"]]
    flip_imgs_with_json(path_imgs, path_json, flip, path_out, args["recursive"])
