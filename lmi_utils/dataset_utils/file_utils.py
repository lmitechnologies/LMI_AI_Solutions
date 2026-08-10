import glob
import logging
import os
import shutil

from PIL import ExifTags, Image

from lmi_utils.dataset_utils.representations import Dataset

logger = logging.getLogger(__name__)

IMAGE_FORMATS = IMG_FORMATS = ["jpeg", "jpg", "png", "tif", "tiff", "heic"]

_EXIF_ORIENTATION_TAG = ExifTags.Base.Orientation
_EXIF_QUARTER_TURNS = {5, 6, 7, 8}  # the orientations that transpose the axes; a flip or half turn does not
_ORIENTATION_UNAPPLIED_FORMATS = {"JPEG", "MPO", "PNG"}


def get_files(directory, extensions):
    """Get all files in a directory with a given extension.

    Args:
        directory (str): the directory to search
        extension (str): the file extension to search for

    Returns:
        list: a list of file paths
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
    return files


def get_images(images_dir):
    """Get all images in a directory.

    Args:
        images_dir (str): the directory to search

    Returns:
        list: a list of image paths
    """
    return get_files(images_dir, IMG_FORMATS)


def update_file_dimensions(dataset, path_imgs):
    """
    update the file dimensions
    """
    for file in dataset.files:
        image_path = os.path.join(path_imgs, file.path)
        if os.path.isfile(image_path) is False:
            raise Exception(f"File not found: {image_path}")
        try:
            # Image.open parses the header only; decoding the pixels to read two integers costs ~100x more.
            with Image.open(image_path) as image:
                width, height = image.size
                # A camera can tag a quarter turn instead of rewriting the pixels. Viewers and cv2 apply
                # the tag, so annotations are drawn against the turned image and must be normalized to it.
                if image.format in _ORIENTATION_UNAPPLIED_FORMATS and image.getexif().get(_EXIF_ORIENTATION_TAG) in _EXIF_QUARTER_TURNS:
                    width, height = height, width
                file.width, file.height = width, height
        except Exception as error:
            raise Exception(f"cannot read image: {file.path}") from error
    return dataset


def load_and_update(annotations_path, path_imgs):
    """
    Load the dataset and update the file dimensions.
    """
    if not os.path.exists(path_imgs):
        raise Exception(f"The image path does not exist: {path_imgs}")
    if not os.path.exists(annotations_path) and not os.path.isfile(annotations_path):
        raise Exception(f"The annotations path does not exist: {annotations_path}")
    dataset = Dataset.load(annotations_path)
    return update_file_dimensions(dataset=dataset, path_imgs=path_imgs)


def copy_images_in_folder(path_img, path_out, fnames, file_id_map):
    """
    copy the images from one folder to another
    Arguments:
        path_img(str): the path of original image folder
        path_out(str): the path of output folder
    """
    os.makedirs(path_out, exist_ok=True)
    if fnames is None:
        raise Exception("fnames cannot be None")
    for fname in fnames:
        logger.debug(f"Copying image {fname} to {path_out}")
        out_name = os.path.basename(fname)
        if not os.path.isfile(os.path.join(path_img, fname)):
            raise Exception(f"File not found: {os.path.join(path_img, fname)}")
        file_id = file_id_map[fname]
        if f"id{file_id}_" not in out_name:
            out_name = f"id{file_id}_{out_name}"
        shutil.copy(os.path.join(path_img, fname), os.path.join(path_out, out_name))
