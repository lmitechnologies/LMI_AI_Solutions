import os
import glob

IMAGE_FORMATS=IMG_FORMATS = ["jpeg", "jpg", "png", "tif", "tiff", "heic"]

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