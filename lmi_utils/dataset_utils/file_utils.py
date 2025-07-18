import os
import glob
import cv2

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

def update_file_dimensions(dataset, path_imgs):
    """
    update the file dimensions
    """
    for file in dataset.files:
        if os.path.isfile(os.path.join(path_imgs, file.path)) is False:
            raise Exception(f'File not found: {file.path}')
        img = cv2.imread(os.path.join(path_imgs, file.path))
        if img is None:
            raise Exception(f'cannot read image: {file.path}')
        h,w = img.shape[:2]
        file.height = h
        file.width = w
    return dataset