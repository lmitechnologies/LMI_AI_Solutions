import os
import glob
import cv2
from dataset_utils.representations import Dataset

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
            raise Exception(f'File not found: {os.path.join(path_imgs, file.path)}')
        img = cv2.imread(os.path.join(path_imgs, file.path))
        if img is None:
            raise Exception(f'cannot read image: {file.path}')
        h,w = img.shape[:2]
        file.height = h
        file.width = w
    return dataset

def load_and_update(annotations_path, path_imgs):
    """
    Load the dataset and update the file dimensions.
    """
    if not os.path.exists(path_imgs):
        raise Exception(f'The image path does not exist: {path_imgs}')
    if not os.path.exists(annotations_path) and not os.path.isfile(annotations_path):
        raise Exception(f'The annotations path does not exist: {annotations_path}')
    dataset = Dataset.load(annotations_path)
    return update_file_dimensions(dataset=dataset, path_imgs=path_imgs)