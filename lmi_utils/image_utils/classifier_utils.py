import os
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IMG_FORMATS = ['.png','.jpg','jpeg','tiff']


def get_im_relative_path(inpath):
    """
    get the relative paths of image files to the input directory. Currently only supports .png, .jpg, .jpeg, .tiff.
    """
    logger.info(f'looking for images in these formats: {IMG_FORMATS}')
    files = []
    for root, dirs, fs in os.walk(inpath):
        relative_path = os.path.relpath(root, inpath)
        logger.info(f'Processing {relative_path} with {len(fs)} images...')
        for file in fs:
            if os.path.splitext(file)[1] in IMG_FORMATS:
                files.append(os.path.join(relative_path, file))
    return files
    