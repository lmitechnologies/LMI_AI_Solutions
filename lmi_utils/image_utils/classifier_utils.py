import os
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IMG_FORMATS = ['.png','.jpg','jpeg','tiff']


def get_im_relative_path(inpath, recursive=True):
    """
    get the relative paths of image files to the input directory. Currently only supports .png, .jpg, .jpeg, .tiff.
    """
    logger.info(f'Looking for images in these formats: {IMG_FORMATS}')
    logger.info(f'Processing recursively on sub directories: {recursive}')
    files = []
    for root, dirs, fs in os.walk(inpath):
        # break if the root is not the input path
        if not recursive and os.path.abspath(root) != os.path.abspath(inpath):
            break
        
        relative_path = os.path.relpath(root, inpath)
        cnt = 0
        for file in fs:
            if os.path.splitext(file)[1] in IMG_FORMATS:
                files.append(os.path.join(relative_path, file))
                cnt += 1
        logger.info(f'Processed {cnt} images in {root}')
    return files
    