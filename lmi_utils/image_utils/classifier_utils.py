import os
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IMG_FORMATS = ['.png','.jpg','jpeg','tiff']


def get_relative_paths(inpath, recursive=True, formats=IMG_FORMATS):
    """
    get the relative paths of image files to the input directory.
    args:
        inpath (str): the input directory
        recursive (bool): whether to search recursively
        formats (list): the image formats to search for, default is ['.png','.jpg','jpeg','tiff'].
    
    """
    if not isinstance(formats, list):
        raise Exception(f'formats must be a list of strings. But got the type: {type(formats)}')
    logger.info(f'Search images with these extensions: {formats}')
    logger.info(f'Search recursively?: {recursive}')
    files = []
    for root, dirs, fs in os.walk(inpath):
        # break if the root is not the input path
        if not recursive and os.path.abspath(root) != os.path.abspath(inpath):
            break
        
        relative_path = os.path.relpath(root, inpath)
        cnt = 0
        for file in fs:
            if os.path.splitext(file)[1] in formats:
                files.append(os.path.join(relative_path, file))
                cnt += 1
        logger.info(f'Load {cnt} images in {root}')
    return files
    