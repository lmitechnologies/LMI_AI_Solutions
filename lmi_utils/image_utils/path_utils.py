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
    
    logger.info(f'Search files with the following extensions:\n {formats}')
    files = []
    for root, dirs, fs in os.walk(inpath):
        cnt = 0
        for file in fs:
            if os.path.splitext(file)[1] in formats:
                files.append(os.path.relpath(os.path.join(root, file), inpath))
                cnt += 1
        logger.info(f'Load {cnt} files in {root}')
        
        if not recursive:
            break
    return files
    
    
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input_path', required=True, help='the path to images')
    ap.add_argument('--recursive', action='store_true', help='process images recursively')
    args = ap.parse_args()
    
    paths = get_relative_paths(args.input_path,args.recursive)
    for p in paths:
        logger.info(f'output path: {p}')