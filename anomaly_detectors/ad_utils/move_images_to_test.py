import shutil
import os
import argparse
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', required=True, help='path to source directory. This could be a parent directory of all images, such as each-sku')
    parser.add_argument('-f', '--file', required=True, help='path to file with a list of images to be moved to test folder')
    
    args = parser.parse_args()
    
    to_be_moved = set()
    with open(args.file, 'r') as f:
        for line in f:
            to_be_moved.add(line.strip())
    logger.info(f'{len(to_be_moved)} images to be removed')
    
    for root, dirs, files in os.walk(args.src):
        for file in files:
            if file in to_be_moved:
                path = os.path.join(root, file)
                
                # create a test folder in the parent directory
                ppath = os.path.dirname(os.path.dirname(path))
                dest = os.path.join(ppath, 'test')
                if not os.path.exists(dest):
                    os.makedirs(dest)
                
                logger.info(f'move {path} to {dest}')
                shutil.move(path, dest)
