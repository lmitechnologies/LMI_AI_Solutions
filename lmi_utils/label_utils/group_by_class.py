import os
import logging
import argparse
import random
import cv2

from .csv_utils import load_csv
from .plot_labels import plot_shape


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def group_by_classes(path_imgs, path_csv, path_out, annot):
    """
    Group images by classes
    if annot is True, the output images will be annotated with labels
    """
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    
    # read csv
    fname_to_shapes,class_map = load_csv(path_csv)
    logger.info(f'found number of images: {len(fname_to_shapes)}')
    
    color_map = {}
    for cls in sorted(class_map.keys()):
        color_map[cls] = tuple([random.randint(0,255) for _ in range(3)])
    
    # move images
    for fname in fname_to_shapes:
        im = cv2.imread(os.path.join(path_imgs, fname))
        if im is None:
            logger.warning(f'cannot read image: {fname}')
            continue
        
        # annotate image
        if annot:
            for shape in fname_to_shapes[fname]:
                plot_shape(shape, im, color_map)
                
        # save image
        for shape in fname_to_shapes[fname]:
            class_name = shape.category
            if not os.path.isdir(os.path.join(path_out, class_name)):
                os.makedirs(os.path.join(path_out, class_name))
            cv2.imwrite(os.path.join(path_out, class_name, fname), im)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--path_imgs", required=True, help="path to images")
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('-o', '--path_out', required=True, help='output path')
    ap.add_argument('--annot', action='store_true', help='[optional] if True, the output images will be annotated with labels')
    args = ap.parse_args()
    
    path_csv = args.path_csv if args.path_csv!='labels.csv' else os.path.join(args.path_imgs, args.path_csv)
    
    #check if annotation exists
    if not os.path.isfile(path_csv):
        raise Exception(f'cannot find file: {path_csv}.')
    
    group_by_classes(args.path_imgs, path_csv, args.path_out, args.annot)
    