import os
import cv2
import glob
import shutil
import argparse
import logging

import label_utils.csv_utils as csv_utils

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--path_imgs', required=True, help='the path to images')
    ap.add_argument('--path_csv', default='labels.csv', help='the path to csv file, default="labels.csv" in path_imgs')
    ap.add_argument('-o','--out_path', required=True, help='the output path')
    args = ap.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # read csv
    dt,_ = csv_utils.load_csv(args.path_csv, args.path_imgs)
    
    lists = glob.glob(args.path_imgs + '/*.png')
    for p in lists:
        fname = os.path.basename(p)
        # skip if the image is not in the csv
        if fname in dt:
            shutil.copy2(p, os.path.join(args.out_path, fname))
    shutil.copy2(args.path_csv, os.path.join(args.out_path, os.path.basename(args.path_csv)))