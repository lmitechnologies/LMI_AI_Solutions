import cv2
import os
import argparse
import glob

import gadget_utils.pipeline_utils as pipeline_utils


BLACK=(0,0,0)

def hstack_imgs(orig_folder, pred_folder, fmt, path_out):
    orig_files = glob.glob(os.path.join(orig_folder, f'*.{fmt}'))
    pred_files = glob.glob(os.path.join(pred_folder, f'*.{fmt}'))
    orig_files.sort()
    pred_files.sort()
    assert len(orig_files)==len(pred_files), 'the number of images in the two folders must be the same'
    for orig_file, pred_file in zip(orig_files, pred_files):
        orig_im = cv2.imread(orig_file)
        pred_im = cv2.imread(pred_file)
        assert orig_im.shape==pred_im.shape, f'the shape of {orig_file} and {pred_file} must be the same'
        hstack_im = cv2.hconcat([orig_im, pred_im])
        cv2.imwrite(os.path.join(path_out, os.path.basename(orig_file)), hstack_im)
    


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--path_orig', required=True, help='the path to the original images')
    ap.add_argument('--path_pred', required=True, help='the path to the predicgtion images')
    ap.add_argument('-o', '--path_out', required=True, help='the output path')
    ap.add_argument('--fmt', default='png', help='the format of the images')
    args=ap.parse_args()

    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)
    
    hstack_imgs(args.path_orig, args.path_pred, args.fmt, args.path_out)
