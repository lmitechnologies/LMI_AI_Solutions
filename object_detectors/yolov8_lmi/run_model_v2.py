import cv2
import logging
import os
import random
import numpy as np
import torch
import collections
from tqdm import tqdm
import argparse
import time

from yolov8_lmi.model import Yolov8, Yolov8Obb, Yolov8Pose
from gadget_utils.pipeline_utils import plot_one_rbox, get_img_path_batches, plot_one_box
from label_utils.rect import Rect
from label_utils.mask import Mask
from label_utils.keypoint import Keypoint
from label_utils.csv_utils import write_to_csv
from label_utils.bbox_utils import get_rotated_bbox
import logging
logger = logging.getLogger(__name__)


BATCH_SIZE = 1
COLORS = [(0,0,255),(255,0,0),(0,255,0),(102,51,153),(255,140,0),(105,105,105),(127,25,27),(9,200,100)]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--wts_file', required=True, help='the path to the model weights file. The type of supported files are: ".pt" or ".engine"')
    parser.add_argument('-i','--path_imgs', required=True, help='the path to the testing images')
    parser.add_argument('-o','--path_out' , required=True, help='the path to the output folder')
    parser.add_argument('--sz', required=True, nargs=2, type=int, help='the model input size, two numbers: h w')
    parser.add_argument('-c','--confidence',default=0.25,type=float,help='[optional] the confidence for all classes, default=0.25')
    parser.add_argument('--csv', action='store_true', help='[optional] whether to save the results to csv file')
    parser.add_argument('--obb', action='store_true', help='[optional] whether to run Oriented Bounding Box model')
    parser.add_argument('--pose', action='store_true', help='[optional] whether to run Pose model')
    parser.add_argument('--el', action='store_false', help='[optional] log level default is ERROR', default=False)
    args = parser.parse_args()
    return args

def get_model(args):
    model = None
    if args.pose:
        model = Yolov8Pose(args.wts_file)
    elif args.obb:
        model = Yolov8Obb(args.wts_file)
    else:
        # default to regular yolov8 model for regular bounding boxes and segmentation masks
        model = Yolov8(args.wts_file)
    return model

def get_color_map(model):
    color_map = {}
    for k in sorted(model.names.keys()):
        v= model.names[k]
        if len(color_map) < len(COLORS):
            color_map[v] = COLORS[len(color_map)]
        else:
            color_map[v] = tuple([random.randint(0,255) for _ in range(3)])
    return color_map

def preprocess_image(img, sz):
    im1 = img
    operators = []
    if sz[0] != img.shape[0] or sz[1] != img.shape[1]:
        rh,rw = sz[0]/img.shape[0],sz[1]/img.shape[1]
        im1 = cv2.resize(img,(sz[1],sz[0]))
        operators.append({'resize':[sz[1], sz[0], img.shape[1], img.shape[0]]})
    return im1, operators

def main():
    # get args
    args = get_args()
    # get model
    model = get_model(args)
    # get color map
    color_map = get_color_map(model)
    # warm up
    model.warmup(args.sz)
    # get image batches
    batches = get_img_path_batches(batch_size=BATCH_SIZE, img_dir=args.path_imgs, fmt='png')

    # create confidence map
    confidence_map = {k:args.confidence for k in model.names.values()}

    # intialize fname to shapes
    fname_to_shapes = collections.defaultdict(list)

    for batch in tqdm(batches):
        for p in batch:
            # load image
            fname = os.path.basename(p)
            im0 = cv2.imread(p,cv2.IMREAD_UNCHANGED)
            if len(im0.shape)==2:
                im0=cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
            im0 = im0[:,:,::-1] #BGR to RGB
            # preprocess image
            im1, operators = preprocess_image(im0, args.sz)
            logger.info(f"Processing image: {fname}")
            logger.info(f"Image shape: {im1.shape}")

            # inference
            results, time_info = model.predict(
                image=im1,
                configs=confidence_map,
                operators=operators,
            )
            annotated_image = model.annotate_image(
                results=results, image=cv2.cvtColor(im0, cv2.COLOR_RGB2BGR), colormap=color_map
            )
            for i in range(len(results.get("classes"))):
                if "masks" in results and len(results["masks"]) > 0:
                    mask = results["masks"][i]
                    m = Mask(
                        im_name=fname,
                        category=results["classes"][i],
                        x_vals=mask[:, 0].tolist(),
                        y_vals=mask[:, 1].tolist(),
                        confidence=results["scores"][i],
                    )
                    fname_to_shapes[fname].append(m)
                
                if "boxes" in results and len(results["boxes"]) > 0:
                    box = np.array(results["boxes"][i])
                    angle = 0
                    if args.obb:
                        bbox = get_rotated_bbox(box)
                        x1, y1, w, h , angle = bbox
                        box  = [
                            x1, y1, x1+w, y1+h
                        ]
                        box = np.array(box)

                    r = Rect(im_name=fname, category=results["classes"][i], up_left=box[:2].astype(int).tolist(), bottom_right=box[2:].astype(int).tolist(), confidence=results["scores"][i], angle=angle)
                    fname_to_shapes[fname].append(r)
                
                if "points" in results and len(results["boxes"]) > 0:
                    kp = np.squeeze(results["points"][i])
                    k = Keypoint(
                        im_name=fname,
                        category=results["classes"][i],
                        x=kp[0],
                        y=kp[1],
                        confidence=results["scores"][i],
                    )
                    fname_to_shapes[fname].append(k)
            cv2.imwrite(os.path.join(args.path_out,fname), annotated_image)
    # write to csv file
    if args.csv:
        write_to_csv(fname_to_shapes, os.path.join(args.path_out, 'preds.csv')) 
    return fname_to_shapes

if __name__ == '__main__':
    main()