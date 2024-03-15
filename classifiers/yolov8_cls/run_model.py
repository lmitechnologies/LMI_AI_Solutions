import cv2
import logging
import os
import random
import numpy as np
import torch
import collections

from yolov8_cls.model import Yolov8_cls
from gadget_utils.pipeline_utils import get_img_path_batches


BATCH_SIZE = 1


if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--wts_file', required=True, help='the path to the model weights file. The type of supported files are: ".pt" or ".engine"')
    parser.add_argument('-i','--path_imgs', required=True, help='the path to the testing images')
    parser.add_argument('-o','--path_out' , required=True, help='the path to the output folder')
    parser.add_argument('--sz', required=True, nargs=2, type=int, help='the model input size, two numbers: h w')
    # parser.add_argument('-c','--confidence',default=0.25,type=float,help='[optional] the confidence for all classes, default=0.25')
    # parser.add_argument('--csv', action='store_true', help='[optional] whether to save the results to csv file')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    model = Yolov8_cls(args.wts_file)
    
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)
        
    # warm up
    t1 = time.time()
    model.warmup(args.sz)
    t2 = time.time()
    logger.info(f'warmup input shape: {args.sz}')
    logger.info(f'warmup proc time -> {t2-t1:.4f}')
        
    fname_to_shapes = collections.defaultdict(list)
    batches = get_img_path_batches(batch_size=BATCH_SIZE, img_dir=args.path_imgs)
    logger.info(f'loaded {len(batches)} with a batch size of {BATCH_SIZE}')
    for batch in batches:
        for p in batch:
            t1 = time.time()
            # load image
            im0 = cv2.imread(p,cv2.IMREAD_UNCHANGED) #BGR format
            if len(im0.shape)==2:
                im0=cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
            im0 = im0[:,:,::-1] #BGR to RGB
            
            # warp image
            im1 = im0
            if args.sz[0] != im0.shape[0] or args.sz[1] != im0.shape[1]:
                logger.warning(f'model input size: {args.sz} is different from image size: {im0.shape}, warping image')
                rh,rw = args.sz[0]/im0.shape[0],args.sz[1]/im0.shape[1]
                im1 = cv2.resize(im0,(args.sz[1],args.sz[0]))
            else:
                rh,rw =1.0,1.0
            
            # inference
            im = model.preprocess(im1)
            preds = model.forward(im)
            results = model.postprocess(preds)
            t2 = time.time()
            
            # TODO save images according to the classes
            fname = os.path.basename(p)
            save_path = os.path.join(args.path_out,fname)
            im_out = np.copy(im0)
            
            # save output image from RGB to BGR
            cv2.imwrite(save_path,im_out[:,:,::-1])
            t3 = time.time()
            logger.info(f'proc time: {t2-t1:.4f}, cycle time: {t3-t1:.4f}\n')
