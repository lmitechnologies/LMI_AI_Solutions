import cv2
import logging
import os
import random
import numpy as np
import torch

from yolov8_lmi.trt.yolo_trt import Yolov8_trt
from gadget_utils.pipeline_utils import plot_one_box, get_img_path_batches


BATCH_SIZE = 1


if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--engine',help='the path to the tensorRT engine file')
    parser.add_argument('-i','--path_imgs',help='the path to the images')
    parser.add_argument('-o','--path_out',help='the path to the output folder')
    parser.add_argument('-c','--confidence',default=0.25,type=float,help='[optional] the confidence for all classes, default=0.25')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.NOTSET)
    
    engine = Yolov8_trt(args.engine)
    logger = logging.getLogger(__name__)
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)
        
    # warm up
    t1 = time.time()
    engine.warmup()
    t2 = time.time()
    logger.info(f'warm up proc time -> {t2-t1:.4f}')
        
    batches = get_img_path_batches(batch_size=BATCH_SIZE, img_dir=args.path_imgs)
    logger.info(f'loaded {len(batches)} with a batch size of {BATCH_SIZE}')
    for batch in batches:
        for p in batch:
            t1 = time.time()
            
            # inference
            im,im0 = engine.load_with_preprocess(p)
            preds = engine.forward(im)
            results = engine.postprocess(preds,im,im0,args.confidence)

            #annotation
            fname = os.path.basename(p)
            save_path = os.path.join(args.path_out,fname)
            im_out = np.copy(im0)
            for i in range(len(results['boxes'])): # loop through batch
                # uppack results
                boxes,scores,classes = results['boxes'][i],results['scores'][i],results['classes'][i]
                masks = results['masks'][i] if 'masks' in results else None
                segments = results['segments'][i] if 'segments' in results else []
                for j in range(len(boxes)-1,-1,-1): # loop through each box
                    mask = masks[j] if masks is not None else None
                    plot_one_box(boxes[j],im_out,mask,label=f'{classes[j]}: {scores[j]:.2f}')
                    if segments and len(segments[j]):
                        seg = np.array(segments[j]).reshape((-1,1,2)).astype(np.int32)
                        cv2.drawContours(im_out, [seg], -1, (0, 255, 0), 1)
                
            # save output image from RGB to BGR
            cv2.imwrite(save_path,im_out[:,:,::-1])
                
            t2 = time.time()
            logger.info(f'proc time of {fname} -> {t2-t1:.4f}')
