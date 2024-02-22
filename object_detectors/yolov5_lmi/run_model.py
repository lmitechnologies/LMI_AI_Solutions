import cv2
import logging
import os
import random
import numpy as np
import torch

from yolov5_lmi.model import Yolov5
from gadget_utils.pipeline_utils import get_img_path_batches

BATCH_SIZE = 1


def plot_one_box(box, img, mask=None, mask_threshold:int=0, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box and mask (optinal) on image img,
                 this function comes from YoLov5 project.
    param: 
        box:    a box likes [x1,y1,x2,y2]
        img:    a opencv image object in BGR format
        mask:   a binary mask for the box
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    if isinstance(box, list):
        box = np.array([x.cpu() for x in box])
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
        
    x1,y1,x2,y2 = box.astype(int)
    c1, c2 = (x1, y1), (x2, y2)
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if mask is not None:
        # mask *= 255
        m = mask>mask_threshold
        blended = (0.4 * np.array(color,dtype=float) + 0.6 * img[m]).astype(np.uint8)
        img[m] = blended
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 4,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )



if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--engine',help='the path to the tensorRT engine file')
    parser.add_argument('--imsz',type=int,nargs=2,help='the image size: h w')
    parser.add_argument('-i','--path_imgs',help='the path to the images')
    parser.add_argument('-o','--path_out',help='the path to the output folder')
    parser.add_argument('-c','--confidence',default=0.25,type=float,help='the confidence for all classes')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.NOTSET)
    
    h,w = args.imsz
    engine = Yolov5(args.engine)
    logger = engine.logger
    logger.info(f'input imsz: {args.imsz}')
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)
    for _ in range(1):
        t1 = time.time()
        engine.warmup(imgsz=(h,w))
        t2 = time.time()
        logger.info(f'warm up proc time -> {t2-t1:.4f}')
        
    batches = get_img_path_batches(batch_size=BATCH_SIZE, img_dir=args.path_imgs)
    logger.info(f'loaded {len(batches)} with a batch size of {BATCH_SIZE}')
    for batch in batches:
        for p in batch:
            t1 = time.time()
            im,im0 = engine.load_with_preprocess(p)
            preds = engine.forward(im)
            results = engine.postprocess(preds,im,im0,args.confidence)
            # batch of 1
            if not len(results['boxes']):
                logger.info(f'no object detected')
                continue
            
            boxes, scores, classes = results['boxes'][0],results['scores'][0],results['classes'][0]
            masks = None
            if len(results['masks']):
                masks = results['masks'][0]

            #annotation
            fname = os.path.basename(p)
            save_path = os.path.join(args.path_out,fname)
            im_out = np.copy(im0)
            for i,xyxy in enumerate(boxes):
                cls,conf = classes[i],scores[i]
                if masks is not None:
                    plot_one_box(xyxy,im_out,masks[i],label=f'{cls}: {conf:.2f}')
                else:
                    plot_one_box(xyxy,im_out,label=f'{cls}: {conf:.2f}')
            cv2.imwrite(save_path,im_out[:,:,::-1])
                
            t2 = time.time()
            logger.info(f'proc time of {fname} -> {t2-t1:.4f}')
            
