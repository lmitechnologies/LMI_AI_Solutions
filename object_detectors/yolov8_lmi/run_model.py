import cv2
import logging
import os
import random
import numpy as np
import torch
import collections
from tqdm import tqdm

from yolov8_lmi.model import Yolov8, Yolov8Obb, Yolov8Pose
from gadget_utils.pipeline_utils import plot_one_rbox, get_img_path_batches, plot_one_box
from label_utils.shapes import Rect, Mask
from label_utils.csv_utils import write_to_csv
from label_utils.bbox_utils import get_rotated_bbox


BATCH_SIZE = 1
COLORS = [(0,0,255),(255,0,0),(0,255,0),(102,51,153),(255,140,0),(105,105,105),(127,25,27),(9,200,100)]


if __name__ == '__main__':
    import argparse
    import time
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
    
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger(__name__)
    if args.el:
        logger.propagate = False

    model = None
    if args.pose:
        model = Yolov8Pose(args.wts_file)
    elif args.obb:
        model = Yolov8Obb(args.wts_file)
    else:
        model = Yolov8(args.wts_file)
    
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)
        
    # get color map
    color_map = {}
    for k in sorted(model.names.keys()):
        v= model.names[k]
        if len(color_map) < len(COLORS):
            color_map[v] = COLORS[len(color_map)]
        else:
            color_map[v] = tuple([random.randint(0,255) for _ in range(3)])
        
    # warm up
    t1 = time.time()
    model.warmup(args.sz)
    t2 = time.time()
    logger.info(f'warmup input shape: {args.sz}')
    logger.info(f'warmup proc time -> {t2-t1:.4f}')
        
    fname_to_shapes = collections.defaultdict(list)
    batches = get_img_path_batches(batch_size=BATCH_SIZE, img_dir=args.path_imgs)
    logger.info(f'loaded {len(batches)} with a batch size of {BATCH_SIZE}')

    for batch in tqdm(batches):
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
            results = model.postprocess(preds,im,im1,args.confidence)
            t2 = time.time()
            
            fname = os.path.basename(p)
            save_path = os.path.join(args.path_out,fname)
            im_out = np.copy(im0)
            
            if len(results['boxes']):
                # uppack results for a single image
                boxes,scores,classes = results['boxes'][0],results['scores'][0],results['classes'][0]
                masks = results['masks'][0] if 'masks' in results else None
                segments = results['segments'][0] if 'segments' in results else []
                points = results['points'][0] if 'points' in results else []
                
                # loop through each box
                for j in range(len(boxes)-1,-1,-1): 
                    # convert box,mask to original image size
                    mask = None
                    if masks is not None:
                        mask = masks[j]
                        mask = cv2.resize(mask,(im_out.shape[1],im_out.shape[0]))
                    box = boxes[j]
                    if args.obb:
                        for b in range(len(box)):
                            box[b] = [box[b][0] /rw, box[b][1] /rh]
                    else:
                        box[[0,2]] /= rw
                        box[[1,3]] /= rh
                    box = box.astype(np.int32)
                    # annotation
                    color = color_map[classes[j]]
                    if args.obb:
                        plot_one_rbox(box,im_out,color=color,label=f'{classes[j]}: {scores[j]:.2f}')
                    else:
                        plot_one_box(box,im_out,mask,color=color,label=f'{classes[j]}: {scores[j]:.2f}')
                        
                    
                    if segments and len(segments[j]):
                        seg = segments[j]
                        # convert segments to original image size
                        seg[:,0] /= rw
                        seg[:,1] /= rh
                        seg = seg.astype(np.int32)
                        seg2 = segments[j].reshape((-1,1,2)).astype(np.int32)
                        cv2.drawContours(im_out, [seg2], -1, color, 1)
                        
                        # add masks to csv
                        if args.csv:
                            M = Mask(im_name=fname, category=classes[j], x_vals=seg[:,0].tolist(), y_vals=seg[:,1].tolist(), confidence=scores[j])
                            fname_to_shapes[fname].append(M)

                    if len(points):
                        pts = points[j]
                        # convert points to original image size
                        pts[:,0] /= rw
                        pts[:,1] /= rh
                        pts = pts.astype(np.int32)
                        for pt in pts:
                            cv2.circle(im_out, tuple(pt), 4, color, -1)
                    
                    # add rects to csv
                    if mask is None and args.csv:
                        bbox = box
                        angle = 0
                        if args.obb:
                            bbox  =  get_rotated_bbox(box)
                            x1, y1, w, h , angle = bbox
                            box  = [
                                x1, y1, x1+w, y1+h
                            ]
                            box = np.array(box)
                        
                        
                        R = Rect(im_name=fname, category=classes[j], up_left=box[:2].astype(int).tolist(), bottom_right=box[2:].astype(int).tolist(), confidence=scores[j], angle=angle)
                        fname_to_shapes[fname].append(R)
                        
                # log
                cnts = collections.Counter(classes)
                logger.info(f'fname: {fname}')
                for c in cnts:
                    logger.info(f'found {cnts[c]} {c}')
            else:
                logger.info(f'fname: {fname} --- no object detected')  
            # save output image from RGB to BGR
            cv2.imwrite(save_path,im_out[:,:,::-1])
            t3 = time.time()
            logger.info(f'proc time: {t2-t1:.4f}, cycle time: {t3-t1:.4f}\n')
            
    # write to csv
    if args.csv:
        write_to_csv(fname_to_shapes, os.path.join(args.path_out, 'preds.csv'))