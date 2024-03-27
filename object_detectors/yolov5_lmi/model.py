from collections import OrderedDict, namedtuple
import cv2
import logging
import os
import numpy as np
import torch
import sys
from typing import Union
import collections

# add yolov5 submodule to the path
YOLO_PATH = os.path.join(os.path.dirname(__file__), '../submodules/yolov5')
if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(f'Cannot find yolov5 submodule at {YOLO_PATH}')
sys.path.insert(0, YOLO_PATH)

from utils.general import non_max_suppression, scale_boxes, scale_segments
from utils.segment.general import masks2segments, process_mask, process_mask_native
from models.common import DetectMultiBackend
from utils.torch_utils import smart_inference_mode


class Yolov5:
    logger = logging.getLogger(__name__)
    
    
    def __init__(self, weights:str, device='gpu', data=None, fp16=False) -> None:
        """
        args:
            weights(str): the path to the tensorRT engine file
            data (str, optional): the path to the yaml file containing class names. Defaults to None.
        """
        if not os.path.isfile(weights):
            raise FileNotFoundError(f'File not found: {weights}')
        
        # set device
        self.device = torch.device('cpu')
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')  
            else:
                self.logger.warning('GPU not available, using CPU')
                
        self.model = DetectMultiBackend(weights, self.device, data=data, fp16=fp16)
        self.model.eval()
        
        # class map < id: class name >
        self.names = self.model.names
        

    @smart_inference_mode()
    def forward(self, im):
        return self.model(im)
        
        
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    
    def warmup(self, imgsz=[640, 640]):
        """
        warm up the model once
        Args:
            imgsz (list, optional): list of [H,W] format. Defaults to (640, 640).
        """
        if isinstance(imgsz, tuple):
            imgsz = list(imgsz)
        # if self.imgsz and self.imgsz != imgsz:
        #     raise Exception(f'The warmup imgsz of {imgsz} does not match with the size of the model: {self.imgsz}!')
        
        imgsz = [1,3]+imgsz
        im = torch.empty(*imgsz, dtype=torch.half if self.model.fp16 else torch.float, device=self.device)  # input
        self.forward(im)
    
    
    def preprocess(self, im):
        """im preprocess
            normalization -> CHW -> contiguous -> BCHW
        Args:
            im0 (numpy.ndarray): the input numpy array, HWC format
        """
        if not isinstance(im, np.ndarray):
            raise TypeError(f'Image type {type(im)} not supported')
        
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        
        im = im.astype(np.float32)
        im /= 255 # normalize to [0,1]
        im = im.transpose((2, 0, 1)) # HWC to CHW
        im = np.ascontiguousarray(im)  # contiguous
        if len(im.shape) == 3:
            im = im[None] # expand for batch dim
        return self.from_numpy(im)
    
    
    def load_with_preprocess(self, im_path:str):
        """im preprocess

        Args:
            im_path (str): the path to the image, could be either .npy, .png, or other image formats
            
        """
        ext = os.path.splitext(im_path)[-1]
        if ext=='.npy':
            im0 = np.load(im_path)
        else:
            im0 = cv2.imread(im_path) #BGR format
            im0 = im0[:,:,::-1] #BGR to RGB
        return self.preprocess(im0),im0
    
    
    @smart_inference_mode()
    def postprocess(self,preds,im,orig_imgs,conf: Union[float, dict],iou=0.45,agnostic=False,max_det=300,return_segments=True):
        """
        Args:
            preds (list): a list of object detection predictions
            im (tensor): the preprocessed image
            orig_imgs (np.ndarray | list): the original images
            conf (dict): confidence threshold dict <class: confidence>.
            iou (float, optional): iou threshold. Defaults to 0.45.
            agnostic (bool, optional): perform class-agnostic NMS. Defaults to False.
            max_det (int, optional): the max number of detections. Defaults to 300.
            return_segments(bool): If True, return the segments of the masks.
        Returns:
            (dict): the dictionary contains several keys: boxes, scores, classes, masks, and (masks, segments if use a segmentation model).
                    the shape of boxes is (B, N, 4), where B is the batch size and N is the number of detected objects.
                    the shape of classes and scores are both (B, N).
                    the shape of masks: (B, H, W, 3), where H and W are the height and width of the input image.
        """
        proto = None
        nm = 0
        if isinstance(preds, (list,tuple)):
            if len(preds)>1 and isinstance(preds[1],torch.Tensor) and preds[1].ndim==4:
                preds,proto = preds[0], preds[1]
                nm = 32
            else:
                preds = preds[0]
        elif not isinstance(preds, torch.Tensor):
            raise TypeError(f'Prediction type {type(preds)} not supported')
        
        # get min confidence for nms
        if isinstance(conf, float):
            conf2 = conf
        elif isinstance(conf, dict):
            conf2 = 1
            class_names = set(self.model.names.values())
            for k,v in conf.items():
                if k in class_names:
                    conf2 = min(conf2, v)
            if conf2 == 1:
                self.logger.warning('No class matches in confidence dict, set to 1.0 for all classes.')
        else:
            raise TypeError(f'Confidence type {type(conf)} not supported')
        
        pred = non_max_suppression(preds,conf2,iou,agnostic=agnostic,max_det=max_det,nm=nm)
        
        # Process predictions
        results = collections.defaultdict(list)
        for i,det in enumerate(pred):  # per image
            if len(det)==0:
                continue
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], orig_img.shape).round()
            xyxy,confs,clss = det[:, :4],det[:, 4],det[:, 5]
            classes = np.array([self.model.names[c.item()] for c in clss])
            
            # filter based on conf
            if isinstance(conf, float):
                thres = np.array([conf]*len(clss))
            if isinstance(conf, dict):
                # set to 1 if c is not in conf
                thres = np.array([conf.get(c,1) for c in classes])
            M = confs > self.from_numpy(thres)
            
            results['boxes'].append(xyxy[M].cpu().numpy())
            results['scores'].append(confs[M].cpu().numpy())
            results['classes'].append(classes[M.cpu().numpy()])
            if proto is not None:
                masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], orig_img.shape[:2])
                masks = masks[M]
                results['masks'].append(masks.cpu().numpy())
                if return_segments:
                    segs = [scale_segments(im.shape[2:], x, orig_img.shape, normalize=False)
                            for x in reversed(masks2segments(masks))]
                    results['segments'].append(segs)
        return results
