from collections import OrderedDict, namedtuple
import cv2
import logging
import os
import numpy as np
import torch
import sys
from typing import Union
import collections
import time

# add yolov5 submodule to the path
YOLO_PATH = os.path.join(os.path.dirname(__file__), '../submodules/yolov5')
if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(f'Cannot find yolov5 submodule at {YOLO_PATH}')
sys.path.insert(0, YOLO_PATH)

from utils.general import non_max_suppression, scale_boxes, scale_segments
from utils.segment.general import masks2segments, process_mask, process_mask_native
from models.common import DetectMultiBackend
from utils.torch_utils import smart_inference_mode

from od_base import ODBase
import gadget_utils.pipeline_utils as pipeline_utils
from yolov8_lmi.model import Yolov8


class Yolov5(ODBase):
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
        
        
    @smart_inference_mode()
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    
    @smart_inference_mode()
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
    
    
    @smart_inference_mode()
    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (np.ndarray | tensor): BCHW for tensor, [(HWC) x B] for list.
        """
        if isinstance(im, np.ndarray):
            im = np.ascontiguousarray(im)  # contiguous
            im = self.from_numpy(im)
        
        # convert to HWC
        if im.ndim == 2:
            im = im.unsqueeze(-1)
        if im.shape[-1] ==1:
            im = im.expand(-1,-1,3)
            
        im = im.unsqueeze(0) # HWC -> BHWC
        img = im.permute((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)

        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img
    
    
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
    
    
    @smart_inference_mode()
    def predict(self, image, configs, operators=[], iou=0.4, agnostic=False, max_det=300, return_segments=True):
        """object detection inference. It runs the preprocess(), forward(), and postprocess() in sequence.
        It converts the results to the original coordinates space if the operators are provided.
        
        Args:
            image (np.ndarry): the input image
            configs (dict): a dictionary of the confidence thresholds for each class, e.g., {'classA':0.5, 'classB':0.6}
            operators (list): a list of dictionaries of the image preprocess operators, such as {'resize':[resized_w, resized_h, orig_w, orig_h]}, {'pad':[pad_left, pad_right, pad_top, pad_bot]}
            iou (float): the iou threshold for non-maximum suppression. defaults to 0.4
            agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.
            max_det (int): The maximum number of detections to return. defaults to 300.
            return_segments(bool): If True, return the segments of the masks.

        Returns:
            list of [results, time info]
            results (dict): a dictionary of the results, e.g., {'boxes':[], 'classes':[], 'scores':[], 'masks':[], 'segments':[]}
            time_info (dict): a dictionary of the time info, e.g., {'preproc':0.1, 'proc':0.2, 'postproc':0.3}
        """
        time_info = {}
        
        # preprocess
        t0 = time.time()
        im = self.preprocess(image)
        time_info['preproc'] = time.time()-t0
        
        # infer
        t0 = time.time()
        pred = self.forward(im)
        time_info['proc'] = time.time()-t0
        
        # postprocess
        t0 = time.time()
        conf_thres = {}
        for k in configs:
            conf_thres[k] = configs[k]
        results = self.postprocess(pred,im,image,conf_thres,iou,agnostic,max_det,return_segments)
        
        # return empty results if no detection
        results_dict = collections.defaultdict(list)
        if not len(results['boxes']):
            time_info['postproc'] = time.time()-t0
            return results_dict, time_info
        
        # only one image, get first batch
        boxes = results['boxes'][0]
        scores = results['scores'][0].tolist()
        classes = results['classes'][0].tolist()

        # deal with segmentation results
        if len(results['masks']):
            masks = results['masks'][0]
            segs = results['segments'][0]
            # convert mask to sensor space
            result_contours = [pipeline_utils.revert_to_origin(seg, operators) for seg in segs]
            masks = pipeline_utils.revert_masks_to_origin(masks, operators)
            results_dict['segments'] = result_contours
            results_dict['masks'] = masks
        
        # convert box to sensor space
        boxes = pipeline_utils.revert_to_origin(boxes, operators)
        results_dict['boxes'] = boxes
        results_dict['scores'] = scores
        results_dict['classes'] = classes
            
        time_info['postproc'] = time.time()-t0
        return results_dict, time_info


    @staticmethod
    def annotate_image(results, image, colormap=None):
        """annotate the object dectector results on the image. If colormap is None, it will use the random colors.
        TODO: text size, thickness, font

        Args:
            results (dict): the results of the object detection, e.g., {'boxes':[], 'classes':[], 'scores':[], 'masks':[], 'segments':[]}
            image (np.ndarray): the input image
            colors (list, optional): a dictionary of colormaps, e.g., {'class-A':(0,0,255), 'class-B':(0,255,0)}. Defaults to None.

        Returns:
            np.ndarray: the annotated image
        """
        return Yolov8.annotate_image(results, image, colormap)
