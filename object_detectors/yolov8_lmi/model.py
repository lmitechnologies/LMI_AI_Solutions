import cv2
import numpy as np
import torch
import os
import collections
import logging
from typing import Union

from ultralytics.utils import ops
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import smart_inference_mode


class Yolov8:
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, weights:str, device='gpu', data=None, fp16=False) -> None:
        """init the model
        Args:
            weights (str): the path to the weights file.
            device (str, optional): GPU or CPU device. Defaults to 'gpu'.
            data (str, optional): the path to dataset yaml file. Defaults to None.
            fp16 (bool, optional): Whether to use fp16. Defaults to False.
        Raises:
            FileNotFoundError: _description_
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
        
        # load model
        self.model = AutoBackend(weights, self.device, data=data, fp16=fp16)
        self.model.eval()
        
        
    @smart_inference_mode()
    def forward(self, im):
        return self.model(im)
        
        
    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    
    def warmup(self, imgsz=[640, 640]):
        """
        Warm up the model by running one forward pass with a dummy input.
        Args:
            imgsz(list): list of [h,w], default to [640,640]
        Returns:
            (None): This method runs the forward pass and don't return any value
        """
        if isinstance(imgsz, tuple):
            imgsz = list(imgsz)
            
        imgsz = [1,3]+imgsz
        im = torch.empty(*imgsz, dtype=torch.half if self.model.fp16 else torch.float, device=self.device)  # input
        self.forward(im)  # warmup
        
        
    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (np.ndarray): BCHW for tensor, [(HWC) x B] for list.
        """
        if not isinstance(im, np.ndarray):
            raise TypeError(f'Image type {type(im)} not supported')
        
        if im.ndim == 2:
            im = np.cvtColor(im, cv2.COLOR_GRAY2RGB)

        im = np.expand_dims(im,axis=0) # HWC -> BHWC
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        img = self.from_numpy(im)

        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img
    
    
    def load_with_preprocess(self, im_path:str):
        """load image and do im preprocess

        Args:
            im_path (str): the path to the image, could be either .npy, .png, or other image formats
            
        Returns:
            (torch.Tensor): the preprocessed image.
            (np.ndarray): the original image.
        """
        ext = os.path.splitext(im_path)[-1]
        if ext=='.npy':
            im0 = np.load(im_path)
        else:
            im0 = cv2.imread(im_path) #BGR format
            im0 = im0[:,:,::-1] #BGR to RGB
        return self.preprocess(im0),im0
    
    
    def postprocess(self, preds, img, orig_imgs, conf: Union[float, dict], iou=0.45, agnostic=False, max_det=300, return_segments=True):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image
            orig_imgs (np.ndarray | list): Original image or list of original images.
            conf_thres (float | dict): int or dictionary of <class: confidence level>.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.
            return_segments(bool): If True, return the segments of the masks.
        Rreturns:
            (dict): the dictionary contains several keys: boxes, scores, classes, masks, and (masks, segments if use a segmentation model).
                    the shape of boxes is (B, N, 4), where B is the batch size and N is the number of detected objects.
                    the shape of classes and scores are both (B, N).
                    the shape of masks: (B, H, W, 3), where H and W are the height and width of the input image.
        """
        
        if isinstance(preds, (list,tuple)):
            # select only inference output
            predict_mask = True if preds[0].shape[1] != 4+len(self.model.names) else False
        elif isinstance(preds, torch.Tensor):
            predict_mask = False
        else:
            raise TypeError(f'Prediction type {type(preds)} not supported')
        proto = None
        if predict_mask:
            proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
            preds = preds[0]
        
        if isinstance(conf, float):
            conf2 = conf
        elif isinstance(conf, dict):
            conf2 = min(conf.values())
        else:
            raise TypeError(f'Confidence type {type(conf)} not supported')
        preds2 = ops.non_max_suppression(preds,conf2,iou,agnostic=agnostic,max_det=max_det,nc=len(self.model.names))
            
        results = collections.defaultdict(list)
        for i, pred in enumerate(preds2): # pred2: [x1, y1, x2, y2, conf, cls, mask1, mask2 ...]
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            
            if not len(pred):  # skip empty boxes
                continue
            
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            xyxy,confs,clss = pred[:, :4],pred[:, 4],pred[:, 5]
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
            if predict_mask:
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
                masks = masks[M]
                results['masks'].append(masks.cpu().numpy())
                if return_segments:
                    segments = [ops.scale_coords(masks.shape[1:], x, orig_img.shape, normalize=False) 
                                for x in ops.masks2segments(masks)]
                    results['segments'].append(segments)
        return results
