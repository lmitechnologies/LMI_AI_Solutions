import cv2
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from typing import Union
from PIL import Image

from ultralytics.utils import ops
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.augment import classify_transforms
from ultralytics.utils.torch_utils import smart_inference_mode

from yolov8_lmi.model import Yolov8


class Yolov8_cls(Yolov8):
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, weights:str, device='gpu', data=None, fp16=False, imgsz=[224,224], crop_fraction=1) -> None:
        """init the model

        Args:
            weights (str): the path to the weights file.
            device (str, optional): _description_. Defaults to 'gpu'.
            data (str, optional): the path to dataset yaml file. Defaults to None.
            fp16 (bool, optional): use fp16 precision. Defaults to False.
            imgsz (list, optional): input image size [h,w]. Defaults to [224,224].
            crop_fraction(float, optional): crop fraction. Defaults to 1.

        Raises:
            FileNotFoundError: _description_
        """
        super().__init__(weights, device, data, fp16)
        
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(imgsz[0], crop_fraction=crop_fraction),
            )
        )
        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"
        
        
    @smart_inference_mode()
    def preprocess(self, img):
        """Prepares input image before inference.
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/predict.py#L36
        """
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.expand_dims(img, 0)
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack(
                    [self.transforms(Image.fromarray(im)) for im in img], dim=0 # not convert from BGR to RGB
                )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
    
    
    @smart_inference_mode()
    def postprocess(self, preds):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            
        """
        
        results = defaultdict(list)
        for pred in preds:
            pred = pred.cpu().numpy()
            idx = pred.argmax()
            results['scores'].append(pred[idx].item())
            results['classes'].append(self.model.names[idx])
        return results
