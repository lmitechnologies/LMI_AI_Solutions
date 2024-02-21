from collections import OrderedDict, namedtuple
import cv2
import logging
import os
import numpy as np
import torch
import tensorrt as trt
import sys
from typing import Union
import collections

# add yolov5 submodule to the path
YOLO_PATH = os.path.join(os.path.dirname(__file__), '../submodules/yolov5')
print(YOLO_PATH)
if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(f'Cannot find yolov5 submodule at {YOLO_PATH}')
sys.path.insert(0, YOLO_PATH)

from utils.general import non_max_suppression, scale_boxes, scale_segments, yaml_load
from utils.segment.general import masks2segments, process_mask, process_mask_native


class Yolov5:
    logger = logging.getLogger('ENGINE')
    
    
    def __init__(self, path_wts:str, data=None) -> None:
        """
        args:
            path_wts(str): the path to the tensorRT engine file
            data (str, optional): the path to the yaml file containing class names. Defaults to None.
        """
        if not torch.cuda.is_available():
            raise EnvironmentError('CUDA not available')
        device = torch.device("cuda:0")
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger_trt = trt.Logger(trt.Logger.INFO)
        with open(path_wts, "rb") as f, trt.Runtime(logger_trt) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        imgsz = list(bindings["images"].shape[2:])
        
        # class names
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"{i}" for i in range(999)}

        self.__dict__.update(locals())  # assign all variables to self


    def forward(self, im):
        if self.fp16:
            im = im.half()
        if self.dynamic and im.shape != self.bindings['images'].shape:
            i = self.model.get_binding_index('images')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(i)))
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
        
        
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    
    def warmup(self, imgsz=(640, 640)):
        """
        warm up the model once
        Args:
            imgsz (tuple, optional): list of [H,W] format. Defaults to (640, 640).
        """
        if isinstance(imgsz, tuple):
            imgsz = list(imgsz)
        if self.imgsz and self.imgsz != imgsz:
            raise Exception(f'The warmup imgsz of {imgsz} does not match with the size of the model: {self.imgsz}!')
        
        imgsz = [1,3]+imgsz
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        self.forward(im)
    
    
    def preprocess(self, im):
        """im preprocess
            normalization -> CHW -> contiguous -> BCHW
        Args:
            im0 (numpy.ndarray): the input numpy array, HWC format
        """
        if not isinstance(im, np.ndarray):
            raise TypeError(f'Image type {type(im)} not supported')
        
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
    
    
    def postprocess(self,preds,im,im0,conf: Union[float, dict],iou_thres=0.45,agnostic=False,max_det=100,return_segments=True):
        """
        Args:
            preds (list): a list of object detection predictions
            im (tensor): the preprocessed image
            im0 (np.ndarray): the original image
            conf (dict): confidence threshold dict <class_id: confidence>.
            proto (tensor, optional): mask predictions. Defaults to None.
            iou_thres (float, optional): iou threshold. Defaults to 0.45.
            max_det (int, optional): the max number of detections. Defaults to 100.
            agnostic (bool, optional): perform class-agnostic NMS. Defaults to False.
        """
        if isinstance(preds, (list,tuple)):
            # select only inference output
            predict_mask = True
        elif isinstance(preds, torch.Tensor):
            predict_mask = False
        else:
            raise TypeError(f'Prediction type {type(preds)} not supported')
        
        proto = None
        nm = 0
        if predict_mask:
            preds,proto = preds[0], preds[1]
            nm = 32
        
        # get lowest confidence
        if isinstance(conf, float):
            conf2 = conf
        elif isinstance(conf, dict):
            conf2 = min(conf.values())
        else:
            raise TypeError(f'Confidence type {type(conf)} not supported')
        
        # Process predictions
        pred = non_max_suppression(preds,conf2,iou_thres,agnostic=agnostic,max_det=max_det, nm=nm)
        results = collections.defaultdict(list)
        for i,det in enumerate(pred):  # per image
            if len(det)==0:
                continue
            
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            xyxy,confs,clss = det[:, :4],det[:, 4],det[:, 5]
            classes = np.array([self.names[c.item()] for c in clss])
            
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
                masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[2:])
                masks = masks[M]
                results['masks'].append(masks.cpu().numpy())
                if return_segments:
                    segs = [scale_segments(im.shape[2:], x, im0.shape, normalize=False)
                            for x in reversed(masks2segments(masks))]
                    results['segments'].append(segs)
        return results
