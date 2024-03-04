import cv2
import numpy as np
import torch
import tensorrt as trt
import json
import os
from collections import OrderedDict,namedtuple,defaultdict
import logging
from enum import Enum
from typing import Union

from ultralytics.utils import ops
from ultralytics.nn.tasks import attempt_load_weights


class FileType(str, Enum):
    ENGINE = '.engine'
    PT = '.pt'


def get_file_type(path_ws):
    _,ext = os.path.splitext(path_ws)
    for mt in FileType:
        if mt.value == ext:
            return mt
    raise TypeError(f'All supported weights files are ".engine" and ".pt". But found this file extension: {ext}')


def check_class_names(names):
    """Check class names. Map imagenet class codes to human-readable names if required. Convert lists to dicts."""
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(f'{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices '
                           f'{min(names.keys())}-{max(names.keys())} defined in your dataset YAML.')
    return names


class Yolov8:
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, path_wts:str, device='gpu') -> None:
        if not os.path.isfile(path_wts):
            raise FileNotFoundError(f'File not found: {path_wts}')
        if device=='gpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA not available.')

        device = torch.device('cuda:0') if device=='gpu' else torch.device('cpu')
        stride = 32  # default stride
        fp16 = False  # default updated below
        model, metadata = None, None
        
        # get the type of file
        file_type = get_file_type(path_wts)
        self.logger.info(f'found weights file type: {file_type}')
        if file_type == FileType.ENGINE:
            logger_trt = trt.Logger(trt.Logger.INFO)
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            # Read file
            with open(path_wts, 'rb') as f, trt.Runtime(logger_trt) as runtime:
                meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
                metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
                model = runtime.deserialize_cuda_engine(f.read())  # read engine
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
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
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
            
            # load metadata
            if metadata:
                for k, v in metadata.items():
                    if k in ('stride', 'batch'):
                        metadata[k] = int(v)
                    elif k in ('imgsz', 'names', 'kpt_shape') and isinstance(v, str):
                        metadata[k] = eval(v)
                stride = metadata['stride']
                task = metadata['task']
                batch = metadata['batch']
                imgsz = metadata['imgsz']
                names = metadata['names']
                self.logger.info(f'engine class names: {names}')
                self.logger.info(f'engine imgsz: {imgsz}')
                kpt_shape = metadata.get('kpt_shape')
            else:
                self.logger.warning(f"WARNING ⚠️ Metadata not found for 'model={path_wts}'")
        elif file_type == FileType.PT:
            imgsz = []
            model = attempt_load_weights(path_wts,device=device,inplace=True,fuse=True)
            if hasattr(model, 'kpt_shape'):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), stride)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            self.logger.info(f'pt class names: {names}')
            model.half() if fp16 else model.float()
            
        # Check names
        names = check_class_names(names)
        self.__dict__.update(locals())  # assign all variables to self
        
        
    def forward(self, im):
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
            
        if self.file_type == FileType.ENGINE:
            if self.dynamic and im.shape != self.bindings['images'].shape:
                self.logger.warning('WARNING ⚠️ Input image size mismatch, attempting to resize')
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        if self.file_type == FileType.PT:
            y = self.model(im)
        
        # debug shapes
        # for i,x in enumerate(y):
        #     if isinstance(x, (list, tuple)):
        #         print(f'{i}: list')
        #         for xx in x:
        #             print(type(xx), xx.shape)
        #     else:
        #         print(f'{i}: ', type(x), x.shape)  
        
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
        
        
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
        if self.imgsz and self.imgsz != imgsz:
            raise Exception(f'The warmup imgsz of {imgsz} does not match with the size of the model: {self.imgsz}!')
            
        imgsz = [1,3]+imgsz
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        self.forward(im)  # warmup
        
        
    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (np.ndarray): BCHW for tensor, [(HWC) x B] for list.
        """
        if not isinstance(im, np.ndarray):
            raise TypeError(f'Image type {type(im)} not supported')

        im = np.expand_dims(im,axis=0) # HWC -> BHWC
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        img = self.from_numpy(im)

        img = img.half() if self.fp16 else img.float()  # uint8 to fp16/32
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
            predict_mask = True if preds[0].shape[1] != 4+len(self.names) else False
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
        preds2 = ops.non_max_suppression(preds,conf2,iou,agnostic=agnostic,max_det=max_det,nc=len(self.names))
            
        results = defaultdict(list)
        for i, pred in enumerate(preds2): # pred2: [x1, y1, x2, y2, conf, cls, mask1, mask2 ...]
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            
            if not len(pred):  # skip empty boxes
                continue
            
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            xyxy,confs,clss = pred[:, :4],pred[:, 4],pred[:, 5]
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
            if predict_mask:
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
                masks = masks[M]
                results['masks'].append(masks.cpu().numpy())
                if return_segments:
                    segments = [ops.scale_coords(masks.shape[1:], x, orig_img.shape, normalize=False) 
                                for x in ops.masks2segments(masks)]
                    results['segments'].append(segments)
        return results
