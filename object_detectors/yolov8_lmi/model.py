import cv2
import numpy as np
import torch
import os
import collections
import logging
from typing import Union
import time

from ultralytics.utils import ops
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import smart_inference_mode

# import LMI AI Solutions modules
from od_base import ODBase
import gadget_utils.pipeline_utils as pipeline_utils



class Yolov8(ODBase):
    
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
        
        # class map < id: class name >
        self.names = self.model.names
        
        
    @smart_inference_mode()
    def forward(self, im):
        return self.model(im)
        
        
    @smart_inference_mode()
    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    
    @smart_inference_mode()
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
        
        
    @smart_inference_mode()
    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (np.ndarray): BCHW for tensor, [(HWC) x B] for list.
        """
        if not isinstance(im, np.ndarray):
            raise TypeError(f'Image type {type(im)} not supported')
        
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

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
    

    def get_min_conf(self, conf:Union[float, dict]):
        """Get the minimum confidence level for non-maximum suppression.

        Args:
            conf (float | dict): int or dictionary of <class: confidence level>.
        """
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
        return conf2
    
    
    @smart_inference_mode()
    def postprocess(self, preds, img, orig_imgs, conf: Union[float, dict], iou=0.45, agnostic=False, max_det=300, return_segments=True):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image
            orig_imgs (np.ndarray | list): Original image or list of original images.
            conf (float | dict): int or dictionary of <class: confidence level>.
            iou (float): The IoU threshold below which boxes will be filtered out during NMS.
            max_det (int): The maximum number of detections to return. defaults to 300.
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
        
        conf2 = self.get_min_conf(conf)
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


    @smart_inference_mode()
    def predict(self, image, configs, operators=[], iou=0.4, agnostic=False, max_det=300, return_segments=True):
        """run yolov8 object detection inference. It runs the preprocess(), forward(), and postprocess() in sequence.
        It converts the results to the original coordinates space if the operators are provided.
        
        Args:
            model (Yolov8): the object detection model loaded memory
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
        results = self.postprocess(pred,im,image,configs,iou,agnostic,max_det,return_segments)
        
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
            masks = pipeline_utils.revert_masks_to_origin(masks, operators)
            results_dict['masks'] = masks
        if return_segments and len(results['segments']):
            segs = results['segments'][0]
            result_contours = [pipeline_utils.revert_to_origin(seg, operators) for seg in segs]
            results_dict['segments'] = result_contours
        
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
        boxes = results['boxes']
        classes = results['classes']
        scores = results['scores']
        masks = results['masks']
        
        
        image2 = image.copy()
        if not len(boxes):
            return image2
        
        for i in range(len(boxes)):
            mask = masks[i] if len(masks) else None
            pipeline_utils.plot_one_box(
                boxes[i],
                image2,
                mask,
                label="{}: {:.2f}".format(
                    classes[i], scores[i]
                ),
                color=colormap[classes[i]] if colormap is not None else None,
            )
        return image2



class Yolov8Obb(Yolov8):
    def __init__(self, weights:str, device='gpu', data=None, fp16=False) -> None:
        super().__init__(weights, device, data, fp16)
        self.logger = logging.getLogger(__name__)
        
    @smart_inference_mode()
    def postprocess(self, preds, img, orig_imgs, conf: Union[float, dict], iou=0.45, agnostic=False, max_det=300, return_segments=True):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image
            orig_imgs (np.ndarray | list): Original image or list of original images.
            conf_thres (float | dict): int or dictionary of <class: confidence level>.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            max_det (int): The maximum number of detections to return. defaults to 300.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.
            return_segments(bool): If True, return the segments of the masks.
        Rreturns:
            (dict): the dictionary contains several keys: boxes, scores, classes, masks, and (masks, segments if use a segmentation model).
                    the shape of boxes is (B, N, 4), where B is the batch size and N is the number of detected objects.
                    the shape of classes and scores are both (B, N).
                    the shape of masks: (B, H, W, 3), where H and W are the height and width of the input image.
        """
        
        # check the datatype of the predictions
        if isinstance(preds, torch.Tensor) != True and isinstance(preds, list) != True:
            self.logger.error(f'Prediction type {type(preds)} not supported expected torch.Tensor or list')
            raise TypeError(f'Prediction type {type(preds)} not supported expected torch.Tensor or list')
        
        # run non-max suppression in xywhr format
        conf2 = self.get_min_conf(conf)
        preds2 = ops.non_max_suppression(preds,conf2,iou,agnostic=agnostic,max_det=max_det,nc=len(self.model.names), rotated=True)
        
        # create a collections dictionary to store the results
        results = collections.defaultdict(list)
        
        for i, pred in enumerate(preds2):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            
            if not len(pred):  # skip empty boxes
                continue
            
            # makes sure to regularize the bounding boxes to xywhr format (range [0, pi/2])
            bboxs = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            # scale the bounding boxes to original image size
            bboxs[:,:4] = ops.scale_boxes(img.shape[2:], bboxs[:, :4], orig_img.shape, xywh=True)

            # get the confidence, class
            confs, clss = pred[:, 4], pred[:, 5]
            # get the class names for the predictions
            classes = np.array([self.model.names[c.item()] for c in clss])
        
            # covert the boxes from xywhr xyxyxyxy format
            bboxs = ops.xywhr2xyxyxyxy(bboxs)
            # filter based on confidence
            if isinstance(conf, float):
                thres = np.array([conf]*len(clss))
            if isinstance(conf, dict):
                # set to 1 if c is not in conf
                thres = np.array([conf.get(c,1) for c in classes])
            
            # filter based on confidence
            M = confs > self.from_numpy(thres)
            
            # append the results boxes, scores, classes
            results['boxes'].append(bboxs[M].cpu().numpy())
            results['scores'].append(confs[M].cpu().numpy())
            results['classes'].append(classes[M.cpu().numpy()])
        return results
    
    @smart_inference_mode()
    def predict(self, image, configs, operators=[], iou=0.4, agnostic=False, max_det=300, return_segments=True):
        """run yolov8 object detection inference. It runs the preprocess(), forward(), and postprocess() in sequence.
        It converts the results to the original coordinates space if the operators are provided.
        
        Args:
            model (Yolov8): the object detection model loaded memory
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
        results = self.postprocess(pred,im,image,configs,iou,agnostic,max_det,return_segments)
        
        # return empty results if no detection
        results_dict = collections.defaultdict(list)
        if not len(results['boxes']):
            time_info['postproc'] = time.time()-t0
            return results_dict, time_info
        
        # handling only one batch
        boxes = results['boxes']
        scores = results['scores'][0].tolist()
        classes = results['classes'][0].tolist()
        
        # convert box to sensor space
        # boxes = [pipeline_utils.revert_to_origin(box, operators) for box in boxes]
        converted_boxes = []
        for box in boxes:
            b = []
            for i in range(len(box)):
                b.append(pipeline_utils.revert_to_origin(box[i], operators))
            converted_boxes.append(b)
        
        results_dict['boxes'] = converted_boxes
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
        boxes = results['boxes']
        classes = results['classes']
        scores = results['scores']

        image2 = image.copy()
        if not len(boxes):
            return image2
        
        for i in range(len(boxes)):
            label = "{}: {:.2f}".format(classes[i], scores[i])
            pipeline_utils.plot_one_rbox(
                boxes[i],
                image2,
                label=label,
                color=colormap[classes[i]] if colormap is not None else None,
            )
        return image2
    


class Yolov8Pose(Yolov8):
    def __init__(self, weights:str, device='gpu', data=None, fp16=False) -> None:
        super().__init__(weights, device, data, fp16)
        
    @smart_inference_mode()
    def postprocess(self, preds, img, orig_imgs, conf: Union[float, dict], iou=0.45, agnostic=False, max_det=300):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image
            orig_imgs (np.ndarray | list): Original image or list of original images.
            conf_thres (float | dict): int or dictionary of <class: confidence>
            iou (float): The IoU threshold below which boxes will be filtered out during NMS.
            max_det (int): The maximum number of detections to return. defaults to 300.
        """
        
        conf2 = self.get_min_conf(conf)
        preds2 = ops.non_max_suppression(preds,conf2,iou,agnostic=agnostic,max_det=max_det,nc=len(self.model.names))
            
        results = collections.defaultdict(list)
        for i, pred in enumerate(preds2): # pred2: [x1, y1, x2, y2, conf, cls, ...]
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            
            if not len(pred):  # skip empty boxes
                continue
            
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            xyxy,confs,clss = pred[:, :4],pred[:, 4],pred[:, 5]
            classes = np.array([self.model.names[c.item()] for c in clss])
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)

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
            results['points'].append(pred_kpts[M].cpu().numpy()) # [n_obj,n_kp,2]
        return results
    
    
    @smart_inference_mode()
    def predict(self, image, configs, operators=[], iou=0.4, agnostic=False, max_det=300):
        """run yolov8 object detection inference. It runs the preprocess(), forward(), and postprocess() in sequence.
        It converts the results to the original coordinates space if the operators are provided.
        
        Args:
            model (Yolov8): the object detection model loaded memory
            image (np.ndarry): the input image
            configs (dict): a dictionary of the confidence thresholds for each class, e.g., {'classA':0.5, 'classB':0.6}
            operators (list): a list of dictionaries of the image preprocess operators, such as {'resize':[resized_w, resized_h, orig_w, orig_h]}, {'pad':[pad_left, pad_right, pad_top, pad_bot]}
            iou (float): the iou threshold for non-maximum suppression. defaults to 0.4
            agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.
            max_det (int): The maximum number of detections to return. defaults to 300.

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
        results = self.postprocess(pred,im,image,configs,iou,agnostic,max_det)
        
        # return empty results if no detection
        results_dict = collections.defaultdict(list)
        if not len(results['boxes']):
            time_info['postproc'] = time.time()-t0
            return results_dict, time_info
        
        # only one image, get first batch
        boxes = results['boxes'][0]
        scores = results['scores'][0].tolist()
        classes = results['classes'][0].tolist()
        points = results['points'][0].tolist()
        
        # convert box to sensor space
        points = [pipeline_utils.revert_to_origin(p, operators) for p in points] # each iter: [n_kp,2]
        boxes = pipeline_utils.revert_to_origin(boxes, operators)
        results_dict['points'] = points
        results_dict['boxes'] = boxes
        results_dict['scores'] = scores
        results_dict['classes'] = classes
            
        time_info['postproc'] = time.time()-t0
        return results_dict, time_info
    
    
    @staticmethod
    def annotate_image(results, image, colormap=None, kp_color=(255,255,255)):
        """annotate the object dectector results on the image. If colormap is None, it will use the random colors.
        TODO: text size, thickness, font

        Args:
            results (dict): the results of the object detection, e.g., {'boxes':[], 'classes':[], 'scores':[], 'masks':[], 'segments':[]}
            image (np.ndarray): the input image
            colors (list, optional): a dictionary of colormaps, e.g., {'class_A':(0,0,255), 'class_B':(0,255,0)}. Defaults to None.

        Returns:
            np.ndarray: the annotated image
        """
        boxes = results['boxes']
        classes = results['classes']
        scores = results['scores']
        points = results['points']
        
        
        image2 = image.copy()
        if not len(boxes):
            return image2
        
        for i in range(len(boxes)):
            pipeline_utils.plot_one_box(
                boxes[i],
                image2,
                label="{}: {:.2f}".format(
                    classes[i], scores[i]
                ),
                color=colormap[classes[i]] if colormap is not None else None,
            )
            
        for i in range(len(points)):
            for j in range(len(points[i])):
                cv2.circle(image2, (int(points[i][j][0]), int(points[i][j][1])), 4, kp_color, -1)
        return image2