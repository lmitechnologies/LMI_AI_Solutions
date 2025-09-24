from functools import lru_cache
import cv2
import numpy as np
import torch
import os
import collections
import logging
from typing import Dict, Union, List
import time

from ultralytics.utils import nms, ops
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import smart_inference_mode

# import LMI AI Solutions modules
from od_core.od_base import ODBase
from od_core.object_detector_registry import ObjectDetectorRegistry
import gadget_utils.pipeline_utils as pipeline_utils



@smart_inference_mode()
def to_numpy(data):
    """Converts a tensor or a list to numpy arrays.

    Args:
        data (torch.Tensor | list): The input tensor or list of tensors.

    Returns:
        (np.ndarray): The converted numpy array.
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f'Data type {type(data)} not supported')


@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=['v1'],
        model_names=['yolo', 'yolov8', 'yolov11'],
        tasks=['od', 'seg', 'instancesegmentation', 'objectdetection'],
        frameworks=['ultralytics', 'ultralytics8'],
    )
)
class Yolo(ODBase):
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, model_path:str, device='gpu', data=None, fp16=False,**kwargs) -> None:
        """init the model
        Args:
            model_path (str): the path to the model_path file.
            device (str, optional): the device to be used, either 'gpu' or 'cpu'. Defaults to 'gpu'.
            data (str, optional): the path to dataset yaml file. Defaults to None.
            fp16 (bool, optional): Whether to use fp16. Defaults to False.
        Raises:
            FileNotFoundError: _description_
        """
        self.image_size = kwargs.get('image_size', [640, 640])
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'File not found: {model_path}')
        
        self._setup_device(device)
        
        # load model
        self.model = AutoBackend(model_path, self.device, data=data, fp16=fp16)
        self.model.eval()
        
        # class map < id: class name >
        self.names = self.model.names
        
        
    def _setup_device(self, device):
        """set up the computation device (CPU or GPU).

        Args:
            device (str): The device to be used, either 'cpu' or 'gpu'.
        """
        if device.lower() not in ['cpu', 'gpu']:
            raise ValueError(f'Invalid device: {device}. Supported devices are "cpu" and "gpu".')
        
        self.device = torch.device('cpu')
        if device.lower() == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.logger.warning('GPU not available, falling back to CPU')
        
        
    @smart_inference_mode()
    def forward(self, im: torch.Tensor):
        return self.model(im)
        
        
    @smart_inference_mode()
    def from_numpy(self, x: np.ndarray) -> torch.Tensor:
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    
    @smart_inference_mode()
    def warmup(self, imgsz=None):
        """
        Warm up the model by running one forward pass with a dummy input.
        Args:
            imgsz(list): list of [h,w], default to None
        Returns:
            (None): This method runs the forward pass and don't return any value
        """
        if imgsz is None:
            imgsz = self.image_size
            
        if isinstance(imgsz, tuple):
            imgsz = list(imgsz)
        
            
        imgsz = [1,3]+imgsz
        im = torch.empty(*imgsz, dtype=torch.half if self.model.fp16 else torch.float, device=self.device)  # input
        self.forward(im)  # warmup
        
        
    @smart_inference_mode()
    def preprocess(self, im: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Prepares input image before inference.

        Args:
            im (np.ndarray | tensor): BCHW for tensor, [(HWC) x B] for list.
        """
        if isinstance(im, np.ndarray):
            im = self.from_numpy(im)
        
        im = im.to(self.device)
        # convert to HWC
        if im.ndim == 2:
            im = im.unsqueeze(-1)
        if im.shape[-1] ==1:
            im = im.expand(-1,-1,3)
            
        im = im.unsqueeze(0) # HWC -> BHWC
        img = im.permute((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        img = img.contiguous()

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
        return self.preprocess(im0.copy()),im0
    

    def _get_min_conf(self, conf:Union[float, dict]) -> float:
        """Get the minimum confidence level for non-maximum suppression.

        Args:
            conf (float | dict): int or dictionary of <class: confidence level>.
        """
        if isinstance(conf, float):
            min_conf = conf
        elif isinstance(conf, dict):
            min_conf = 1
            class_names = set(self.model.names.values())
            for k,v in conf.items():
                if k in class_names:
                    min_conf = min(min_conf, v)
            if min_conf == 1:
                self.logger.warning('No class matches in confidence dict, set to 1.0 for all classes.')
        else:
            raise TypeError(f'Confidence type {type(conf)} not supported')
        return min_conf
    
    
    @smart_inference_mode()
    def _get_thresholds(self, conf:Union[float, dict], num_preds:int, classes:list) -> torch.Tensor:
        """Get the thresholds for each class.

        Args:
            conf (float | dict): int or dictionary of <class: confidence level>.
            num_preds (int): the number of predictions.
            classes (list): the list of class names for each prediction.
            
        Returns:
            (torch.Tensor): the thresholds for each prediction.
        """
        if isinstance(conf, float):
            thres = np.array([conf]*num_preds)
        elif isinstance(conf, dict):
            # set to 1 if c is not in conf
            thres = np.array([conf.get(c,1) for c in classes])
        else:
            raise TypeError(f'Confidence type {type(conf)} not supported')
        return self.from_numpy(thres)
    
    
    @lru_cache(maxsize=1)
    def to_segments(self, masks: torch.Tensor, img_shape: tuple) -> List[np.ndarray]:
        """Convert masks to segments.

        Args:
            masks (torch.Tensor): the masks to be converted, shape (n, h, w)
            img_shape (tuple): the shape of the image, (h, w, c)
            
        Returns:
            (list): a list of segments, each segment is a numpy array of shape (n, 2)
        """
        segments = [ops.scale_coords(masks.shape[1:], x, img_shape, normalize=False) 
                    for x in ops.masks2segments(masks)]
        return segments
    
    
    @smart_inference_mode()
    def postprocess(self, preds, img, orig_imgs, conf: Union[float, dict], iou=0.45, agnostic=False, max_det=300, return_segments=True):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image
            orig_imgs (np.ndarray | torch.Tensor | list): Original image or list of original images. If this is a tensor or a list of tensors, this function will return tensor results.
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
                    the shape of segments: [ (n1,2), (n2,2), ...]
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
            proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
            preds = preds[0]
        
        min_conf = self._get_min_conf(conf)
        preds2 = nms.non_max_suppression(preds,min_conf,iou,agnostic=agnostic,max_det=max_det,nc=len(self.model.names))
            
        results = collections.defaultdict(list)
        for i, pred in enumerate(preds2): # pred2: [x1, y1, x2, y2, conf, cls, mask1, mask2 ...]
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            return_tensor = isinstance(orig_img, torch.Tensor)
            
            if not len(pred):  # skip empty boxes
                continue
            
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            xyxy,confs,clss = pred[:, :4],pred[:, 4],pred[:, 5]
            classes = np.array([self.model.names[c.item()] for c in clss])
            
            # filter based on conf
            thres = self._get_thresholds(conf, len(clss), classes)
            M = confs > thres
            
            if predict_mask:
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
                masks = masks[M]
                results['masks'].append(masks if return_tensor else masks.cpu().numpy())
                if return_segments:
                    segments = self.to_segments(masks, orig_img.shape)
                    if return_tensor:
                        segments = [self.from_numpy(x) for x in segments]   # list of [ (n1,2), (n2,2), ... ]
                    results['segments'].append(segments)
                    
            if return_tensor:
                results['boxes'].append(xyxy[M])
                results['scores'].append(confs[M])
            else:
                results['boxes'].append(xyxy[M].cpu().numpy())
                results['scores'].append(confs[M].cpu().numpy())
            results['classes'].append(classes[M.cpu().numpy()].tolist())
        return results
    
    
    def _revert_coordinates(self, results: Dict, operators: List[Dict]) -> Dict:
        """Reverts prediction coordinates to the original pre-transform space."""
        if not operators or not len(results['boxes']):
            return results

        # Assumes single-image batch processing from predict()
        results['boxes'] = pipeline_utils.revert_to_origin(results['boxes'], operators)
        if 'masks' in results:
            results['masks'] = pipeline_utils.revert_masks_to_origin(results['masks'], operators)
        if 'segments' in results:
            results['segments'] = [pipeline_utils.revert_to_origin(seg, operators) for seg in results['segments']]
        
        return results


    @smart_inference_mode()
    def predict(self, image, configs, operators=[], iou=0.4, agnostic=False, max_det=300, return_segments=True, **kwargs):
        """run Yolo object detection inference. It runs the preprocess(), forward(), and postprocess() in sequence.
        It converts the results to the original coordinates space if the operators are provided.
        
        Args:
            model (Yolo): the object detection model loaded memory
            image (np.ndarry | tensor): the input image
            configs (dict | float): a float or a dictionary of the confidence thresholds for each class, e.g., {'classA':0.5, 'classB':0.6}
            operators (list): a list of dictionaries of the image preprocess operators, such as {'resize':[resized_w, resized_h, orig_w, orig_h]}, {'pad':[pad_left, pad_right, pad_top, pad_bot]}
            iou (float): the iou threshold for non-maximum suppression. defaults to 0.4
            agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.
            max_det (int): The maximum number of detections to return. defaults to 300.
            return_segments(bool): If True, return the segments of the masks. Defaults to True.
        Returns:
            list of [results, time info]
            results (dict): a dictionary of the results, e.g., {
                'boxes': numpy or tensor
                'classes': a list of strings (NOT tensor)
                'scores': numpy or tensor
                'masks': numpy or tensor
                'segments': numpy or tensor
            }
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
        post_args = {
            'conf': configs,
            'iou': iou,
            'agnostic': agnostic,
            'max_det': max_det,
            'return_segments': return_segments
        }
        results = self.postprocess(pred,im,image,**post_args)
        results_dict = collections.defaultdict(list)
        
        # return empty results if no detection
        if not len(results['boxes']):
            time_info['postproc'] = time.time()-t0
            return results_dict, time_info

        # Extract results for single image
        for k,v in results.items():
            results_dict[k] = v[0]
        
        # Revert coordinates if needed
        results_dict = self._revert_coordinates(results_dict, operators)
        time_info['postproc'] = time.time()-t0
        
        return results_dict, time_info
    
    
    @staticmethod
    @smart_inference_mode()
    def annotate_image(results, image, colormap=None, line_thickness=None, hide_label=False, hide_bbox=False):
        """annotate the object dectector results on the image. If colormap is None, it will use the random colors.

        Args:
            results (dict): the results of the object detection, e.g., {'boxes':[], 'classes':[], 'scores':[], 'masks':[], 'segments':[]}
            image (np.ndarray): the input image
            colors (list, optional): a dictionary of colormaps, e.g., {'class-A':(0,0,255), 'class-B':(0,255,0)}. Defaults to None.
            line_thickness (int, optional): the thickness of the bounding box. Defaults to None.
            hide_bbox (bool,optional): hide the bounding box
        Returns:
            np.ndarray: the annotated image
        """
        boxes = results['boxes']
        classes = results['classes']
        scores = results['scores']
        masks = results['masks']
        points = results['points']
        
        image = to_numpy(image).copy()
        if not len(boxes):
            return image
        
        # convert to numpy
        boxes = to_numpy(boxes)
        points = to_numpy(points)
        if len(masks):
            masks = to_numpy(masks)
        
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # plot boxes and masks
        for i in range(len(boxes)):
            label = "{}: {:.2f}".format(classes[i], scores[i])
            args = {
                'label': None if hide_label else label, 
                'color': None if colormap is None else colormap[classes[i]], 
                'line_thickness':line_thickness, 
                'hide_bbox':hide_bbox
                }
            
            if boxes[i].shape == (4,2):
                pipeline_utils.plot_one_rbox(boxes[i],image,**args)
            elif boxes[i].shape == (4,):
                mask = masks[i] if len(masks) else None
                pipeline_utils.plot_one_box(boxes[i],image,mask,**args)
                
        # plot keypoints
        points = points.astype(int)
        for i in range(len(points)):
            for j in range(len(points[i])):
                cv2.circle(image, (points[i][j][0], points[i][j][1]), 4, (255,255,255), -1)
                
        return image


@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=['v1'], 
        model_names=['yolo','yolov8', 'yolov11'], 
        tasks=['obb'], 
        frameworks=['ultralytics', 'ultralytics8']
    )
)
class YoloObb(Yolo):
    def __init__(self, model_path:str, device='gpu', data=None, fp16=False, **kwargs) -> None:
        super().__init__(model_path, device, data, fp16)
        self.logger = logging.getLogger(__name__)
        
    @smart_inference_mode()
    def postprocess(self, preds, img, orig_imgs, conf: Union[float, dict], iou=0.45, agnostic=False, max_det=300, **kwargs):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image
            orig_imgs (np.ndarray | torch.Tensor | list): Original image or list of original images. If this is a tensor or a list of tensors, this function will return tensor results.
            conf_thres (float | dict): int or dictionary of <class: confidence level>.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            max_det (int): The maximum number of detections to return. defaults to 300.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.
        Rreturns:
            (dict): the dictionary contains several keys: boxes, scores, classes, masks, and (masks, segments if use a segmentation model).
                    the shape of boxes is (B, N, 4), where B is the batch size and N is the number of detected objects.
                    the shape of classes and scores are both (B, N).
                    the shape of masks: (B, H, W, 3), where H and W are the height and width of the input image.
        """
        # run non-max suppression in xywhr format
        min_conf = self._get_min_conf(conf)
        preds2 = nms.non_max_suppression(preds,min_conf,iou,agnostic=agnostic,max_det=max_det,nc=len(self.model.names), rotated=True)
        
        results = collections.defaultdict(list)
        for i, pred in enumerate(preds2):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            return_tensor = isinstance(orig_img, torch.Tensor)
            
            if not len(pred):  # skip empty boxes
                continue
            
            # makes sure to regularize the bounding boxes to xywhr format (range [0, pi/2])
            bboxs = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            bboxs[:,:4] = ops.scale_boxes(img.shape[2:], bboxs[:, :4], orig_img.shape, xywh=True)
            confs, clss = pred[:, 4], pred[:, 5]
            classes = np.array([self.model.names[c.item()] for c in clss])
        
            # covert the boxes from xywhr xyxyxyxy format
            bboxs = ops.xywhr2xyxyxyxy(bboxs)
            
            # filter based on conf
            thres = self._get_thresholds(conf, len(clss), classes)
            M = confs > thres
            
            # append the results boxes, scores, classes
            if return_tensor:
                results['boxes'].append(bboxs[M])   # [n_obj, 4, 2]
                results['scores'].append(confs[M])
            else:
                results['boxes'].append(bboxs[M].cpu().numpy())
                results['scores'].append(confs[M].cpu().numpy())
            results['classes'].append(classes[M.cpu().numpy()].tolist())
        return results
    
    
    def _revert_coordinates(self, results: Dict, operators: List[Dict]) -> Dict:
        """Reverts OBB coordinates to the original pre-transform space."""
        if not operators or not len(results['boxes']):
            return results
        
        boxes = results['boxes']
        reverted_boxes = [pipeline_utils.revert_to_origin(box, operators) for box in boxes]
        results['boxes'] = torch.stack(reverted_boxes) if isinstance(boxes, torch.Tensor) else np.array(reverted_boxes)
        return results
    

@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=['v1'], 
        model_names=['yolo','yolov8', 'yolov11'], 
        tasks=['pose'], 
        frameworks=['ultralytics', 'ultralytics8']
    )
)
class YoloPose(Yolo):
    def __init__(self, model_path:str, device='gpu', data=None, fp16=False, **kwargs) -> None:
        super().__init__(model_path, device, data, fp16)
        
        
    @smart_inference_mode()
    def postprocess(self, preds, img, orig_imgs, conf: Union[float, dict], iou=0.45, agnostic=False, max_det=300, **kwargs):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image
            orig_imgs (np.ndarray | torch.Tensor | list): Original image or list of original images. If this is a tensor or a list of tensors, this function will return tensor results.
            conf_thres (float | dict): int or dictionary of <class: confidence>
            iou (float): The IoU threshold below which boxes will be filtered out during NMS.
            max_det (int): The maximum number of detections to return. defaults to 300.
        """
        min_conf = self._get_min_conf(conf)
        preds2 = nms.non_max_suppression(preds,min_conf,iou,agnostic=agnostic,max_det=max_det,nc=len(self.model.names))
            
        results = collections.defaultdict(list)
        for i, pred in enumerate(preds2): # pred2: [x1, y1, x2, y2, conf, cls, ...]
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            return_tensor = isinstance(orig_img, torch.Tensor)
            
            if not len(pred):  # skip empty boxes
                continue
            
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            xyxy,confs,clss = pred[:, :4],pred[:, 4],pred[:, 5]
            classes = np.array([self.model.names[c.item()] for c in clss])
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)

            # filter based on conf
            thres = self._get_thresholds(conf, len(clss), classes)
            M = confs > thres
            
            if return_tensor:
                results['boxes'].append(xyxy[M])
                results['scores'].append(confs[M])
                results['points'].append(pred_kpts[M])
            else:
                results['boxes'].append(xyxy[M].cpu().numpy())
                results['scores'].append(confs[M].cpu().numpy())
                results['points'].append(pred_kpts[M].cpu().numpy()) # [n_obj,n_kp,3]
            results['classes'].append(classes[M.cpu().numpy()].tolist())
        return results
    
    
    def _revert_coordinates(self, results: Dict, operators: List[Dict]) -> Dict:
        """Reverts pose coordinates to the original pre-transform space."""
        if not operators:
            return results

        if results.get('boxes') is not None:
             results['boxes'] = pipeline_utils.revert_to_origin(results['boxes'], operators)
        if results.get('points') is not None:
            points = results['points']
            # TODO: add visibility if needed, which is points[:,-1]
            if len(points) and points.shape[-1] == 3:
                points = points[:,:,:-1]
            reverted_points = [pipeline_utils.revert_to_origin(p, operators) for p in points] # each iter: [n_kp,2]
            results['points'] = torch.stack(reverted_points) if isinstance(points, torch.Tensor) else np.array(reverted_points)

        return results
