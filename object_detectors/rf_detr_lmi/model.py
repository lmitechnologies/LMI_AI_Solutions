from od_core.od_base import ODBase
from od_core.object_detector_registry import ObjectDetectorRegistry
from od_core.results import Results
import gadget_utils.pipeline_utils as pipeline_utils
import logging
import os
import torch
import numpy as np
from rfdetr import RFDETRMedium, RFDETRLarge, RFDETRSmall, RFDETRNano, RFDETRBase


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

class RfdetrTRT(ODBase):
    def __init__(self, model_path: str, device='cuda', fp16=False, **kwargs) -> None:
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        
        with open(model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.current_idx = 0
        self.streams = [cuda.Stream(), cuda.Stream()]
        
        self.buffer_sets = [self._allocate_buffers() for _ in range(2)]
        
        self.input_shape = self.buffer_sets[0]['inputs'][0]["shape"]
        self.input_dtype = self.buffer_sets[0]['inputs'][0]["dtype"]

    def _allocate_buffers(self):
        """Allocates one set of pinned host memory and device memory."""
        inputs = []
        outputs = []
        all_bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
            
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            binding = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "host": host_mem,
                "device": device_mem,
            }
            
            all_bindings.append(binding)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(binding)
            else:
                outputs.append(binding)
        
        return {"inputs": inputs, "outputs": outputs, "all": all_bindings}

        def warmup(self):
        """Warm up the model by running a dummy inference."""
        dummy_input = np.zeros(self.input_shape, dtype=self.input_dtype)
        self.forward(np.ascontiguousarray(dummy_input, dtype=self.input_dtype))

    
    def preprocess(self, image: np.ndarray, **kwargs):
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image in numpy array format.
        """
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return np.ascontiguousarray(image, dtype=self.input_dtype)       

    def forward(self, image: np.ndarray, **kwargs) -> list:
        """
        Perform inference.
        """
        bufs = self.buffer_sets[self.current_idx]
        stream = self.streams[self.current_idx]
        
        np.copyto(bufs['inputs'][0]['host'], image.ravel())

        cuda.memcpy_htod_async(bufs['inputs'][0]['device'], bufs['inputs'][0]['host'], stream)

        for b in bufs['all']:
            self.context.set_tensor_address(b['name'], int(b['device']))
            
        self.context.execute_async_v3(stream_handle=stream.handle)
        results = []
        for out in bufs['outputs']:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
            results.append(out)
        stream.synchronize()

        self.current_idx = 1 - self.current_idx
        return [r['host'].reshape(r['shape']) for r in results]
    
    def postprocess(outputs, **kwargs) -> dict:
        # outputs[0]: boxes (1, N, 4), outputs[1]: logits (1, N, classes)
        dets_data = outputs[0][0]
        labels_data = outputs[1][0]
            
        res = self.input_shape[2]  # assuming square input
        orig_h, orig_w = kwargs.get('original_size', (res, res))
        scale_w, scale_h = orig_w / res, orig_h / res
            
        final_boxes, final_scores, final_ids = [], [], []
            
        # sigmoid and max
        scores = 1 / (1 + np.exp(-labels_data))  # sigmoid
        max_scores = np.max(scores, axis=1)
        max_ids = np.argmax(scores, axis=1) 
            
        mask = (max_scores > 0.5) & (max_ids >= 0)
            
        for i in np.where(mask)[0]:
            # CXCYWH to XYXY
            cx, cy, w, h = dets_data[i] * res
            x1, y1 = (cx - w/2) * scale_w, (cy - h/2) * scale_h
            x2, y2 = (cx + w/2) * scale_w, (cy + h/2) * scale_h
                
            final_boxes.append([x1, y1, x2, y2])
            final_scores.append(max_scores[i])
            final_ids.append(max_ids[i])
       
        final_boxes = np.array(final_boxes)
        final_scores = np.array(final_scores)
        final_ids = np.array(final_ids)
        
        return Results(
            boxes = torch.from_numpy(final_boxes) if len(final_boxes) > 0 else [],
            scores = torch.from_numpy(final_scores) if len(final_scores) > 0 else [],
            classes = final_ids if len(final_ids) > 0 else []
        )

    def predict(self, image, configs={}, operators=[], **kwargs):
        """Perform object detection on a list of images.

        Args:
            image (np.ndarray): Input image in numpy array format.
            configs (dict): Configuration dictionary for confidence thresholding
            operators (list, optional): List of operators to apply. Defaults to [].
            iou (float, optional): IoU threshold for NMS. Defaults to 0.4.
            agnostic (bool, optional): Class-agnostic NMS flag. Defaults to False.
            max_det (int, optional): Maximum number of detections per image. Defaults to 300.

        Returns:
            Results: Object containing detection results.
        """
        preprocessed_image = self.preprocess(image, **kwargs)
        # inference
        outputs = self.forward(preprocessed_image, **kwargs)
        # postprocess
        results = self.postprocess(outputs, configs=configs, operators=operators, **kwargs)
        results = results.to_dict(return_tensor=False)
        if results == {}:
            results = {
                'boxes': [],
                'scores': [],
                'classes': []
            }
        return results
    
    @staticmethod
    def annotate_image(results, image, colormap=None, line_thickness=None, hide_label=False, hide_bbox=False):
        """annotate model results on the image. If colormap is None, it will use the random colors.

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

        image = to_numpy(image).copy()
        if not len(boxes):
            return image
        
        # convert to numpy
        boxes = to_numpy(boxes)
        
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
            pipeline_utils.plot_one_box(boxes[i],image,None,**args)
                
        return image
        

class RfdetrPT(ODBase):
    
    logger = logging.getLogger('RFDETR')
    logger.setLevel(logging.INFO)
    
    def __init__(self, model_path:str, device='cuda', data=None, fp16=False,**kwargs) -> None:
        self.image_size = kwargs.get('image_size', [640, 640])
        
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            device = 'cpu'

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'File not found: {model_path}')
        
        # TODO load model and support all types
        model_type = kwargs.get('model_type', 'medium').lower()
        self.class_names = {}
        self.model = None
        if model_type == 'medium':
            self.model = RFDETRMedium(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        elif model_type == 'large':
            self.model = RFDETRLarge(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        elif model_type == 'small':
            self.model = RFDETRSmall(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        elif model_type == 'nano':
            self.model = RFDETRNano(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        elif model_type == 'base':
            self.model = RFDETRBase(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        else:
            raise ValueError(f'Unsupported model type: {model_type}. Supported types are "medium".')
        if self.model is None:
            raise ValueError('Model loading failed.')
        
        self.class_names = self.model.class_names
        self.model.optimize_for_inference()
        
    def warmup(self):
        """Warm up the model by running a dummy inference."""
        self.model.predict(np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8))
    
    def preprocess(self, image: np.ndarray, **kwargs):
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image in numpy array format.
        """
        pass # No preprocessing needed for RF-DETR        
    
    def forward(self, image, **kwargs) -> dict:
        """Perform object detection on a list of images.

        Args:
            image (np.ndarray): Input image in numpy array format.

        Returns:
            Results: Object containing detection results.
        """

        return self.model.predict(image)
    
    def _construct_results(self, preds, **kwargs) -> dict:
        return Results(
            boxes = preds.xyxy,
            scores = preds.confidence,
            classes = preds.class_id
        )

    def postprocess(self, preds, **kwargs) -> Results:
        """Postprocess the model outputs.

        Args:
            outputs (dict): Model outputs.

        Returns:
            dict: Postprocessed outputs.
        """
        conf = kwargs.get('conf', 0.5)
        if isinstance(conf, float):
            conf_thresholds = {cls_name: conf for cls_name in self.class_names.values()}
        elif isinstance(conf, dict):
            conf_thresholds = conf
        else:
            raise ValueError('conf should be a float or a dict')
        
        operators = kwargs.get('operators', [])
        boxes = np.array(preds.xyxy)
        scores = np.array(preds.confidence)
        classes = preds.class_id
        # convert class ids to names
        classes = np.array([self.class_names[int(c)+1] for c in classes])
        mask = scores >= np.vectorize(conf_thresholds.get)(classes, 1.0)

        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # revert to origin
        if len(operators) > 0:
            boxes = pipeline_utils.revert_to_origin(boxes, operators)
        
        return Results(
            boxes = torch.from_numpy(boxes) if len(boxes) > 0 else [],
            scores = torch.from_numpy(scores) if len(scores) > 0 else [],
            classes = [] if classes is None else classes
        )

    @staticmethod
    def annotate_image(results, image, colormap=None, line_thickness=None, hide_label=False, hide_bbox=False):
        """annotate model results on the image. If colormap is None, it will use the random colors.

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

        image = to_numpy(image).copy()
        if not len(boxes):
            return image
        
        # convert to numpy
        boxes = to_numpy(boxes)
        
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
            pipeline_utils.plot_one_box(boxes[i],image,None,**args)
                
        return image
        

    def predict(self, image, configs={}, operators=[], **kwargs):
        """Perform object detection on a list of images.

        Args:
            image (np.ndarray): Input image in numpy array format.
            configs (dict): Configuration dictionary for confidence thresholding
            operators (list, optional): List of operators to apply. Defaults to [].
            iou (float, optional): IoU threshold for NMS. Defaults to 0.4.
            agnostic (bool, optional): Class-agnostic NMS flag. Defaults to False.
            max_det (int, optional): Maximum number of detections per image. Defaults to 300.

        Returns:
            Results: Object containing detection results.
        """
        # inference
        outputs = self.forward(image, **kwargs)
        # postprocess
        results = self.postprocess(outputs, configs=configs, operators=operators, **kwargs)
        results = results.to_dict(return_tensor=False)
        if results == {}:
            results = {
                'boxes': [],
                'scores': [],
                'classes': []
            }
        return results
        
        

    



    


    

