from od_core.od_base import ODBase
from od_core.object_detector_registry import ObjectDetectorRegistry
from od_core.results import Results
import gadget_utils.pipeline_utils as pipeline_utils
import logging
import os
import torch
import numpy as np
from rfdetr import RFDETRMedium, RFDETRLarge, RFDETRSmall, RFDETRNano, RFDETRBase
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


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




@ObjectDetectorRegistry.register(metadata=dict(versions=["v1"], model_names=["rfdetr"], tasks=["od","seg", "instancesegmentation", "objectdetection"], frameworks=["rfdetr"]))
class RfdetrModel(ODBase):
    """
    RfdetrModel is a factory class for creating object detection models based on the Rfdetr framework.
    Attributes:
        _registry (dict): A dictionary that maps file extensions to their corresponding model wrapper classes.
    Methods:
        register(format):
            Registers a model wrapper class for a specific file format.
            Args:
                format (str): The file extension format to register the wrapper class for.
            Returns:
                function: A decorator function that registers the wrapper class.
        __new__(cls, model_path, class_map, *args, **kwargs):
            Creates an instance of the appropriate model wrapper class based on the file extension of the model_path.
            Args:
                model_path (str): The file path to the model file.
                class_map (dict): A dictionary mapping class IDs to class names.
                *args: Additional positional arguments to pass to the model wrapper class.
                **kwargs: Additional keyword arguments to pass to the model wrapper class.
            Returns:
                object: An instance of the appropriate model wrapper class.
            Raises:
                ValueError: If the file extension of model_path is not registered.
    """
    _registry = {}

    @classmethod
    def register(cls, format):
        def decorator(wrapper_cls):
            cls._registry[format] = wrapper_cls
            return wrapper_cls
        return decorator
    
    def __new__(cls, model_path, *args,**kwargs):
        ext = model_path.split(".")[-1]
        wrapper_cls = cls._registry.get(ext)
        if wrapper_cls is None:
            raise ValueError("Invalid model file extension")
        
        return wrapper_cls(model_path, *args, **kwargs)

@RfdetrModel.register('engine')
class RfdetrTRT(RfdetrModel):

    logger = logging.getLogger('RFDETR')
    logger.setLevel(logging.INFO)

    def __init__(self, model_path: str, device='cuda', fp16=False, **kwargs) -> None:
        self.image_size = kwargs.get('image_size', (640, 640))
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.trt_logger)
        
        with open(model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.current_idx = 0
        self.streams = [cuda.Stream(), cuda.Stream()]
        
        self.buffer_sets = [self._allocate_buffers() for _ in range(2)]
        
        self.input_shape = self.buffer_sets[0]['inputs'][0]["shape"]
        self.input_dtype = self.buffer_sets[0]['inputs'][0]["dtype"]
        self.num_classes = self.buffer_sets[0]['outputs'][1]["shape"][-1]

        class_map = kwargs.get("class_map", None)
        if class_map is None:
            raise ValueError("class_map is required for [Detectron2TRT]")
        self.class_map = {
            int(k): str(v) for k, v in class_map.items()
        }
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))

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
        # Convert to float and Normalize (0.0 to 1.0)
        input_img = image.astype(np.float32) / 255.0

        # HWC to CHW (C++ memcpy logic equivalent)
        # Apply normalization: (val - mean) / std
        means = np.array(self.means, dtype=np.float32)
        stds = np.array(self.stds, dtype=np.float32)
        input_img = (input_img - means) / stds
        
        # Transpose to NCHW
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, axis=0)
        
        return np.array([input_img], dtype=self.input_dtype)

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
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def postprocess(self,outputs,image, **kwargs) -> Results:
        orig_h, orig_w = image.shape[:2]
        configs = kwargs.get("configs")
        if configs is None:
            self.logger.warning("configs is None. Using default value of 1.0 for all classes.")
            configs = {k if isinstance(k,str) else v: 1.0 for k,v in self.class_map.items()}
        if isinstance(configs, dict) is False and isinstance(configs, (int, float)):
            configs = {k if isinstance(k,str) else v: configs for k,v in self.class_map.items()}
        else:
            self.logger.warning("configs should be a dictionary of class confidence thresholds. Using default value of 1.0 for all classes.")
        
        if len(outputs) < 2:
            raise RuntimeError(f"Expected at least 2 output tensors, got {len(outputs)}")

        dets_data = outputs[0][0]
        labels_data = outputs[1][0]
        logger.info(labels_data.shape)
        scores_all = sigmoid(labels_data)
        
        max_scores = np.max(scores_all, axis=1)
        max_class_indices = np.argmax(scores_all, axis=1)
        
        max_class_indices = self.class_map_func(max_class_indices)

        mask = max_scores >= np.vectorize(configs.get)(max_class_indices, 1.0)
        
        filtered_scores = max_scores[mask]
        filtered_classes = max_class_indices[mask]
        filtered_dets = dets_data[mask] # shape (N, 4)

        if filtered_dets.shape[0] == 0:
            return Results(
                boxes = [],
                scores = [],
                classes = []
            )

        cx = filtered_dets[:, 0] * orig_w
        cy = filtered_dets[:, 1] * orig_h
        w  = filtered_dets[:, 2] * orig_w
        h  = filtered_dets[:, 3] * orig_h

        x_min = cx - w / 2.0
        y_min = cy - h / 2.0
        x_max = cx + w / 2.0
        y_max = cy + h / 2.0
        final_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

        return Results(
            boxes = torch.from_numpy(final_boxes),
            scores = torch.from_numpy(filtered_scores),
            classes = filtered_classes
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
        
@RfdetrModel.register('pth')
class RfdetrPT(ODBase):
    
    logger = logging.getLogger('RFDETR')
    logger.setLevel(logging.INFO)
    
    def __init__(self, model_path:str, **kwargs) -> None:
        print(f"kwargs: {kwargs}")
        
        if torch.cuda.is_available() and kwargs.get('device', 'cuda') == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'File not found: {model_path}')
        
        # TODO load model and support all types
        model_type = kwargs.get('model_type', 'medium').lower()
        self.class_names = {}
        self.model = None
        if model_type == 'medium':
            self.image_size = (kwargs.get('image_size')[0] if kwargs.get('image_size') is not None else 576,
                               kwargs.get('image_size')[1] if kwargs.get('image_size') is not None else 576)
            self.model = RFDETRMedium(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        elif model_type == 'large':
            self.image_size = (kwargs.get('image_size')[0] if kwargs.get('image_size') is not None else 560,
                               kwargs.get('image_size')[1] if kwargs.get('image_size') is not None else 560)
            self.model = RFDETRLarge(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        elif model_type == 'small':
            self.image_size = (kwargs.get('image_size')[0] if kwargs.get('image_size') is not None else 512,
                               kwargs.get('image_size')[1] if kwargs.get('image_size') is not None else 512)

            self.model = RFDETRSmall(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        elif model_type == 'nano':
            self.image_size = (kwargs.get('image_size')[0] if kwargs.get('image_size') is not None else 384,
                               kwargs.get('image_size')[1] if kwargs.get('image_size') is not None else 384)
            self.model = RFDETRNano(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        elif model_type == 'base':
            self.image_size = (kwargs.get('image_size')[0] if kwargs.get('image_size') is not None else 560,
                               kwargs.get('image_size')[1] if kwargs.get('image_size') is not None else 560)
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
        conf = kwargs.get('configs', 0.5)
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
        classes = np.array([self.class_names[int(c)] for c in classes])
        if len(boxes) == 0:
            return Results(
                boxes = [],
                scores = [],
                classes = []
            )
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
        
        

    



    


    

