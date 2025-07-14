from typing import Dict, List
from od_core.od_base import ODBase
from od_core.object_detector_registry import ObjectDetectorRegistry
import numpy as np
from gadget_utils.pipeline_utils import plot_one_box, revert_to_origin, revert_mask_to_origin
from postprocess_utils.mask_utils import rescale_masks,mask_to_polygon_cv2
import cv2
import logging
import torch
import torchvision
from torchvision.ops import batched_nms
import time

@ObjectDetectorRegistry.register(metadata=dict(versions=["v0"], model_names=["mask_rcnn", "faster_rcnn"], tasks=["od","seg", "instancesegmentation", "objectdetection"], frameworks=["detectron2"]))
class Detectron2Model(ODBase):
    """
    Detectron2Model is a factory class for creating object detection models based on the Detectron2 framework.
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
    

@Detectron2Model.register("engine")
class Detectron2TRT(ODBase):
    
    logger = logging.getLogger('Detectron2TRT')
    logger.setLevel(logging.INFO)
    
    def __init__(self, model_path, **kwargs):
        """
        Initialize the Detectron2 model with TensorRT engine.
        Args:
            model_path (str): Path to the serialized TensorRT engine file.
            class_map (dict): Dictionary mapping class IDs to class names.
        Attributes:
            engine (trt.ICudaEngine): The TensorRT engine.
            context (trt.IExecutionContext): The execution context for the engine.
            model_inputs (list): List of input tensor bindings.
            model_outputs (list): List of output tensor bindings.
            allocations (list): List of memory allocations for input and output tensors.
            input_shape (list): Shape of the input tensor.
            input_dtype (numpy.dtype): Data type of the input tensor.
            class_map (dict): Dictionary mapping class IDs to class names.
        """
        """source: https://github.com/NVIDIA/TensorRT/tree/release/10.4/samples/python/detectron2"""
        
        import tensorrt as trt
        from cuda import cudart
        import detectron2_lmi.utils.common_runtime as common
        
        trt_logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(trt_logger, namespace="")
        with open(model_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Setup I/O bindings
        self.model_inputs = []
        self.model_outputs = []
        self.allocations = []
        device = kwargs.get("device", "cuda")
        self.device = torch.device(device)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
                "size": size,
            }
            self.allocations.append(allocation)
            if is_input:
                self.model_inputs.append(binding)
            else:
                self.model_outputs.append(binding)
        
        self.input_shape = self.model_inputs[0]["shape"]
        self.input_dtype = self.model_inputs[0]["dtype"]
        self.image_size = [self.input_shape[2], self.input_shape[3]]
        class_map = kwargs.get("class_map", None)
        if class_map is None:
            raise ValueError("class_map is required for [Detectron2TRT]")
        try:
            self.class_map = {
                int(k): str(v) for k, v in class_map.items()
            }
        except Exception as e:
            self.class_map = {
                int(v): str(k) for k, v in class_map.items()
            }
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))
    
    def warmup(self):
        """
        Perform a warmup operation for the model.

        This method runs a forward pass with a randomly generated input tensor
        to warm up the model. It helps in preparing the model for actual inference
        by initializing necessary components and reducing the initial latency.

        The input tensor is generated with the same shape and data type as the
        expected input during inference.

        Parameters:
        None

        Returns:
        None
        """
        for _ in range(1):
            image_h, image_w = self.image_size
            input = np.random.rand(self.batch_size, 3, image_h, image_w).astype(self.input_dtype)
            self.forward(input)
        
    def preprocess(self, images: np.ndarray):
        """
        Preprocesses a batch of images for input into the model.

        Args:
            images (np.ndarray): A batch of images to preprocess. Each image should be in the format (H, W, C).

        Returns:
            np.ndarray: A batch of preprocessed images with shape (batch_size, 3, image_h, image_w).
        """
        image_h, image_w = self.image_size
        inputs = np.zeros((self.batch_size, 3, image_h, image_w), dtype=self.input_dtype)
        for i in range(0, self.batch_size):
            image = images[i]
            inputs[i] = image.astype(np.float32).transpose(2, 0, 1)
        return inputs
    
    def forward(self, inputs):
        """
        Perform a forward pass through the model.

        Args:
            inputs (numpy.ndarray): The input data to be processed by the model.

        Returns:
            list: A list of numpy arrays containing the model's output data.
        """
        import detectron2_lmi.utils.common_runtime as common
        
        outputs = []
        for out in self.model_outputs:
            outputs.append(np.zeros(out["shape"], dtype=out["dtype"]))
        common.memcpy_host_to_device(
            self.model_inputs[0]["allocation"], np.ascontiguousarray(inputs)
        )
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.model_outputs[o]["allocation"])
        return outputs
    
    def postprocess(self, images, predictions, **kwargs):
        """
        Post-process the predictions from the object detection model.

        Args:
            images (list): List of input images.
            predictions (tuple): Tuple containing (num_preds, boxes, scores, classes[, masks]).
            **kwargs: Additional keyword arguments for processing.
                - confs (dict): Dict of confidence thresholds per class.
                - mask_threshold (float): Threshold for mask binarization.
                - process_masks (bool): Whether to process masks.
                - operators (list): Geometric ops to revert.
                - return_segments (bool): Whether to compute polygons.
                - iou (float): IoU threshold for NMS. Default: 0.5.
                - max_det (int): Max boxes to keep per image after NMS. Default: 100.
        Returns:
            dict: {
                "boxes": list of ndarray [N×4],
                "scores": list of ndarray [N],
                "classes": list of ndarray [N],
                "masks": list (if any),
                "segments": list (if any),
            }
        """
        # prepare outputs
        results = {"boxes": [], "scores": [], "classes": [], "masks": [], "segments": []}
        if len(predictions) == 0:
            return results

        # unpack kwargs
        confs          = kwargs.get("confs", {})
        mask_threshold = kwargs.get("mask_threshold", 0.5)
        process_masks  = kwargs.get("process_masks", True)
        operators      = kwargs.get("operators", [])
        iou_threshold  = kwargs.get("iou", 0.5)
        max_detections = kwargs.get("max_det", 100)
        return_segs    = kwargs.get("return_segments", False)

        # image dims
        image_h, image_w = images[0].shape[:2]

        # unpack predictions
        if len(predictions) == 5:
            _, boxes, scores, classes, masks = predictions
        else:
            _, boxes, scores, classes = predictions
            masks = None

        # map classes
        classes = self.class_map_func(classes)

        # scale to pixel coords
        if len(boxes) > 0:
            scale = np.array([image_w, image_h, image_w, image_h])
            boxes = (boxes * scale).astype(np.int32)

        # per-image containers
        proc_boxes, proc_scores, proc_classes = [], [], []
        proc_masks, proc_segs                 = [], []

        t0 = time.time()
        for b in range(self.batch_size):
            raw_boxes   = boxes[b]
            raw_scores  = scores[b]
            raw_classes = classes[b]
            raw_masks   = masks[b] if masks is not None else None

            # NMS on raw detections 
            if raw_boxes.shape[0] > 0:
                tb = torch.tensor(raw_boxes,   device=self.device).float()
                ts = torch.tensor(raw_scores,  device=self.device).float()
                tc = torch.tensor(raw_classes, device=self.device).long()

                keep_idx = batched_nms(tb, ts, tc, iou_threshold)

                if keep_idx.numel() > max_detections:
                    # gather the scores of all kept detections
                    kept_scores = ts[keep_idx]
                    # sort them descending
                    _, order = kept_scores.sort(descending=True)
                    # pick only the top‐max_detections by score
                    keep_idx = keep_idx[order[:max_detections]]


                raw_boxes   = raw_boxes[keep_idx]
                raw_scores  = raw_scores[keep_idx]
                raw_classes = raw_classes[keep_idx]
                if raw_masks is not None:
                    raw_masks = raw_masks[keep_idx]

            # confidence filtering
            conf_fn = np.vectorize(lambda c: confs.get(c, 1.0))
            keep_conf = raw_scores >= conf_fn(raw_classes)

            batch_boxes   = raw_boxes[keep_conf]
            batch_scores  = raw_scores[keep_conf]
            batch_classes = raw_classes[keep_conf]
            batch_masks   = raw_masks[keep_conf] if raw_masks is not None else None

            # mask processing
            segs, final_masks = [], []
            if process_masks and batch_masks is not None:
                m_tensor   = torch.from_numpy(batch_masks).to(self.device)
                box_tensor = torch.from_numpy(batch_boxes).to(self.device)
                resized    = rescale_masks(m_tensor, box_tensor, (image_h, image_w), mask_threshold)

                # revert geometric ops on masks
                if operators:
                    resized = torch.stack([
                        torch.from_numpy(
                            revert_mask_to_origin(m.cpu().numpy(), operators)
                        ) for m in resized
                    ])

                final_masks = [m.cpu().numpy() for m in resized]

                if return_segs:
                    for m in resized:
                        arr = m.cpu().numpy()
                        poly = mask_to_polygon_cv2(arr)
                        segs.append(
                            revert_to_origin(poly, operators) if operators else poly
                        )
            else:
                final_masks = batch_masks if batch_masks is not None else []

            # Revert ops on boxes
            batch_boxes = revert_to_origin(batch_boxes, operators)

            # collect
            proc_boxes.append(batch_boxes)
            proc_scores.append(batch_scores)
            proc_classes.append(batch_classes)
            proc_masks.append(final_masks)
            proc_segs.append(segs)

        t1 = time.time()

        return {
            "boxes":    proc_boxes,
            "scores":   proc_scores,
            "classes":  proc_classes,
            "masks":    proc_masks,
            "segments": proc_segs,
        }

    def predict(self, images, **kwargs):
        """
        Perform prediction on the given images.

        Args:
            images (list or np.ndarray): The input images to be processed.
            **kwargs: Additional keyword arguments for postprocessing.

        Returns:
            list: The predictions after postprocessing.

        Logs:
            The time taken for postprocessing in milliseconds.
        """
        t0 = time.time()
        
        if isinstance(images, np.ndarray):
            # if the input is a single image
            shp = images.shape
            if len(shp) == 3:
                images = [images]
            elif len(shp) == 4 and images.shape[0] != self.batch_size:
                self.logger.error(f"Batch size mismatch: {images.shape[0]} != {self.batch_size}")
                return {}
        
        predictions = self.forward(self.preprocess(images))
        predictions = self.postprocess(images, predictions,**kwargs)
        t1 = time.time()
        self.logger.info(f"proc-time {(t1-t0)*1000.0:.2f} ms")
        return predictions
    
    def annotate_image(self, result, image, color_map=None, **kwargs):
        """
        Annotates an image with bounding boxes, class labels, scores, and masks.

        Args:
            result (dict): A dictionary containing detection results with keys:
                - "classes" (list): List of detected class labels.
                - "scores" (list): List of confidence scores for each detected class.
                - "boxes" (list): List of bounding boxes for each detected object.
                - "masks" (list, optional): List of masks for each detected object.
            image (numpy.ndarray): The image to annotate.
            color_map (dict, optional): A dictionary mapping class labels to colors.

        Returns:
            numpy.ndarray: The annotated image.
        """
        for i in range(len(result["classes"])):
            plot_one_box(
                result["boxes"][i],
                image,
                label=f"{result['classes'][i]}:{result['scores'][i]:.2f}",
                mask=result["masks"][i] if len(result["masks"]) > 0 else None,
                color=color_map,
            )
        return image

@Detectron2Model.register("pt")
class Detectron2PT(ODBase):
    
    logger = logging.getLogger('Detectron2PT')
    logger.setLevel(logging.INFO)
   
    def __init__(self, model_path,**kwargs):
        device = kwargs.get("device", "cuda")
        if not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
        except Exception as e:
            self.logger.exception(f"❗ Failed to load model: {e}")
        
        class_map = kwargs.get("class_map", None)
        if class_map is None:
            raise ValueError("class_map is required for [Detectron2PT]")
        try:
            self.class_map = {
                int(k): str(v) for k, v in class_map.items()
            }
        except Exception as e:
            # handle the case where class_map is in reverse order
            self.class_map = {
                int(v): str(k) for k, v in class_map.items()
            }
        self.batch_size = kwargs.get('batch_size', 1)
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))
        self.image_size = kwargs.get('image_size', [640, 640])
    
    def warmup(self, **kwargs):
        """
        Perform a warmup operation for the model.

        This method generates a random input tensor with the specified image size and 
        passes it through the model to perform a warmup. This can be useful to initialize 
        model parameters and optimize performance before actual inference.

        Args:
            **kwargs: Arbitrary keyword arguments.
                img_size (tuple, optional): A tuple specifying the height and width of the image.
                                            If not provided, an error is logged.

        Raises:
            ValueError: If 'img_size' is not provided in kwargs.
        """
        image_size = kwargs.get('img_size', self.image_size)
        image_h, image_w = image_size[0], image_size[1]
        images = [np.random.rand(image_h, image_w, 3).astype(np.float32) for _ in range(self.batch_size)]
        input = [
            dict(image=torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).to(dtype=torch.float32).to(self.device)) for image in images
        ]
        self.forward(input)
        
    def preprocess(self, images: np.ndarray) -> List[Dict[str, torch.Tensor]]:
        """
        Preprocesses a batch of images for input into the model.

        Args:
            images (np.ndarray): A numpy array of images to be preprocessed. 
                                 Each image is expected to be in HWC format.

        Returns:
            list: A list of dictionaries where each dictionary contains a single key 'image' 
                  with the preprocessed image as a value. The image is converted to float32 
                  and transposed to CHW format.
        """
        image_h, image_w = images[0].shape[0], images[0].shape[1]
        inputs = np.zeros((len(images), 3, image_h, image_w), dtype=np.float32)
        inputs = [
            dict(image=torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).to(dtype=torch.float32).to(self.device)) for image in images
        ]
        return inputs
    
    def forward(self, inputs):
        """
        Perform a forward pass through the model with the given inputs.

        Args:
            inputs (torch.Tensor): The input tensor to be passed through the model.

        Returns:
            torch.Tensor: The model's predictions for the given inputs.
        """
        with torch.no_grad():
            predictions = self.model.forward(inputs)
        return predictions
    
    def postprocess(self, images, predictions, **kwargs):
        """
        Post-processes the predictions from the model to filter and format the results.

        Args:
            images (list): A list of input images.
            predictions (list): A list of model outputs, each a dict with:
                - "scores" (Tensor): confidence scores [N]
                - "pred_classes" (Tensor): class indices [N]
                - "pred_boxes" (Tensor): boxes [N×4]
                - "pred_masks" (Tensor, optional): masks [N×1×H×W]
            **kwargs:
                - confs (dict): class → confidence threshold (default 0.5)
                - mask_threshold (float): binarization threshold for masks (default 0.5)
                - process_masks (bool): whether to rescale/process masks (default True)
                - operators (list): geometric ops for reverting to original coords
                - return_segments (bool): whether to compute polygon segments from masks
                - iou_threshold (float): IoU threshold for NMS (default 0.5)
                - max_detections (int): max detections per image after NMS (default 100)

        Returns:
            dict with keys "boxes", "scores", "classes", "masks", "segments",
            each a list of length batch_size.
        """
        results = {"boxes": [], "scores": [], "classes": [], "masks": [], "segments": []}
        if not predictions:
            return results

        confs          = kwargs.get("confs", {})
        mask_threshold = kwargs.get("mask_threshold", 0.5)
        process_masks  = kwargs.get("process_masks", True)
        operators      = kwargs.get("operators", [])
        return_segs    = kwargs.get("return_segments", False)
        iou_threshold  = kwargs.get("iou", 0.5)
        max_detections = kwargs.get("max_det", 100)

        for idx, output in enumerate(predictions):
            image_h, image_w = images[idx].shape[:2]

            # Raw tensors
            raw_scores  = output["scores"]
            raw_classes = output["pred_classes"]
            raw_boxes   = output["pred_boxes"]
            raw_masks   = output.get("pred_masks", None)

            # 1) NMS on raw outputs
            if raw_boxes.numel() > 0:
                keep_nms = batched_nms(
                    raw_boxes.to(self.device).float(),
                    raw_scores.to(self.device).float(),
                    raw_classes.to(self.device).long(),
                    iou_threshold
                )
                if keep_nms.numel() > max_detections:
                    # gather the scores of all kept detections
                    kept_scores = raw_scores[keep_nms]
                    # sort them descending
                    _, order = kept_scores.sort(descending=True)
                    # pick only the top‐max_detections by score
                    keep_nms = keep_nms[order[:max_detections]]

                raw_scores  = raw_scores[keep_nms]
                raw_classes = raw_classes[keep_nms]
                raw_boxes   = raw_boxes[keep_nms]
                if raw_masks is not None:
                    raw_masks = raw_masks[keep_nms]
            else:
                raw_masks = None

            #  class‐aware confidence filtering
            scores_np  = raw_scores.cpu().numpy()
            classes_np = self.class_map_func(raw_classes.cpu().numpy())
            # build threshold array
            thr_arr = np.vectorize(lambda c: confs.get(c, 0.5))(classes_np)
            keep_conf = scores_np >= thr_arr

            if keep_conf.any():
                batch_scores  = scores_np[keep_conf]
                batch_classes = classes_np[keep_conf]
                batch_boxes   = raw_boxes[keep_conf]
                batch_masks   = raw_masks[keep_conf] if raw_masks is not None else None
            else:
                # no detections
                results["boxes"].append(np.zeros((0, 4), dtype=np.int32))
                results["scores"].append(np.array([]))
                results["classes"].append(np.array([]))
                results["masks"].append([])
                results["segments"].append([])
                continue

            # process masks & segments
            batch_segments = []
            if process_masks and batch_masks is not None and batch_masks.numel() > 0:
                # squeeze channel dim
                masks_t = batch_masks.to(self.device).squeeze(1)
                boxes_t = batch_boxes.to(self.device)
                rescaled = rescale_masks(masks_t, boxes_t, (image_h, image_w), mask_threshold)

                # revert geometric ops on masks
                if operators:
                    rescaled = torch.stack([
                        torch.from_numpy(
                            revert_mask_to_origin(m.cpu().numpy(), operators)
                        ) for m in rescaled
                    ])

                final_masks = [m.cpu().numpy() for m in rescaled]

                if return_segs:
                    for m in rescaled:
                        arr = m.cpu().numpy()
                        poly = mask_to_polygon_cv2(arr)
                        if operators:
                            poly = revert_to_origin(poly, operators)
                        batch_segments.append(poly)
            else:
                final_masks = (
                    batch_masks.cpu().numpy() if isinstance(batch_masks, torch.Tensor) else []
                )

            # 4) Revert boxes to original coords
            boxes_np = batch_boxes.cpu().numpy()
            boxes_np = revert_to_origin(boxes_np, operators)

            # collect
            results["boxes"].append(boxes_np)
            results["scores"].append(batch_scores)
            results["classes"].append(batch_classes)
            results["masks"].append(final_masks)
            results["segments"].append(batch_segments)

        return results
    
    def predict(self, images, **kwargs):
        """
        Perform prediction on the given images.

        This method preprocesses the input images, performs forward pass to get predictions,
        and then postprocesses the predictions to generate the final results.

        Args:
            images (list or array-like): The input images to be processed.
            operators (optional): Not yet supported. Default is None.
            **kwargs: Additional keyword arguments for postprocessing.

        Returns:
            results: The final processed results after prediction.

        Raises:
            NotImplementedError: If operators is not None, indicating that the feature is not yet supported.
        """
        t0 = time.time()
        if isinstance(images, np.ndarray):
            # if the input is a single image
            shp = images.shape
            if len(shp) == 3:
                images = [images]
            
        # preprocess
        inputs = self.preprocess(images)
        # forward
        predictions = self.forward(inputs)
        # postprocess
        results = self.postprocess(images, predictions,**kwargs)
        t1= time.time()
        self.logger.info(f"proc-time {(t1-t0)*1000.0:.2f} ms")
        return results
        
    
    def annotate_image(self, result, image, color_map=None, **kwargs):
        """
        Annotates an image with bounding boxes, class labels, scores, and masks.

        Args:
            result (dict): A dictionary containing detection results with keys:
                - "classes" (list): List of detected class labels.
                - "scores" (list): List of confidence scores for each detected class.
                - "boxes" (list): List of bounding boxes for each detected object.
                - "masks" (list, optional): List of masks for each detected object.
            image (numpy.ndarray): The image to annotate.
            color_map (dict, optional): A dictionary mapping class labels to colors.

        Returns:
            numpy.ndarray: The annotated image.
        """
        for i in range(len(result["classes"])):
            plot_one_box(
                result["boxes"][i],
                image,
                label=f"{result['classes'][i]}:{result['scores'][i]:.2f}",
                mask=result["masks"][i] if len(result["masks"]) > 0 else None,
                color=color_map,
            )
        return image
    