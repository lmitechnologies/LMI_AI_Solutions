import logging
import os

import gadget_utils.pipeline_utils as pipeline_utils
import numpy as np
import torch
from od_core.object_detector_registry import ObjectDetectorRegistry
from od_core.od_base import ODBase
from od_core.results import Results

try:
    from rfdetr import RFDETR2XLarge, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall, RFDETRXLarge
except ImportError:
    pass
import cv2

try:
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError:
    pass


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
        raise TypeError(f"Data type {type(data)} not supported")


@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=["v1"], model_names=["rfdetr"], tasks=["od", "seg", "instancesegmentation", "objectdetection"], frameworks=["rfdetr"]
    )
)
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

    def __new__(cls, model_path, *args, **kwargs):
        ext = model_path.split(".")[-1]
        wrapper_cls = cls._registry.get(ext)
        if wrapper_cls is None:
            raise ValueError("Invalid model file extension")

        return wrapper_cls(model_path, *args, **kwargs)


@RfdetrModel.register("engine")
class RfdetrTRT(ODBase):
    logger = logging.getLogger("RFDETR")
    logger.setLevel(logging.INFO)

    def __init__(self, model_path: str, device="cuda", fp16=False, **kwargs) -> None:
        self.image_size = kwargs.get("image_size", (640, 640))
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

        self.input_shape = self.buffer_sets[0]["inputs"][0]["shape"]
        self.input_dtype = self.buffer_sets[0]["inputs"][0]["dtype"]
        self.num_classes = self.buffer_sets[0]["outputs"][1]["shape"][-1]

        class_map = kwargs.get("class_map", None)
        if class_map is None:
            raise ValueError("class_map is required for [RfdetrTRT]")
        self.class_map = {int(k): str(v) for k, v in class_map.items()}
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

        np.copyto(bufs["inputs"][0]["host"], image.ravel())

        cuda.memcpy_htod_async(bufs["inputs"][0]["device"], bufs["inputs"][0]["host"], stream)

        for b in bufs["all"]:
            self.context.set_tensor_address(b["name"], int(b["device"]))

        self.context.execute_async_v3(stream_handle=stream.handle)
        results = []
        for out in bufs["outputs"]:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
            results.append(out)
        stream.synchronize()

        self.current_idx = 1 - self.current_idx
        return [r["host"].reshape(r["shape"]) for r in results]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def postprocess(self, outputs, image, **kwargs) -> Results:
        orig_h, orig_w = image.shape[:2]
        configs = kwargs.get("configs")
        if configs is None:
            self.logger.warning("configs is None. Using default value of 1.0 for all classes.")
            configs = {k if isinstance(k, str) else v: 1.0 for k, v in self.class_map.items()}
        if isinstance(configs, dict) is False and isinstance(configs, (int, float)):
            configs = {k if isinstance(k, str) else v: configs for k, v in self.class_map.items()}
        else:
            self.logger.warning(
                "configs should be a dictionary of class confidence thresholds. Using default value of 1.0 for all classes."
            )

        if len(outputs) < 2:
            raise RuntimeError(f"Expected at least 2 output tensors, got {len(outputs)}")

        dets_data = outputs[0][0]
        labels_data = outputs[1][0]
        scores_all = self.sigmoid(labels_data)

        max_scores = np.max(scores_all, axis=1)
        max_class_indices = np.argmax(scores_all, axis=1)

        max_class_indices = self.class_map_func(max_class_indices)

        mask = max_scores >= np.vectorize(configs.get)(max_class_indices, 1.0)

        filtered_scores = max_scores[mask]
        filtered_classes = max_class_indices[mask]
        filtered_dets = dets_data[mask]  # shape (N, 4)

        if filtered_dets.shape[0] == 0:
            return Results(boxes=[], scores=[], classes=[])

        cx = filtered_dets[:, 0] * orig_w
        cy = filtered_dets[:, 1] * orig_h
        w = filtered_dets[:, 2] * orig_w
        h = filtered_dets[:, 3] * orig_h

        x_min = cx - w / 2.0
        y_min = cy - h / 2.0
        x_max = cx + w / 2.0
        y_max = cy + h / 2.0
        final_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

        return Results(boxes=torch.from_numpy(final_boxes), scores=torch.from_numpy(filtered_scores), classes=filtered_classes)

    def predict(self, image, configs, operators=None, **kwargs):
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
        if operators is None:
            operators = []
        preprocessed_image = self.preprocess(image, **kwargs)
        # inference
        outputs = self.forward(preprocessed_image, **kwargs)
        # postprocess
        results = self.postprocess(outputs, image=image, configs=configs, operators=operators, **kwargs)
        results = results.to_dict(return_tensor=False)
        if results == {}:
            results = {"boxes": [], "scores": [], "classes": []}
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
        boxes = results["boxes"]
        classes = results["classes"]
        scores = results["scores"]

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
                "label": None if hide_label else label,
                "color": None if colormap is None else colormap[classes[i]],
                "line_thickness": line_thickness,
                "hide_bbox": hide_bbox,
            }
            pipeline_utils.plot_one_box(boxes[i], image, None, **args)

        return image


@RfdetrModel.register("pt")
class RfdetrPT(ODBase):
    logger = logging.getLogger("RFDETR")
    logger.setLevel(logging.INFO)

    def __init__(self, model_path: str, device="cuda", **kwargs) -> None:
        self.image_size = kwargs.get("image_size") or [640, 640]
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]

        # load torchscript model
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        class_map = kwargs.get("class_map", None)
        if class_map is None:
            raise ValueError("class_map is required for [RfdetrTRT]")
        self.class_map = {int(k): str(v) for k, v in class_map.items()}
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))
        self.device = device

    def warmup(self):
        """Warm up the model by running a dummy inference."""
        dummy_input = torch.zeros((1, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32).to(self.device)
        self.forward(dummy_input)

    def preprocess(self, image: np.ndarray, **kwargs):
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image in numpy array format.
        """
        input_img = image.astype(np.float32) / 255.0
        means = np.array(self.means, dtype=np.float32)
        stds = np.array(self.stds, dtype=np.float32)
        input_img = (input_img - means) / stds
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, axis=0)
        return input_img

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def postprocess(self, outputs, image, **kwargs) -> Results:
        orig_h, orig_w = image.shape[:2]
        configs = kwargs.get("configs")
        if configs is None:
            self.logger.warning("configs is None. Using default value of 1.0 for all classes.")
            configs = {k if isinstance(k, str) else v: 1.0 for k, v in self.class_map.items()}
        if isinstance(configs, dict) is False and isinstance(configs, (int, float)):
            configs = {k if isinstance(k, str) else v: configs for k, v in self.class_map.items()}
        else:
            self.logger.warning(
                "configs should be a dictionary of class confidence thresholds. Using default value of 1.0 for all classes."
            )

        if len(outputs) < 2:
            raise RuntimeError(f"Expected at least 2 output tensors, got {len(outputs)}")

        dets_data = outputs[0][0]
        labels_data = outputs[1][0]
        if isinstance(labels_data, torch.Tensor):
            labels_data = labels_data.cpu().numpy()
        if isinstance(dets_data, torch.Tensor):
            dets_data = dets_data.cpu().numpy()

        scores_all = self.sigmoid(labels_data)

        max_scores = np.max(scores_all, axis=1)
        max_class_indices = np.argmax(scores_all, axis=1)

        max_class_indices = self.class_map_func(max_class_indices)

        mask = max_scores >= np.vectorize(configs.get)(max_class_indices, 1.0)

        filtered_scores = max_scores[mask]
        filtered_classes = max_class_indices[mask]
        filtered_dets = dets_data[mask]  # shape (N, 4)

        if filtered_dets.shape[0] == 0:
            return Results(boxes=[], scores=[], classes=[])

        cx = filtered_dets[:, 0] * orig_w
        cy = filtered_dets[:, 1] * orig_h
        w = filtered_dets[:, 2] * orig_w
        h = filtered_dets[:, 3] * orig_h

        x_min = cx - w / 2.0
        y_min = cy - h / 2.0
        x_max = cx + w / 2.0
        y_max = cy + h / 2.0
        final_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

        return Results(boxes=torch.from_numpy(final_boxes), scores=torch.from_numpy(filtered_scores), classes=filtered_classes)

    def forward(self, image: np.ndarray, **kwargs) -> list:
        """
        Perform inference.
        """
        if isinstance(image, np.ndarray):
            input_tensor = torch.from_numpy(image)
        else:
            input_tensor = image
        input_tensor = input_tensor.to(self.device).float()
        with torch.no_grad():
            outputs = self.model(input_tensor)
        return outputs

    def predict(self, image, configs, operators=None, **kwargs):
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
        if operators is None:
            operators = []
        preprocessed_image = self.preprocess(image, **kwargs)
        # inference
        outputs = self.forward(preprocessed_image, **kwargs)
        # postprocess
        results = self.postprocess(outputs, image=image, configs=configs, operators=operators, **kwargs)
        results = results.to_dict(return_tensor=False)
        if results == {}:
            results = {"boxes": [], "scores": [], "classes": []}
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
        boxes = results["boxes"]
        classes = results["classes"]
        scores = results["scores"]

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
                "label": None if hide_label else label,
                "color": None if colormap is None else colormap[classes[i]],
                "line_thickness": line_thickness,
                "hide_bbox": hide_bbox,
            }
            pipeline_utils.plot_one_box(boxes[i], image, None, **args)

        return image


@RfdetrModel.register("pth")
class RfdetrPTH(ODBase):
    """RF-DETR PyTorch model wrapper for object detection.

    This class provides an interface for RF-DETR models loaded from PyTorch checkpoint files.
    Supports multiple model variants: nano, medium, large, xlarge, and 2xlarge.
    """

    logger = logging.getLogger("RFDETR")
    logger.setLevel(logging.INFO)

    # Model configuration: {model_type: (default_resolution, model_class)}
    MODEL_CONFIGS = {
        "nano": (384, RFDETRNano),
        "small": (512, RFDETRSmall),
        "medium": (576, RFDETRMedium),
        "large": (704, RFDETRLarge),
        "xlarge": (700, RFDETRXLarge),
        "2xlarge": (880, RFDETR2XLarge),
    }

    DEFAULT_MODEL_TYPE = "medium"
    DEFAULT_CONFIDENCE = 0.5

    def __init__(self, model_path: str, **kwargs) -> None:
        """Initialize RF-DETR model from checkpoint.

        Args:
            model_path: Path to the model checkpoint file (.pth)
            **kwargs: Additional configuration options:
                - model_type: Model variant (nano/medium/large/xlarge/2xlarge). Default: medium
                - device: Device to run on (cuda/cpu). Default: cuda if available
                - image_size: Tuple of (height, width). Default: model-specific

        Raises:
            FileNotFoundError: If model_path does not exist
            ValueError: If model_type is not supported
        """
        self.logger.debug(f"Initializing RfdetrPTH with kwargs: {kwargs}")

        # Validate model path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Determine device
        self.device = self._get_device(kwargs.get("device", "cuda"))

        # Get model type and validate
        model_type = kwargs.get("model_type", self.DEFAULT_MODEL_TYPE).lower()
        if model_type not in self.MODEL_CONFIGS:
            supported = ", ".join(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Unsupported model type: '{model_type}'. Supported types: {supported}")

        # Get model configuration
        default_resolution, model_class = self.MODEL_CONFIGS[model_type]

        # Set image size (use provided or default)
        custom_size = kwargs.get("image_size")
        if custom_size is not None:
            self.image_size = (custom_size[0], custom_size[1])
        else:
            self.image_size = (default_resolution, default_resolution)

        # Initialize model
        self.logger.info(
            f"Loading {model_type} RF-DETR model from {model_path} "
            f"with resolution {self.image_size[0]}x{self.image_size[1]} on {self.device}"
        )
        self.model = model_class(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)

        self.class_names = self.model.class_names
        self.model.optimize_for_inference()

    def _get_device(self, requested_device: str) -> str:
        """Determine the device to use for inference.

        Args:
            requested_device: Requested device (cuda/cpu)

        Returns:
            Device string (cuda/cpu)
        """
        if requested_device == "cuda" and torch.cuda.is_available():
            return "cuda"

        if requested_device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Falling back to CPU.")

        return "cpu"

    def warmup(self) -> None:
        """Warm up the model by running a dummy inference.

        This helps initialize CUDA kernels and prepare the model for actual inference.
        """
        dummy_image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        self.model.predict(dummy_image)
        self.logger.debug("Model warmup completed")

    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Preprocess the input image for the model.

        Note: RF-DETR handles preprocessing internally, so no preprocessing is needed here.

        Args:
            image: Input image in numpy array format (HWC, uint8)
            **kwargs: Additional preprocessing parameters (unused)

        Returns:
            The input image unchanged
        """
        return image

    def forward(self, image: np.ndarray, **kwargs) -> dict:
        """Perform forward pass through the model.

        Args:
            image: Input image in numpy array format (HWC, uint8)
            **kwargs: Additional inference parameters (unused)

        Returns:
            Model predictions containing bounding boxes, scores, and class IDs
        """
        return self.model.predict(image)

    def _construct_results(self, preds, **kwargs) -> Results:
        """Construct Results object from model predictions.

        Args:
            preds: Raw predictions from the model
            **kwargs: Additional parameters (unused)

        Returns:
            Results object containing boxes, scores, and class IDs
        """
        return Results(boxes=preds.xyxy, scores=preds.confidence, classes=preds.class_id)

    def postprocess(self, preds, **kwargs) -> Results:
        """Postprocess the model outputs by applying confidence thresholds and operators.

        Args:
            preds: Raw predictions from the model containing xyxy, confidence, and class_id
            **kwargs: Additional parameters:
                - configs: Confidence threshold (float) or per-class thresholds (dict)
                - operators: List of operators to revert coordinate transformations

        Returns:
            Results object containing filtered boxes, scores, and class names

        Raises:
            ValueError: If configs is not a float or dict
        """
        # Get confidence thresholds
        configs = kwargs.get("configs", self.DEFAULT_CONFIDENCE)
        conf_thresholds = self._parse_confidence_config(configs)

        # Extract predictions
        boxes = np.array(preds.xyxy)
        scores = np.array(preds.confidence)
        class_ids = preds.class_id

        # Early return if no detections
        if len(boxes) == 0:
            return Results(boxes=[], scores=[], classes=[])

        # Convert class IDs to names
        classes = np.array([self.class_names[c + 1] for c in class_ids])

        # Filter by confidence thresholds
        mask = scores >= np.vectorize(conf_thresholds.get)(classes, 1.0)
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # Apply coordinate transformations if operators provided
        operators = kwargs.get("operators", [])
        if operators:
            boxes = pipeline_utils.revert_to_origin(boxes, operators)

        # Convert to tensors if results exist
        return Results(
            boxes=torch.from_numpy(boxes) if len(boxes) > 0 else [],
            scores=torch.from_numpy(scores) if len(scores) > 0 else [],
            classes=classes if len(classes) > 0 else [],
        )

    def _parse_confidence_config(self, configs) -> dict:
        """Parse confidence configuration into a per-class threshold dictionary.

        Args:
            configs: Confidence threshold (float) or per-class thresholds (dict)

        Returns:
            Dictionary mapping class names to confidence thresholds

        Raises:
            ValueError: If configs is not a float or dict
        """
        if isinstance(configs, (int, float)):
            return {cls_name: float(configs) for cls_name in self.class_names.values()}
        elif isinstance(configs, dict):
            return configs
        else:
            raise ValueError(f"configs must be a float or dict, got {type(configs).__name__}")

    @staticmethod
    def annotate_image(
        results: dict,
        image: np.ndarray,
        colormap: dict = None,
        line_thickness: int = None,
        hide_label: bool = False,
        hide_bbox: bool = False,
    ) -> np.ndarray:
        """Annotate detection results on the input image.

        Args:
            results: Detection results dictionary containing:
                - boxes: List or array of bounding boxes [x1, y1, x2, y2]
                - classes: List of class names
                - scores: List of confidence scores
            image: Input image as numpy array
            colormap: Dictionary mapping class names to RGB tuples, e.g.,
                {'class-A': (0, 0, 255), 'class-B': (0, 255, 0)}.
                If None, random colors will be used
            line_thickness: Thickness of bounding box lines. If None, auto-calculated
            hide_label: If True, do not display class labels and scores
            hide_bbox: If True, do not display bounding boxes

        Returns:
            Annotated image as numpy array (copy of input)
        """
        boxes = results["boxes"]
        classes = results["classes"]
        scores = results["scores"]

        # Create a copy of the image
        image = to_numpy(image).copy()

        # Early return if no detections
        if not len(boxes):
            return image

        # Convert boxes to numpy
        boxes = to_numpy(boxes)

        # Convert grayscale to RGB if needed
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Plot boxes and labels
        for i in range(len(boxes)):
            label = f"{classes[i]}: {scores[i]:.2f}"
            plot_args = {
                "label": None if hide_label else label,
                "color": None if colormap is None else colormap.get(classes[i]),
                "line_thickness": line_thickness,
                "hide_bbox": hide_bbox,
            }
            pipeline_utils.plot_one_box(boxes[i], image, None, **plot_args)

        return image

    def predict(self, image: np.ndarray, configs, operators=None, **kwargs) -> dict:
        """Perform object detection on an input image.

        This method runs the complete inference pipeline: forward pass and postprocessing.

        Args:
            image: Input image in numpy array format (HWC, uint8)
            configs: Confidence threshold (float) or per-class thresholds (dict)
            operators: List of coordinate transformation operators to revert. Defaults to None
            **kwargs: Additional inference parameters

        Returns:
            Dictionary containing detection results:
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - scores: List of confidence scores
                - classes: List of class names

        Note:
            Unlike some DETR variants, RF-DETR handles NMS internally,
            so iou, agnostic, and max_det parameters are not used.
        """
        if operators is None:
            operators = []

        # Inference
        outputs = self.forward(image, **kwargs)

        # Postprocess
        results = self.postprocess(outputs, configs=configs, operators=operators, **kwargs)
        results_dict = results.to_dict(return_tensor=False)

        # Ensure consistent return format
        if not results_dict:
            results_dict = {"boxes": [], "scores": [], "classes": []}

        return results_dict
