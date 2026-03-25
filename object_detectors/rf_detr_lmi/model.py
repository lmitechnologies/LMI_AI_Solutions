import logging
import os

import cv2
import numpy as np
import torch

import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils
from object_detectors.od_core.object_detector_registry import ObjectDetectorRegistry
from object_detectors.od_core.od_base import ODBase
from object_detectors.od_core.results import Results


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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RfdetrBase(ODBase):
    """Shared base class for all RF-DETR model backends.

    Provides common utilities: class-map setup, confidence config parsing,
    the predict pipeline template, and image annotation.
    """

    logger = logging.getLogger("RFDETR")

    def _setup_class_map(self, class_map: dict) -> None:
        """Initialize class_map and a vectorized name-lookup from a {int_id: str_name} mapping."""
        self.class_map = {int(k): str(v) for k, v in class_map.items()}
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))

    def _parse_confidence_config(self, configs, class_names) -> dict:
        """Parse configs into a per-class threshold dict.

        Args:
            configs: None (defaults to 1.0), float/int (global threshold), or dict (per-class).
            class_names: Iterable of class name strings used when building a uniform dict.

        Returns:
            dict mapping class name → confidence threshold.
        """
        if configs is None:
            self.logger.warning("configs is None. Using default value of 1.0 for all classes.")
            return {name: 1.0 for name in class_names}
        if isinstance(configs, (int, float)):
            return {name: float(configs) for name in class_names}
        if isinstance(configs, dict):
            return configs
        raise ValueError(f"configs must be a float or dict, got {type(configs).__name__}")

    def postprocess(self, outputs, **kwargs) -> Results:
        """Postprocess cxcywh logit outputs (used by RfdetrTRT and RfdetrPT).

        Expects outputs[0][0] to be box coordinates (N, 4) in cxcywh normalized format
        and outputs[1][0] to be class logits (N, num_classes). Scales boxes to pixel
        coordinates and filters by per-class confidence thresholds.

        Args:
            outputs: Raw model outputs (list of at least 2 tensors/arrays).
            **kwargs:
                image (np.ndarray): Original image, used to scale boxes.
                configs: Confidence threshold (float) or per-class dict.
                operators (list): Coordinate transform operators to revert.

        Returns:
            Results object with xyxy boxes, scores, and class names.
        """
        image = kwargs["image"]
        orig_h, orig_w = image.shape[:2]
        configs = self._parse_confidence_config(kwargs.get("configs"), list(self.class_map.values()))

        if len(outputs) < 2:
            raise RuntimeError(f"Expected at least 2 output tensors, got {len(outputs)}")

        dets_data = to_numpy(outputs[0][0])
        labels_data = to_numpy(outputs[1][0])
        scores_all = sigmoid(labels_data)

        max_scores = np.max(scores_all, axis=1)
        max_class_indices = np.argmax(scores_all, axis=1)
        max_class_indices = self.class_map_func(max_class_indices)

        mask = max_scores >= np.vectorize(configs.get)(max_class_indices, 1.0)
        filtered_scores = max_scores[mask]
        filtered_classes = max_class_indices[mask]
        filtered_dets = dets_data[mask]

        if filtered_dets.shape[0] == 0:
            return Results(boxes=[], scores=[], classes=[])

        cx = filtered_dets[:, 0] * orig_w
        cy = filtered_dets[:, 1] * orig_h
        w = filtered_dets[:, 2] * orig_w
        h = filtered_dets[:, 3] * orig_h
        x_min, y_min = cx - w / 2.0, cy - h / 2.0
        x_max, y_max = cx + w / 2.0, cy + h / 2.0
        final_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

        operators = kwargs.get("operators", [])
        if operators:
            final_boxes = pipeline_utils.revert_to_origin(final_boxes, operators)

        return Results(boxes=torch.from_numpy(final_boxes), scores=torch.from_numpy(filtered_scores), classes=filtered_classes)

    def predict(self, image, configs, operators=None, **kwargs) -> dict:
        """Run the full inference pipeline: preprocess → forward → postprocess.

        Args:
            image (np.ndarray): Input image in HWC uint8 format.
            configs: Confidence threshold (float) or per-class thresholds (dict).
            operators (list, optional): Coordinate transform operators to revert.

        Returns:
            dict with keys 'boxes', 'scores', 'classes'.
        """
        if operators is None:
            operators = []
        preprocessed = self.preprocess(image, **kwargs)
        outputs = self.forward(preprocessed, **kwargs)
        results = self.postprocess(outputs, image=image, configs=configs, operators=operators, **kwargs)
        results_dict = results.to_dict(return_tensor=False)
        if not results_dict:
            results_dict = {"boxes": [], "scores": [], "classes": []}
        return results_dict

    @staticmethod
    def annotate_image(results, image, colormap=None, line_thickness=None, hide_label=False, hide_bbox=False) -> np.ndarray:
        """Annotate detection results on the image.

        Args:
            results (dict): Detection results with keys 'boxes', 'classes', 'scores'.
            image (np.ndarray): Input image.
            colormap (dict, optional): Maps class names to RGB tuples. Defaults to None (random colors).
            line_thickness (int, optional): Bounding box line thickness. Defaults to None.
            hide_label (bool): If True, suppress class/score labels.
            hide_bbox (bool): If True, suppress bounding boxes.

        Returns:
            np.ndarray: Annotated copy of the image.
        """
        boxes = results["boxes"]
        classes = results["classes"]
        scores = results["scores"]

        image = to_numpy(image).copy()
        if not len(boxes):
            return image

        boxes = to_numpy(boxes)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

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


@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=["v1"], model_names=["rfdetr"], tasks=["od", "seg", "instancesegmentation", "objectdetection"], frameworks=["rfdetr"]
    )
)
class RfdetrModel(ODBase):
    """Factory that dispatches to the correct backend based on model file extension.

    Supported extensions:
        .engine → RfdetrTRT (TensorRT)
        .pt     → RfdetrPT  (TorchScript)
        .pth    → RfdetrPTH (PyTorch checkpoint via rfdetr library)
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
class RfdetrTRT(RfdetrBase):
    def __init__(self, model_path: str, device="cuda", fp16=False, **kwargs) -> None:
        try:
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as e:
            raise ImportError("pycuda and tensorrt are required for RfdetrTRT. Install them with: pip install pycuda tensorrt") from e

        self.cuda = cuda
        self.trt = trt
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

        class_map = kwargs.get("class_map")
        if class_map is None:
            raise ValueError("class_map is required for RfdetrTRT")
        self._setup_class_map(class_map)

    def _allocate_buffers(self):
        """Allocates one set of pinned host memory and device memory."""
        inputs = []
        outputs = []
        all_bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))

            host_mem = self.cuda.pagelocked_empty(self.trt.volume(shape), dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)

            binding = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "host": host_mem,
                "device": device_mem,
            }

            all_bindings.append(binding)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                inputs.append(binding)
            else:
                outputs.append(binding)

        return {"inputs": inputs, "outputs": outputs, "all": all_bindings}

    def warmup(self):
        """Warm up the model by running a dummy inference."""
        dummy_input = np.zeros(self.input_shape, dtype=self.input_dtype)
        self.forward(np.ascontiguousarray(dummy_input, dtype=self.input_dtype))

    def preprocess(self, image: np.ndarray, **kwargs):
        """Preprocess the input image for TensorRT inference."""
        input_img = image.astype(np.float32) / 255.0
        means = np.array(self.means, dtype=np.float32)
        stds = np.array(self.stds, dtype=np.float32)
        input_img = (input_img - means) / stds
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, axis=0)
        return np.array([input_img], dtype=self.input_dtype)

    def forward(self, image: np.ndarray, **kwargs) -> list:
        """Perform async TensorRT inference."""
        bufs = self.buffer_sets[self.current_idx]
        stream = self.streams[self.current_idx]

        np.copyto(bufs["inputs"][0]["host"], image.ravel())
        self.cuda.memcpy_htod_async(bufs["inputs"][0]["device"], bufs["inputs"][0]["host"], stream)

        for b in bufs["all"]:
            self.context.set_tensor_address(b["name"], int(b["device"]))

        self.context.execute_async_v3(stream_handle=stream.handle)
        results = []
        for out in bufs["outputs"]:
            self.cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
            results.append(out)
        stream.synchronize()

        self.current_idx = 1 - self.current_idx
        return [r["host"].reshape(r["shape"]) for r in results]


@RfdetrModel.register("pt")
class RfdetrPT(RfdetrBase):
    def __init__(self, model_path: str, device="cuda", **kwargs) -> None:
        self.image_size = kwargs.get("image_size") or [640, 640]
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.device = device

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        class_map = kwargs.get("class_map")
        if class_map is None:
            raise ValueError("class_map is required for RfdetrPT")
        self._setup_class_map(class_map)

    def warmup(self):
        """Warm up the model by running a dummy inference."""
        dummy_input = torch.zeros((1, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32).to(self.device)
        self.forward(dummy_input)

    def preprocess(self, image: np.ndarray, **kwargs):
        """Preprocess the input image for TorchScript inference."""
        input_img = image.astype(np.float32) / 255.0
        means = np.array(self.means, dtype=np.float32)
        stds = np.array(self.stds, dtype=np.float32)
        input_img = (input_img - means) / stds
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, axis=0)
        return input_img

    def forward(self, image: np.ndarray, **kwargs) -> list:
        """Perform TorchScript inference."""
        if isinstance(image, np.ndarray):
            input_tensor = torch.from_numpy(image)
        else:
            input_tensor = image
        input_tensor = input_tensor.to(self.device).float()
        with torch.no_grad():
            return self.model(input_tensor)


@RfdetrModel.register("pth")
class RfdetrPTH(RfdetrBase):
    """RF-DETR PyTorch model wrapper for object detection.

    This class provides an interface for RF-DETR models loaded from PyTorch checkpoint files.
    Supports multiple model variants: nano, small, medium, large, xlarge, and 2xlarge.
    """

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
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

        model_configs = {
            "nano": (384, RFDETRNano),
            "small": (512, RFDETRSmall),
            "medium": (576, RFDETRMedium),
            "large": (704, RFDETRLarge),
            # "xlarge": (700, RFDETRXLarge),
            # "2xlarge": (880, RFDETR2XLarge),
        }

        self.logger.debug(f"Initializing RfdetrPTH with kwargs: {kwargs}")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.device = self._get_device(kwargs.get("device", "cuda"))

        model_type = kwargs.get("model_type", self.DEFAULT_MODEL_TYPE).lower()
        if model_type not in model_configs:
            supported = ", ".join(model_configs.keys())
            raise ValueError(f"Unsupported model type: '{model_type}'. Supported types: {supported}")

        default_resolution, model_class = model_configs[model_type]

        custom_size = kwargs.get("image_size")
        if custom_size is not None:
            self.image_size = (custom_size[0], custom_size[1])
        else:
            self.image_size = (default_resolution, default_resolution)

        self.logger.info(
            f"Loading {model_type} RF-DETR model from {model_path} "
            f"with resolution {self.image_size[0]}x{self.image_size[1]} on {self.device}"
        )
        self.model = model_class(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        self.class_names = self.model.class_names
        self.model.optimize_for_inference()

    def _get_device(self, requested_device: str) -> str:
        """Return 'cuda' if available and requested, otherwise 'cpu'."""
        if requested_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested_device == "cuda":
            self.logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    def warmup(self) -> None:
        """Warm up the model by running a dummy inference."""
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

    def postprocess(self, preds, **kwargs) -> Results:
        """Postprocess rfdetr library predictions by applying confidence thresholds.

        Args:
            preds: Predictions from rfdetr containing xyxy, confidence, and class_id.
            **kwargs:
                configs: Confidence threshold (float) or per-class thresholds (dict).
                operators: List of coordinate transform operators to revert.

        Returns:
            Results object with filtered boxes, scores, and class names.
        """
        configs = self._parse_confidence_config(
            kwargs.get("configs", self.DEFAULT_CONFIDENCE),
            self.class_names.values(),
        )

        boxes = np.array(preds.xyxy)
        scores = np.array(preds.confidence)
        class_ids = preds.class_id

        # Early return if no detections
        if len(boxes) == 0:
            return Results(boxes=[], scores=[], classes=[])

        # Convert class IDs to names
        classes = np.array([self.class_names[c] for c in class_ids])

        mask = scores >= np.vectorize(configs.get)(classes, 1.0)
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
