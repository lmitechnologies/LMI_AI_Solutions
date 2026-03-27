import collections
import logging
import os
import time
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch

import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils
from object_detectors.od_core.model_factory import ModelFactory
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

    def _preprocess_single(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single HWC image to CHW normalized array."""
        input_img = image.astype(np.float32) / 255.0
        means = np.array(self.means, dtype=np.float32)
        stds = np.array(self.stds, dtype=np.float32)
        input_img = (input_img - means) / stds
        return input_img.transpose(2, 0, 1)

    def _normalize_operators(self, operators, batch_size: int) -> list:
        """Normalize operators to a per-image list of operator chains.

        Args:
            operators: None, a single chain (list[dict]), or per-image chains (list[list[dict]]).
            batch_size: Number of images in the batch.

        Returns:
            List of operator chains, one per image.
        """
        if operators is None:
            return [[] for _ in range(batch_size)]
        if len(operators) > 0 and isinstance(operators[0], dict):
            return [operators] * batch_size
        if len(operators) != batch_size:
            raise ValueError(f"operators length ({len(operators)}) must match batch size ({batch_size})")
        return operators

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

    def _postprocess_single(self, dets_data, labels_data, orig_h, orig_w, configs, operators=None):
        """Postprocess a single image's cxcywh logit outputs.

        Args:
            dets_data (np.ndarray): Box coordinates (N, 4) in cxcywh normalized format.
            labels_data (np.ndarray): Class logits (N, num_classes).
            orig_h (int): Original image height.
            orig_w (int): Original image width.
            configs (dict): Per-class confidence thresholds.
            operators (list): Coordinate transform operators to revert.

        Returns:
            Results object with xyxy boxes, scores, and class names.
        """
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

        if operators:
            final_boxes = pipeline_utils.revert_to_origin(final_boxes, operators)

        return Results(boxes=torch.from_numpy(final_boxes), scores=torch.from_numpy(filtered_scores), classes=filtered_classes)

    def postprocess(self, outputs, **kwargs) -> List[Results]:
        """Postprocess cxcywh logit outputs for a batch (used by RfdetrTRT and RfdetrPT).

        Expects outputs[0] to be box coordinates (B, N, 4) in cxcywh normalized format
        and outputs[1] to be class logits (B, N, num_classes). Scales boxes to pixel
        coordinates and filters by per-class confidence thresholds.

        Args:
            outputs: Raw model outputs (list of at least 2 tensors/arrays).
            **kwargs:
                images (list[np.ndarray]): Original images, used to scale boxes.
                configs: Confidence threshold (float) or per-class dict.
                ops_list (list[list]): Per-image coordinate transform operators.

        Returns:
            List of Results objects, one per image.
        """
        images = kwargs["images"]
        configs = self._parse_confidence_config(kwargs.get("configs"), list(self.class_map.values()))
        ops_list = kwargs.get("ops_list", [[] for _ in range(len(images))])

        if len(outputs) < 2:
            raise RuntimeError(f"Expected at least 2 output tensors, got {len(outputs)}")

        all_dets = to_numpy(outputs[0])
        all_labels = to_numpy(outputs[1])

        results = []
        for i, image in enumerate(images):
            orig_h, orig_w = image.shape[:2]
            results.append(self._postprocess_single(all_dets[i], all_labels[i], orig_h, orig_w, configs, ops_list[i]))
        return results

    def predict(
        self,
        image: Union[np.ndarray, List[np.ndarray]],
        configs,
        operators: Optional[Union[List[Dict], List[List[Dict]]]] = None,
        **kwargs,
    ) -> tuple:
        """Run the full inference pipeline: preprocess → forward → postprocess.

        Supports both single image and batch inference.

        Args:
            image: A single HWC image or a list of HWC images.
                All images in a batch must have the same dimensions.
            configs: Confidence threshold (float) or per-class thresholds (dict).
            operators: Operators for coordinate reversion. Accepts:
                - None: no coordinate reversion.
                - list[dict]: a single operator chain, applied to all images.
                - list[list[dict]]: per-image operator chains (length must match batch size).

        Returns:
            (results, time_info)
            results (dict): a dictionary where each value is a list of length B (batch size), e.g., {
                'boxes': [numpy, ...],
                'scores': [numpy, ...],
                'classes': [list of strings, ...],
            }
            time_info (dict): timing info with keys 'preproc', 'proc', 'postproc'.
        """
        time_info = {}

        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        batch_size = len(images)

        ops_list = self._normalize_operators(operators, batch_size)

        # preprocess
        t0 = time.time()
        preprocessed = self.preprocess(images, **kwargs)
        time_info["preproc"] = time.time() - t0

        # forward
        t0 = time.time()
        outputs = self.forward(preprocessed, **kwargs)
        time_info["proc"] = time.time() - t0

        # postprocess
        t0 = time.time()
        list_results = self.postprocess(outputs, images=images, configs=configs, ops_list=ops_list, **kwargs)

        final = collections.defaultdict(list)
        for result in list_results:
            result_dict = result.to_dict(return_tensor=False)
            for k in ("boxes", "scores", "classes"):
                final[k].append(result_dict.get(k, []))
        time_info["postproc"] = time.time() - t0

        return dict(final), time_info

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
class RfdetrModel(ModelFactory, ODBase):
    """Factory that dispatches to the correct backend based on model file extension.

    Supported extensions:
        .engine → RfdetrTRT (TensorRT)
        .pt     → RfdetrPT  (TorchScript)
        .pth    → RfdetrPTH (PyTorch checkpoint via rfdetr library)
    """

    _registry = {}


@RfdetrModel.register("engine")
class RfdetrTRT(RfdetrBase):
    def __init__(self, model_path: str, **kwargs) -> None:
        try:
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as e:
            raise ImportError("pycuda and tensorrt are required for RfdetrTRT. Install them with: pip install pycuda tensorrt") from e

        self.cuda = cuda
        self.trt = trt
        # self.image_size = kwargs.get("image_size", (640, 640))
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.trt_logger)

        with open(model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Inspect input tensor for shape and dynamic batch info
        self.input_name = self.engine.get_tensor_name(0)
        profile_shape = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self.input_dtype = self.trt.nptype(self.engine.get_tensor_dtype(self.input_name))

        if profile_shape:
            self.min_batch = profile_shape[0][0]
            self.opt_batch = profile_shape[1][0]
            self.max_batch = profile_shape[2][0]
            # Use the opt shape (without batch) as the base input shape
            self.input_shape_no_batch = tuple(profile_shape[1][1:])  # (C, H, W)
        else:
            # Static shape engine
            static_shape = self.engine.get_tensor_shape(self.input_name)
            self.min_batch = static_shape[0]
            self.opt_batch = static_shape[0]
            self.max_batch = static_shape[0]
            self.input_shape_no_batch = tuple(static_shape[1:])

        # Gather output tensor metadata
        self.output_info = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
                # Get the shape with -1 for dynamic dims
                shape = self.engine.get_tensor_shape(name)
                self.output_info.append({"name": name, "dtype": dtype, "shape": tuple(shape)})

        self.num_classes = self.output_info[1]["shape"][-1]

        class_map = kwargs.get("class_map")
        if class_map is None:
            raise ValueError("class_map is required for RfdetrTRT")
        self._setup_class_map(class_map)

    def _allocate_for_batch(self, batch_size):
        """Allocate host and device buffers for a given batch size."""
        if batch_size > self.max_batch:
            raise ValueError(f"Batch size {batch_size} exceeds engine max batch size {self.max_batch}")

        # Set input shape with actual batch size
        input_shape = (batch_size, *self.input_shape_no_batch)
        self.context.set_input_shape(self.input_name, input_shape)

        buffers = {}
        # Input buffer
        input_nbytes = int(np.prod(input_shape)) * np.dtype(self.input_dtype).itemsize
        buffers["input_host"] = self.cuda.pagelocked_empty(int(np.prod(input_shape)), self.input_dtype)
        buffers["input_device"] = self.cuda.mem_alloc(input_nbytes)
        buffers["input_shape"] = input_shape

        # Output buffers
        buffers["outputs"] = []
        for info in self.output_info:
            # Replace dynamic batch dim (-1) with actual batch size
            out_shape = tuple(batch_size if d == -1 else d for d in info["shape"])
            out_nbytes = int(np.prod(out_shape)) * np.dtype(info["dtype"]).itemsize
            buffers["outputs"].append(
                {
                    "name": info["name"],
                    "shape": out_shape,
                    "dtype": info["dtype"],
                    "host": self.cuda.pagelocked_empty(int(np.prod(out_shape)), info["dtype"]),
                    "device": self.cuda.mem_alloc(out_nbytes),
                }
            )

        return buffers

    def warmup(self):
        """Warm up the model by running a dummy inference."""
        dummy_input = np.zeros((1, *self.input_shape_no_batch), dtype=self.input_dtype)
        self.forward(np.ascontiguousarray(dummy_input, dtype=self.input_dtype))

    def preprocess(self, images: Union[np.ndarray, List[np.ndarray]], **kwargs) -> np.ndarray:
        """Preprocess input image(s) for TensorRT inference.

        Args:
            images: A single HWC image or a list of HWC images.

        Returns:
            np.ndarray: BCHW array with dtype matching the engine input.
        """
        if isinstance(images, list):
            batch = np.stack([self._preprocess_single(img) for img in images])
        else:
            batch = np.expand_dims(self._preprocess_single(images), axis=0)
        return np.ascontiguousarray(batch, dtype=self.input_dtype)

    def forward(self, image: np.ndarray, **kwargs) -> list:
        """Perform async TensorRT inference with dynamic batch size.

        Args:
            image: BCHW numpy array.

        Returns:
            List of output arrays, each with shape (B, ...).
        """
        batch_size = image.shape[0]
        bufs = self._allocate_for_batch(batch_size)

        # Copy input to device
        np.copyto(bufs["input_host"], image.ravel())
        self.cuda.memcpy_htod_async(bufs["input_device"], bufs["input_host"], self.stream)

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(bufs["input_device"]))
        for out in bufs["outputs"]:
            self.context.set_tensor_address(out["name"], int(out["device"]))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs back
        for out in bufs["outputs"]:
            self.cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        return [out["host"].reshape(out["shape"]) for out in bufs["outputs"]]


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

    def preprocess(self, images: Union[np.ndarray, List[np.ndarray]], **kwargs) -> np.ndarray:
        """Preprocess input image(s) for TorchScript inference.

        Args:
            images: A single HWC image or a list of HWC images.

        Returns:
            np.ndarray: BCHW array.
        """
        if isinstance(images, list):
            return np.stack([self._preprocess_single(img) for img in images])
        return np.expand_dims(self._preprocess_single(images), axis=0)

    def forward(self, image: np.ndarray, **kwargs) -> list:
        """Perform TorchScript inference one image at a time.

        Args:
            image: BCHW numpy array or tensor.

        Returns:
            List of output tensors, each with batch dimension (B, ...).
        """
        if isinstance(image, np.ndarray):
            input_tensor = torch.from_numpy(image)
        else:
            input_tensor = image
        input_tensor = input_tensor.to(self.device).float()

        all_outputs = []
        with torch.no_grad():
            for i in range(input_tensor.shape[0]):
                single_input = input_tensor[i : i + 1]
                outputs = self.model(single_input)
                all_outputs.append(outputs)

        # Concatenate along batch dimension: each element across images
        num_outputs = len(all_outputs[0])
        return [torch.cat([all_outputs[i][j] for i in range(len(all_outputs))], dim=0) for j in range(num_outputs)]


@RfdetrModel.register("pth")
class RfdetrPTH(RfdetrBase):
    """RF-DETR PyTorch model wrapper for object detection.

    This class provides an interface for RF-DETR models loaded from PyTorch checkpoint files.
    Supports multiple model variants: nano, small, medium, large.
    """

    DEFAULT_MODEL_TYPE = "medium"
    DEFAULT_CONFIDENCE = 0.5

    def __init__(self, model_path: str, **kwargs) -> None:
        """Initialize RF-DETR model from checkpoint.

        Args:
            model_path: Path to the model checkpoint file (.pth)
            **kwargs: Additional configuration options:
                - model_type: Model variant (nano/small/medium/large). Default: medium
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
            # "xlarge": (700, RFDETRXLarge),    # require license
            # "2xlarge": (880, RFDETR2XLarge),  # require license
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

    def forward(self, images, **kwargs):
        """Perform forward pass through the model using a for loop (RF-DETR library throws an exception for batch size mismatch).

        Args:
            images: A single image or list of images in numpy array format (HWC, uint8, RGB).

        Returns:
            List of model predictions, one per image.
        """
        if not isinstance(images, list):
            images = [images]
        return [self.model.predict(img) for img in images]

    def _postprocess_pth_single(self, preds, configs, operators=None):
        """Postprocess a single image's rfdetr library predictions.

        Args:
            preds: Predictions from rfdetr containing xyxy, confidence, and class_id.
            configs (dict): Per-class confidence thresholds.
            operators (list): Coordinate transform operators to revert.

        Returns:
            Results object with filtered boxes, scores, and class names.
        """
        boxes = np.array(preds.xyxy)
        scores = np.array(preds.confidence)
        class_ids = preds.class_id

        if len(boxes) == 0:
            return Results(boxes=[], scores=[], classes=[])

        classes = np.array([self.class_names[c] for c in class_ids])

        mask = scores >= np.vectorize(configs.get)(classes, 1.0)
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        if operators:
            boxes = pipeline_utils.revert_to_origin(boxes, operators)

        return Results(
            boxes=torch.from_numpy(boxes) if len(boxes) > 0 else [],
            scores=torch.from_numpy(scores) if len(scores) > 0 else [],
            classes=classes if len(classes) > 0 else [],
        )

    def predict(
        self,
        image: Union[np.ndarray, List[np.ndarray]],
        configs=None,
        operators: Optional[Union[List[Dict], List[List[Dict]]]] = None,
        **kwargs,
    ) -> tuple:
        """Run inference using the rfdetr library's native batch support.

        Args:
            image: A single HWC image or a list of HWC images (RGB, uint8).
            configs: Confidence threshold (float) or per-class thresholds (dict).
            operators: Operators for coordinate reversion. Accepts:
                - None: no coordinate reversion.
                - list[dict]: a single operator chain, applied to all images.
                - list[list[dict]]: per-image operator chains (length must match batch size).

        Returns:
            (results, time_info)
            results (dict): dict where each value is a list of length B.
            time_info (dict): timing info with keys 'preproc', 'proc', 'postproc'.
        """
        time_info = {}

        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        batch_size = len(images)

        ops_list = self._normalize_operators(operators, batch_size)

        configs = self._parse_confidence_config(
            configs if configs is not None else self.DEFAULT_CONFIDENCE,
            self.class_names.values(),
        )

        # forward (rfdetr library handles preprocessing internally)
        time_info["preproc"] = 0.0
        t0 = time.time()
        preds = self.forward(images, **kwargs)
        time_info["proc"] = time.time() - t0

        # postprocess
        t0 = time.time()
        final = collections.defaultdict(list)
        for pred, ops in zip(preds, ops_list):
            result = self._postprocess_pth_single(pred, configs, ops)
            result_dict = result.to_dict(return_tensor=False)
            for k in ("boxes", "scores", "classes"):
                final[k].append(result_dict.get(k, []))
        time_info["postproc"] = time.time() - t0

        return dict(final), time_info
