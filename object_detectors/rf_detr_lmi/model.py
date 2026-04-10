import logging
import os
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from rfdetr.models.postprocess import PostProcess

from object_detectors.od_core.model_factory import ModelFactory
from object_detectors.od_core.object_detector_registry import ObjectDetectorRegistry
from object_detectors.od_core.od_base import ODBase
from object_detectors.od_core.results import Results


class RfdetrBase(ODBase):
    """Shared base class for all RF-DETR model backends.

    Provides common utilities: class-map setup, confidence config parsing,
    the predict pipeline template, and image annotation.
    """

    logger = logging.getLogger("RFDETR")

    def _preprocess_single(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single HWC image to CHW normalized array."""
        input_img = image.astype(np.float32) / 255.0
        means = np.array(self.means, dtype=np.float32)
        stds = np.array(self.stds, dtype=np.float32)
        input_img = (input_img - means) / stds
        return input_img.transpose(2, 0, 1)

    @staticmethod
    def _masks_to_segments(masks) -> List[np.ndarray]:
        """Convert binary masks to contour segments.

        Args:
            masks: (N, H, W) tensor or numpy array of binary masks.

        Returns:
            List of N numpy arrays, each with shape (M, 2) containing (x, y) contour points.
            Returns an empty array for masks with no contours.
        """
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        segments = []
        for mask in masks:
            binary = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                merged = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0).astype(np.float32)
                segments.append(merged)
            else:
                segments.append(np.zeros((0, 2), dtype=np.float32))
        return segments

    def _postprocess_single(self, output, configs, ops, orig_img) -> Results:
        """Postprocess a single image's decoded output from PostProcess.

        Args:
            output (dict): Decoded output with keys 'boxes', 'scores', 'labels',
                and optionally 'masks'. Values are tensors on self.device.
            configs (dict): Per-class confidence thresholds.
            ops (list): Coordinate transform operators to revert.
            orig_img: Original image used to determine tensor vs numpy output.

        Returns:
            Results object with filtered xyxy boxes, scores, class names, optional masks, and optional segments.
        """
        classes = self.class_map_func(output["labels"].cpu().numpy())
        boxes, scores, classes, masks, _ = self._apply_confidence_filter(
            output["scores"], output["boxes"], classes, configs, masks=output.get("masks")
        )

        segments = self._masks_to_segments(masks) if len(masks) > 0 else None
        result = Results(boxes=boxes, scores=scores, classes=classes, masks=masks if len(masks) > 0 else None, segments=segments)
        return self._apply_revert_to_result(result, orig_img, ops)

    def preprocess(self, images: Union[np.ndarray, List[np.ndarray]], **kwargs) -> np.ndarray:
        """Preprocess input image(s) to BCHW normalized array."""
        if isinstance(images, list):
            return np.stack([self._preprocess_single(img) for img in images])
        return np.expand_dims(self._preprocess_single(images), axis=0)

    def postprocess(self, outputs, **kwargs) -> List[Results]:
        """Postprocess outputs for a batch using the rfdetr PostProcess decoder.

        Expects outputs[0] to be box coordinates (B, N, 4) in cxcywh normalized format,
        outputs[1] to be class logits (B, N, num_classes), and optionally outputs[2]
        to be instance masks. Inputs may be tensors or numpy arrays.

        Args:
            outputs: Raw model outputs (list of at least 2 tensors/arrays).
            **kwargs:
                images (list[np.ndarray]): Original images, used to determine target sizes.
                configs: Confidence threshold (float) or per-class dict.
                operators (list[list]): Per-image coordinate transform operators.

        Returns:
            List of Results objects, one per image.
        """
        images = kwargs["images"]
        configs = self._parse_confidence_config(kwargs.get("configs"), list(self.class_map.values()))
        operators = kwargs.get("operators", [[] for _ in range(len(images))])

        if len(outputs) < 2:
            raise RuntimeError(f"Expected at least 2 output tensors, got {len(outputs)}")

        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(self.device)
            return x.to(self.device)

        return_predictions = {
            "pred_logits": to_tensor(outputs[1]),
            "pred_boxes": to_tensor(outputs[0]),
        }
        if len(outputs) == 3:
            return_predictions["pred_masks"] = to_tensor(outputs[2])

        orig_sizes = [img.shape[:2] for img in images]
        target_sizes = torch.tensor(orig_sizes, device=self.device)
        rs = self.postprocessor(return_predictions, target_sizes=target_sizes)
        return [self._postprocess_single(r, configs, operators[i], images[i]) for i, r in enumerate(rs)]


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
    def __init__(self, model_path: str, class_map: dict, **kwargs) -> None:
        # keep kwargs for the required image_size argument in object detection classes.
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError("tensorrt is required for RfdetrTRT. Install it with: pip install tensorrt") from e

        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(trt_logger)

        with open(model_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.torch_stream = torch.cuda.Stream()

        # Inspect input tensor for shape and dynamic batch info
        self.input_name = self.engine.get_tensor_name(0)
        profile_shape = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.torch_dtype = torch.float16 if self.input_dtype == np.float16 else torch.float32

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
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                # Get the shape with -1 for dynamic dims
                shape = self.engine.get_tensor_shape(name)
                self.output_info.append({"name": name, "dtype": dtype, "shape": tuple(shape)})

        self.num_classes = self.output_info[1]["shape"][-1]

        self.device = "cuda"
        self.postprocessor = PostProcess(num_select=300)
        self._setup_class_map(class_map)

        self._buf_cache = {}

    def _get_buffers(self, batch_size):
        """Return cached GPU tensor buffers for a given batch size, allocating on first use."""
        if batch_size in self._buf_cache:
            return self._buf_cache[batch_size]

        if batch_size > self.max_batch:
            raise ValueError(f"Batch size {batch_size} exceeds engine max batch size {self.max_batch}")

        input_shape = (batch_size, *self.input_shape_no_batch)

        buffers = {}
        buffers["input"] = torch.empty(input_shape, dtype=self.torch_dtype, device="cuda")
        buffers["input_shape"] = input_shape

        buffers["outputs"] = []
        for info in self.output_info:
            out_shape = tuple(batch_size if d == -1 else d for d in info["shape"])
            torch_dt = torch.float16 if info["dtype"] == np.float16 else torch.float32
            buffers["outputs"].append(
                {
                    "name": info["name"],
                    "shape": out_shape,
                    "tensor": torch.empty(out_shape, dtype=torch_dt, device="cuda"),
                }
            )

        self._buf_cache[batch_size] = buffers
        return buffers

    def warmup(self):
        """Warm up the model by running a dummy inference."""
        dummy_input = np.zeros((1, *self.input_shape_no_batch), dtype=self.input_dtype)
        self.forward(np.ascontiguousarray(dummy_input, dtype=self.input_dtype))

    def preprocess(self, images: Union[np.ndarray, List[np.ndarray]], **kwargs) -> np.ndarray:
        """Preprocess input image(s) for TensorRT inference, casting to engine input dtype."""
        batch = super().preprocess(images, **kwargs)
        return np.ascontiguousarray(batch, dtype=self.input_dtype)

    def _forward_single_batch(self, image: np.ndarray, **kwargs) -> list:
        """Run TensorRT inference for a single batch that fits within engine limits.

        Args:
            image: BCHW numpy array with batch_size <= self.max_batch.

        Returns:
            List of output numpy arrays, each with shape (B, ...).
        """
        batch_size = image.shape[0]
        bufs = self._get_buffers(batch_size)

        # Set input shape for this batch size
        self.context.set_input_shape(self.input_name, bufs["input_shape"])

        # Copy input numpy array into the pre-allocated GPU tensor
        bufs["input"].copy_(torch.from_numpy(image))

        # Set tensor addresses to GPU memory managed by PyTorch
        self.context.set_tensor_address(self.input_name, bufs["input"].data_ptr())
        for out in bufs["outputs"]:
            self.context.set_tensor_address(out["name"], out["tensor"].data_ptr())

        # Execute on the PyTorch CUDA stream
        self.context.execute_async_v3(stream_handle=self.torch_stream.cuda_stream)
        self.torch_stream.synchronize()

        return [out["tensor"].cpu().numpy() for out in bufs["outputs"]]

    def forward(self, image: np.ndarray, **kwargs) -> list:
        """Perform TensorRT inference with dynamic batch size.

        Args:
            image: BCHW numpy array.

        Returns:
            List of output arrays, each with shape (B, ...).
        """
        batch_size = image.shape[0]

        # If batch exceeds engine max, process in chunks and concatenate
        if batch_size > self.max_batch:
            chunks = [image[i : i + self.max_batch] for i in range(0, batch_size, self.max_batch)]
            all_outputs = [self._forward_single_batch(chunk, **kwargs) for chunk in chunks]
            return [np.concatenate([chunk_out[i] for chunk_out in all_outputs], axis=0) for i in range(len(all_outputs[0]))]

        return self._forward_single_batch(image, **kwargs)


@RfdetrModel.register("pt")
class RfdetrPT(RfdetrBase):
    def __init__(self, model_path: str, class_map: dict, device: str = "cuda", image_size: Optional[List[int]] = None) -> None:
        self.image_size = image_size or [640, 640]
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.device = device

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        self.postprocessor = PostProcess(num_select=300)
        self._setup_class_map(class_map)

    def warmup(self):
        """Warm up the model by running a dummy inference."""
        dummy_input = torch.zeros((1, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32).to(self.device)
        self.forward(dummy_input)

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

    def __init__(
        self,
        model_path: str,
        class_map: dict,
        device: str = "cuda",
        model_type: str = DEFAULT_MODEL_TYPE,
        image_size: Optional[tuple] = None,
    ) -> None:
        """Initialize RF-DETR model from checkpoint.

        Args:
            model_path: Path to the model checkpoint file (.pth)
            class_map: Dict mapping class indices to names (required)
            device: Device to run on (cuda/cpu). Default: cuda if available
            model_type: Model variant (nano/small/medium/large). Default: medium
            image_size: Tuple of (height, width). Default: model-specific

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

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.device = self._get_device(device)

        model_type = model_type.lower()
        if model_type not in model_configs:
            supported = ", ".join(model_configs.keys())
            raise ValueError(f"Unsupported model type: '{model_type}'. Supported types: {supported}")

        default_resolution, model_class = model_configs[model_type]

        if image_size is not None:
            self.image_size = (image_size[0], image_size[1])
        else:
            self.image_size = (default_resolution, default_resolution)

        self.logger.info(
            f"Loading {model_type} RF-DETR model from {model_path} "
            f"with resolution {self.image_size[0]}x{self.image_size[1]} on {self.device}"
        )
        self.model = model_class(pretrain_weights=model_path, resolution=self.image_size[0], device=self.device)
        self._setup_class_map(class_map)
        if set(self.class_map.values()) != set(self.model.class_names):
            raise ValueError(
                f"Provided class_map values {set(self.class_map.values())} do not match model class names {set(self.model.class_names)}"
            )
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

    def _postprocess_single(self, preds, configs, operators, orig_img):
        """Postprocess a single image's rfdetr library predictions.

        Args:
            preds: Predictions from rfdetr containing xyxy, confidence, and class_id.
            configs (dict): Per-class confidence thresholds.
            operators (list): Coordinate transform operators to revert.
            orig_img: Original image used to determine tensor vs numpy output.

        Returns:
            Results object with filtered boxes, scores, class names, optional masks, and optional segments.
        """
        boxes = np.array(preds.xyxy)
        scores = np.array(preds.confidence)
        class_ids = preds.class_id

        if len(boxes) == 0:
            return Results()

        classes = np.array([self.class_map[c] for c in class_ids])
        raw_masks = preds.mask if hasattr(preds, "mask") and preds.mask is not None else None
        boxes, scores, classes, masks, _ = self._apply_confidence_filter(scores, boxes, classes, configs, masks=raw_masks)

        segments = self._masks_to_segments(masks) if len(masks) > 0 else None
        result = Results(
            boxes=boxes if len(boxes) > 0 else None,
            scores=scores if len(scores) > 0 else None,
            classes=classes if len(classes) > 0 else None,
            masks=masks if len(masks) > 0 else None,
            segments=segments,
        )
        return self._apply_revert_to_result(result, orig_img, operators)

    def preprocess(self, images, **kwargs):
        """RF-DETR handles preprocessing internally; pass images through unchanged."""
        return images

    def postprocess(self, outputs, **kwargs) -> List[Results]:
        """Postprocess rfdetr library predictions for a batch.

        Args:
            outputs: List of per-image rfdetr prediction objects.
            **kwargs:
                configs: Confidence threshold (float) or per-class dict. Defaults to DEFAULT_CONFIDENCE.
                operators (list[list]): Per-image coordinate transform operators.

        Returns:
            List of Results objects, one per image.
        """
        configs = self._parse_confidence_config(
            kwargs.get("configs") or self.DEFAULT_CONFIDENCE,
            self.class_map.values(),
        )
        operators = kwargs.get("operators", [[] for _ in range(len(outputs))])
        images = kwargs.get("images", [None] * len(outputs))
        return [self._postprocess_single(pred, configs, ops, img) for pred, ops, img in zip(outputs, operators, images)]
