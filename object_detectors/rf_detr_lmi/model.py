import logging
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from rfdetr.models.postprocess import PostProcess

from lmi_common.model_factory import ModelFactory
from lmi_common.trt_engine import TRTEngine
from lmi_utils.image_utils.types import ImageBatch, ImageLike
from object_detectors.od_core.object_detector_registry import ObjectDetectorRegistry
from object_detectors.od_core.od_base import ODBase
from object_detectors.od_core.results import Results


class RfdetrBase(ODBase):
    """Shared base class for all RF-DETR model backends.

    Provides common utilities: class-map setup, confidence config parsing,
    the predict pipeline template, and image annotation.
    """

    logger = logging.getLogger("RFDETR")

    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def _init_common(self) -> None:
        """Initialize normalization constants and postprocessor shared by all backends."""
        self.means = self.MEANS
        self.stds = self.STDS
        self.postprocessor = PostProcess(num_select=300)

    def warmup(self) -> None:
        """Warm up the model by running a dummy inference. Requires self.image_size."""
        dummy_input = torch.zeros((1, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32).to(self.device)
        self.forward(dummy_input)

    def _preprocess_single(self, image: ImageLike) -> torch.Tensor:
        """Preprocess a single HWC uint8 image: convert to CHW float [0, 1], normalize, move to device."""
        if isinstance(image, np.ndarray):
            img_tensor = F.to_tensor(image).to(self.device)
        else:
            img_tensor = image.permute(2, 0, 1).to(self.device).float() / 255.0
        return F.normalize(img_tensor, self.means, self.stds)

    def preprocess(self, images: ImageBatch) -> torch.Tensor:
        """Preprocess input image(s) to BCHW normalized tensor."""
        if isinstance(images, list):
            return torch.stack([self._preprocess_single(img) for img in images])
        return torch.unsqueeze(self._preprocess_single(images), dim=0)

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

    def _postprocess_single(self, output, configs, ops, return_segments) -> Results:
        """Postprocess a single image's decoded output from PostProcess.

        Args:
            output (dict): Decoded output with keys 'boxes', 'scores', 'labels',
                and optionally 'masks'. Values are tensors on self.device.
            configs (dict): Per-class confidence thresholds.
            ops (list): Coordinate transform operators to revert.
            return_segments (bool): Whether to convert masks to segments.

        Returns:
            Results object with filtered xyxy boxes, scores, class names, optional masks, and optional segments.
        """
        classes = self.class_map_func(output["labels"].cpu().numpy())
        is_seg = "masks" in output  # model specific
        boxes, scores, classes, masks, _ = self._apply_confidence_filter(
            output["scores"], output["boxes"], classes, configs, masks=output.get("masks")
        )

        # masks from rf-detr are (N, 1, H, W); squeeze to (N, H, W) for downstream use
        if len(masks) > 0:
            masks = masks.squeeze(1)
        segments = (
            [torch.from_numpy(s).to(self.device) for s in self._masks_to_segments(masks)] if len(masks) > 0 and return_segments else None
        )
        result = Results(
            boxes=boxes,
            scores=scores,
            classes=classes,
            masks=masks if len(masks) > 0 else None,
            segments=segments,
            is_seg=is_seg,
        )
        return self._apply_revert_to_result(result, ops)

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
                return_segments (bool): Whether to convert masks to segments.

        Returns:
            List of Results objects, one per image.
        """
        images = kwargs["images"]
        configs = self._parse_confidence_config(kwargs.get("configs"), list(self.class_map.values()))
        operators = kwargs.get("operators", [[] for _ in range(len(images))])
        return_segments = kwargs.get("return_segments", True)

        if len(outputs) < 2:
            raise RuntimeError(f"Expected at least 2 output tensors, got {len(outputs)}")

        return_predictions = {
            "pred_logits": outputs[1],
            "pred_boxes": outputs[0],
        }
        if len(outputs) == 3:
            return_predictions["pred_masks"] = outputs[2]

        orig_sizes = [img.shape[:2] for img in images]
        target_sizes = torch.tensor(orig_sizes, device=self.device)
        rs = self.postprocessor(return_predictions, target_sizes=target_sizes)
        return [self._postprocess_single(r, configs, operators[i], return_segments) for i, r in enumerate(rs)]


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
        self._setup_device("cuda")
        self.trt = TRTEngine(model_path, device=str(self.device))
        if len(self.trt._input_names) != 1:
            raise ValueError(f"Expected a single-input TRT engine, got inputs: {self.trt._input_names}")
        self.input_shape = self.trt.input_shape  # (C, H, W)
        self.input_dtype = self.trt.input_dtype
        if not self.trt.is_dynamic:
            self.fixed_batch_size = self.trt.max_batch
        self._init_common()
        self._setup_class_map(class_map)

    def warmup(self):
        """Warm up the model by running a dummy inference."""
        dummy = torch.zeros((self.fixed_batch_size, *self.input_shape), dtype=self.input_dtype, device=self.device)
        self.forward(dummy)

    def forward(self, image: torch.Tensor) -> list:
        """Perform TensorRT inference.

        Args:
            image: BCHW tensor with batch_size <= self.fixed_batch_size.

        Returns:
            List of output tensors on self.device, each with shape (B, ...).
        """
        return self.trt.infer(image)


@RfdetrModel.register("pt")
class RfdetrPT(RfdetrBase):
    def __init__(self, model_path: str, class_map: dict, device: str = "cuda", image_size: Optional[List[int]] = None) -> None:
        self.image_size = image_size or [640, 640]
        self._setup_device(device)
        self._init_common()

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.fixed_batch_size = 1
        self._setup_class_map(class_map)

    def forward(self, image: torch.Tensor, **kwargs) -> list:
        """Perform TorchScript inference on a single image (batch=1).

        Args:
            image: BCHW tensor with batch size 1.

        Returns:
            List of output tensors, each with batch dimension (1, ...).
        """
        return self.model(image)


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
        from rfdetr import (
            RFDETRLarge,
            RFDETRMedium,
            RFDETRNano,
            RFDETRSeg2XLarge,
            RFDETRSegLarge,
            RFDETRSegMedium,
            RFDETRSegNano,
            RFDETRSegSmall,
            RFDETRSmall,
        )

        model_configs = {
            "nano": (384, RFDETRNano),
            "small": (512, RFDETRSmall),
            "medium": (576, RFDETRMedium),
            "large": (704, RFDETRLarge),
            # "xlarge": (700, RFDETRXLarge),    # require license
            # "2xlarge": (880, RFDETR2XLarge),  # require license
            "seg-nano": (384, RFDETRSegNano),
            "seg-small": (512, RFDETRSegSmall),
            "seg-medium": (576, RFDETRSegMedium),
            "seg-large": (704, RFDETRSegLarge),
            "seg-xlarge": (700, RFDETRSeg2XLarge),
        }

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self._setup_device(device)

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
        self.fixed_batch_size = 1
        self._init_common()

    def forward(self, image: torch.Tensor, **kwargs) -> list:
        """Perform inference on a single image (batch=1).

        Args:
            image: BCHW tensor with batch size 1.

        Returns:
            List of output tensors, each with batch dimension (1, ...).
        """
        return self.model.model.inference_model(image)
