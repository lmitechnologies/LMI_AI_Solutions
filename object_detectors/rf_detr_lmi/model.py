import inspect
import logging
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES
from rfdetr.models.postprocess import PostProcess

from lmi_common.model_factory import ModelFactory
from lmi_common.onnx_engine import ONNXEngine
from lmi_common.trt_engine import TRTEngine
from lmi_utils.image_utils.types import ImageLike
from object_detectors.od_core.od_base import ODBase
from object_detectors.od_core.results import Results
from object_detectors.rf_detr_lmi.metadata import RfdetrMetadata

# Added in rfdetr 1.9.1; passing it to older versions raises TypeError.
_POSTPROCESS_TAKES_SCORE_THRESHOLD = "score_threshold" in inspect.signature(PostProcess.forward).parameters


class RfdetrBase(ODBase):
    """Shared base class for all RF-DETR model backends.

    Provides common utilities: class-map setup, confidence config parsing,
    the predict pipeline template, and image annotation.
    """

    logger = logging.getLogger("RFDETR")

    RESIZE_PRESERVE_ASPECT = False  # stretch (see preprocess())

    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def _init_common(self) -> None:
        """Initialize normalization constants shared by all backends; each backend sets its own postprocessor."""
        self.means = self.MEANS
        self.stds = self.STDS

    @staticmethod
    def _num_classes_from_checkpoint(model_path: str) -> Optional[int]:
        """Read the class count the checkpoint's detection head was trained with.

        rfdetr constructs the model with its config-default class count and then aligns the head to
        the checkpoint, logging a mismatch warning on every load. Passing the checkpoint's own count
        as num_classes makes the sizes match up front and skips that warning. The class_embed bias
        holds one row per class plus a background slot, so the class count is its length minus one.

        Returns None when the key is absent (an unexpected checkpoint layout), leaving the caller to
        fall back to rfdetr's default alignment.
        """
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        class_bias = checkpoint.get("model", {}).get("class_embed.bias")
        if class_bias is None:
            # PyTorch Lightning native .ckpt layout prefixes model weights with "model."
            class_bias = checkpoint.get("state_dict", {}).get("model.class_embed.bias")
        if class_bias is None:
            return None
        return class_bias.shape[0] - 1

    @staticmethod
    def _class_map_from_names(class_names, num_logit_slots: Optional[int] = None) -> dict:
        """Map an ordered list of class names to the label ids the model emits.

        Mirrors rfdetr's own predict(): COCO-pretrained checkpoints emit sparse COCO category ids (1-90) while
        fine-tuned models emit 0-based indices. rfdetr distinguishes them by the COCO names plus a num_classes
        wider than the name list; num_logit_slots is that count (excluding background), when a backend can supply it.
        """
        names = list(class_names)
        if names == list(COCO_CLASS_NAMES) and (num_logit_slots is None or num_logit_slots > len(names)):
            return {coco_id: names[i] for i, coco_id in enumerate(COCO_CLASSES) if i < len(names)}
        return dict(enumerate(names))

    @staticmethod
    def _resolve_class_map(
        model_path: str, embedded_names: Optional[List[str]], provided: Optional[dict], num_logit_slots: Optional[int] = None
    ) -> dict:
        """Resolve class_map from the class names embedded in the model file, or from an explicit override.

        Args:
            model_path: Path to the model file, for the error message.
            embedded_names: Class names recorded at export, or None for a model carrying none.
            provided: Override class_map, used as-is when given. Needed only for a model exported without embedded names.
            num_logit_slots: Detection-head class count, used to spot COCO-pretrained checkpoints.

        Returns:
            Dict mapping int index to str class name.

        Raises:
            ValueError: If the model file carries no class names and no override was given.
        """
        if provided is not None:
            return provided
        if not embedded_names:
            raise ValueError(
                f"class_map not provided and no class names embedded in {model_path}. "
                "Pass class_map explicitly, or re-export the model with rf_detr_lmi/cli.py to embed them."
            )
        RfdetrBase.logger.info(f"Loaded {len(embedded_names)} class names embedded in {model_path}")
        return RfdetrBase._class_map_from_names(embedded_names, num_logit_slots)

    def warmup(self) -> None:
        """Warm up the model by running a dummy inference. Requires self.image_size."""
        batch = self.fixed_batch_size or 1
        dummy_input = torch.zeros((batch, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32).to(self.device)
        self.forward(dummy_input)

    def _to_float_chw(self, image: ImageLike) -> torch.Tensor:
        """Convert an HWC uint8 image (ndarray or tensor) to a CHW float [0, 1] tensor on the model device."""
        if isinstance(image, np.ndarray):
            return F.to_tensor(image).to(self.device)
        return image.permute(2, 0, 1).to(self.device).float() / 255.0

    def preprocess(self, images: List[ImageLike]) -> torch.Tensor:
        """Preprocess input image(s) to a BCHW normalized tensor.

        RF-DETR is trained with a square (stretch) resize, applied without antialiasing on the float tensor.
        """
        if not isinstance(images, list):
            images = [images]

        def resize_stretch(img_tensor, size):
            # antialias=False mirrors the cv2.INTER_LINEAR training resize; rfdetr's predict() only matches from 1.9.0.
            return F.resize(img_tensor, [size[0], size[1]], antialias=False)

        tensors = [self._to_float_chw(img) for img in images]
        tensors = self._fit_to_input_size(tensors, preserve_aspect=False, resize_fn=resize_stretch, channels_first=True)
        return torch.stack([F.normalize(t, self.means, self.stds) for t in tensors])

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
                operators (list): per-image-sliced preprocessing history (one slice per image).
                    Produced by ODBase._normalize_operators from the unified history.
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
        extra = {"score_threshold": self._mask_score_floor(configs)} if _POSTPROCESS_TAKES_SCORE_THRESHOLD else {}
        rs = self.postprocessor(return_predictions, target_sizes=target_sizes, **extra)
        return [self._postprocess_single(r, configs, operators[i], return_segments) for i, r in enumerate(rs)]

    @staticmethod
    def _mask_score_floor(configs: dict) -> float:
        """Lowest score _apply_confidence_filter can keep, letting PostProcess skip masks it would discard anyway.

        PostProcess tests ``score > floor`` while ours is ``score >= threshold``, so step one float32 below.
        """
        smallest = min(configs.values(), default=1.0)
        return float(np.nextafter(np.float32(smallest), np.float32("-inf")))


class RfdetrModel(ModelFactory, ODBase):
    """Factory that dispatches to the correct backend based on model file extension.

    Supported extensions:
        .engine → RfdetrTRT (TensorRT)
        .onnx   → RfdetrONNX (ONNX Runtime)
        .pth    → RfdetrPTH (PyTorch checkpoint via rfdetr library)
    """

    _registry = {}


class _RfdetrEngine(RfdetrBase):
    """Shared base for the compiled-graph backends. Subclasses set `_engine_cls`.

    Both engines are built from the same rfdetr ONNX export, whose outputs are ordered
    (dets, labels[, masks]) — the order RfdetrBase.postprocess unpacks.

    Class names come from the metadata embedded in the model file at export; the class_map argument is
    an override, needed only for a model exported without it.
    """

    _engine_cls = None

    def __init__(self, model_path: str, class_map: Optional[dict] = None, device: str = "cuda", **kwargs) -> None:
        self._setup_device(device)
        self.engine = self._engine_cls(model_path, device=str(self.device))
        if len(self.engine._input_names) != 1:
            raise ValueError(f"Expected a single-input {type(self).__name__} model, got inputs: {self.engine._input_names}")
        self.input_shape = self.engine.input_shape  # (C, H, W)
        self.image_size = list(self.input_shape[-2:])  # (H, W) — used by the input-size guard
        self.input_dtype = self.engine.input_dtype
        if not self.engine.is_dynamic:
            self.fixed_batch_size = self.engine.max_batch
        self._init_common()
        metadata = RfdetrMetadata.from_engine(self.engine.metadata, model_path)
        # No rfdetr model here, so rebuild its postprocessor from the num_select recorded at export.
        self.postprocessor = PostProcess(num_select=metadata.num_select)
        # Outputs are (dets, labels[, masks]); labels is (B, queries, num_classes + background).
        num_logit_slots = self.engine._output_buffers[1].shape[-1] - 1
        self._setup_class_map(self._resolve_class_map(model_path, metadata.class_names, class_map, num_logit_slots))

    def warmup(self):
        """Warm up the model by running a dummy inference."""
        batch = self.fixed_batch_size or 1
        dummy = torch.zeros((batch, *self.input_shape), dtype=self.input_dtype, device=self.device)
        self.forward(dummy)

    def forward(self, image: torch.Tensor) -> list:
        """Run the engine.

        Args:
            image: BCHW tensor with batch_size <= self.engine.max_batch.

        Returns:
            List of output tensors on self.device, each with shape (B, ...).
        """
        return self.engine.infer(image)

    def release(self) -> None:
        self.engine.release()


@RfdetrModel.register("engine")
class RfdetrTRT(_RfdetrEngine):
    """TensorRT backend (CUDA only)."""

    _engine_cls = TRTEngine

    def __init__(self, model_path: str, class_map: Optional[dict] = None, **kwargs) -> None:
        kwargs.pop("device", None)
        super().__init__(model_path, class_map=class_map, device="cuda", **kwargs)


@RfdetrModel.register("onnx")
class RfdetrONNX(_RfdetrEngine):
    """ONNX Runtime backend. Runs on CUDA or CPU."""

    _engine_cls = ONNXEngine


@RfdetrModel.register("pth")
class RfdetrPTH(RfdetrBase):
    """RF-DETR PyTorch model wrapper for object detection.

    This class provides an interface for RF-DETR models loaded from PyTorch checkpoint files.
    Supports multiple model variants: nano, small, medium, large.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str,
        class_map: Optional[dict] = None,
        device: str = "cuda",
        image_size: Optional[tuple] = None,
        batch_size: int = 1,
    ) -> None:
        """Initialize RF-DETR model from checkpoint.

        Args:
            model_path: Path to the model checkpoint file (.pth)
            model_type: Model variant (nano/small/medium/large/ or seg-nano/seg-small/seg-medium/seg-large/seg-xlarge/seg-2xlarge).
            class_map: Dict mapping class indices to class names. Defaults to the checkpoint's built-in class names; pass it
                only to override the index mapping — values must still exactly match the model's class names.
            device: Device to run on (cuda/cpu). Default: cuda if available
            image_size: Tuple of (height, width); must be square. Default: model-specific
            batch_size: Number of images per forward pass. Fixed at load time, so predict() chunks to it and zero-pads
                a short final chunk. Default: 1

        Raises:
            FileNotFoundError: If model_path does not exist
            ValueError: If model_type is not supported, image_size is not square, or batch_size is not positive
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
            RFDETRSegXLarge,
            RFDETRSmall,
        )

        model_configs = {
            "nano": (384, RFDETRNano),
            "small": (512, RFDETRSmall),
            "medium": (576, RFDETRMedium),
            "large": (704, RFDETRLarge),
            # "xlarge": (700, RFDETRXLarge),    # require license
            # "2xlarge": (880, RFDETR2XLarge),  # require license
            "seg-nano": (312, RFDETRSegNano),
            "seg-small": (384, RFDETRSegSmall),
            "seg-medium": (432, RFDETRSegMedium),
            "seg-large": (504, RFDETRSegLarge),
            "seg-xlarge": (624, RFDETRSegXLarge),
            "seg-2xlarge": (768, RFDETRSeg2XLarge),
        }

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        if batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer; got {batch_size}")

        self._setup_device(device)

        model_type = model_type.lower()
        if model_type not in model_configs:
            supported = ", ".join(model_configs.keys())
            raise ValueError(f"Unsupported model type: '{model_type}'. Supported types: {supported}")

        default_resolution, model_class = model_configs[model_type]

        if image_size is not None:
            # rfdetr takes a single resolution and traces the graph at it; a non-square image_size
            # would only fail once the first frame reaches the traced model.
            if image_size[0] != image_size[1]:
                raise ValueError(f"RF-DETR runs at a square resolution; got image_size=({image_size[0]}, {image_size[1]})")
            self.image_size = (image_size[0], image_size[1])
        else:
            self.image_size = (default_resolution, default_resolution)

        self.logger.info(
            f"Loading {model_type} RF-DETR model from {model_path} "
            f"with resolution {self.image_size[0]}x{self.image_size[1]} on {self.device}"
        )
        model_kwargs = {"pretrain_weights": model_path, "resolution": self.image_size[0], "device": self.device}
        num_classes = self._num_classes_from_checkpoint(model_path)
        if num_classes is not None:
            model_kwargs["num_classes"] = num_classes
        self.model = model_class(**model_kwargs)
        if class_map is None:
            # Same quantity rfdetr's own predict() keys the sparse-COCO decision on.
            class_map = self._class_map_from_names(self.model.class_names, getattr(self.model.model.args, "num_classes", None))
        elif set(class_map.values()) != set(self.model.class_names):
            raise ValueError(
                f"Provided class_map values {set(class_map.values())} do not match model class names {set(self.model.class_names)}"
            )
        self._setup_class_map(class_map)
        # The traced graph has batch_size baked in, so fixed_batch_size must match it.
        # rfdetr renamed optimize_for_inference() to inference() in 1.9.0; the old name is still an alias there.
        optimize = getattr(self.model, "inference", None) or self.model.optimize_for_inference
        optimize(batch_size=batch_size)
        self.fixed_batch_size = batch_size
        self._init_common()
        # Built from the checkpoint's resolved config, and survives the optimize call above.
        self.postprocessor = self.model.model.postprocess

    def forward(self, image: torch.Tensor, **kwargs) -> list:
        """Perform inference on a batch of images.

        Args:
            image: BCHW tensor with batch size exactly self.fixed_batch_size.

        Returns:
            List of output tensors, each with batch dimension (B, ...).
        """
        return self.model.model.inference_model(image)
