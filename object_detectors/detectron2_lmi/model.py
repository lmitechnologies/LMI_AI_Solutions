import logging
from typing import Dict, List

import numpy as np
import torch
import torchvision  # noqa: F401

from lmi_common.model_factory import ModelFactory
from lmi_common.trt_engine import TRTEngine
from lmi_utils.image_utils.types import ImageBatch
from lmi_utils.postprocess_utils.mask_utils import mask_to_polygon_cv2, rescale_masks
from object_detectors.od_core.object_detector_registry import ObjectDetectorRegistry
from object_detectors.od_core.od_base import ODBase
from object_detectors.od_core.results import Results


@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=["v0"],
        model_names=["mask_rcnn", "faster_rcnn"],
        tasks=["od", "seg", "instancesegmentation", "objectdetection"],
        frameworks=["detectron2"],
    )
)
class Detectron2Model(ModelFactory, ODBase):
    """Factory that dispatches to the correct backend based on model file extension.

    Supported extensions:
        .engine → Detectron2TRT (TensorRT)
        .pt     → Detectron2PT  (TorchScript)
    """

    _registry = {}


class Detectron2Base(ODBase):
    """Shared base class for all Detectron2 model backends.

    Provides common utilities: the predict pipeline, mask postprocessing, and image annotation.
    """

    def _postprocess_masks(self, raw_masks, boxes, image_size, mask_threshold, **kwargs):
        """Rescale masks and optionally compute polygon segments.

        Args:
            raw_masks: Masks as torch.Tensor of shape (N,H,W) or (N,1,H,W); the latter is squeezed to (N,H,W).
            boxes: Bounding boxes as torch.Tensor, used by rescale_masks.
            image_size: (image_h, image_w) target size for rescale_masks.
            mask_threshold: Binarization threshold for rescale_masks.
            **kwargs: Forwarded predict kwargs; reads ``return_segments``.

        Returns:
            tuple: (batch_masks, batch_segments)
        """
        batch_segments = []
        if len(raw_masks) == 0:
            return raw_masks, batch_segments

        def _to_np(m):
            return m.cpu().numpy() if isinstance(m, torch.Tensor) else m

        if raw_masks.dim() == 4:
            raw_masks = raw_masks.squeeze(1)
        batch_masks = rescale_masks(raw_masks, boxes, image_size, mask_threshold)

        if kwargs.get("return_segments", True):
            batch_segments = [mask_to_polygon_cv2(_to_np(m)) for m in batch_masks]

        return batch_masks, batch_segments

    def _parse_postprocess_kwargs(self, kwargs: dict, batch_size: int) -> tuple:
        """Extract common postprocess keyword arguments.

        Returns:
            tuple: (confs, mask_threshold, operators)
        """
        return (
            self._parse_confidence_config(kwargs.pop("configs", None), list(self.class_map.values())),
            kwargs.pop("mask_threshold", 0.5),
            kwargs.pop("operators", [[] for _ in range(batch_size)]),
        )

    def _build_single_result(
        self,
        batch_boxes,
        batch_scores,
        batch_classes,
        raw_masks,
        ops: list,
        image_size: tuple,
        mask_threshold: float,
        is_seg: bool = False,
        **kwargs,
    ) -> Results:
        """Apply mask postprocessing, assemble a Results object, and revert coordinates.

        Mask postprocessing runs automatically when the model outputs masks (raw_masks is
        non-empty). Pass an empty list or None for raw_masks to skip it.

        Args:
            batch_boxes: Boxes for a single image (tensor or ndarray).
            batch_scores: Scores for a single image (tensor or ndarray).
            batch_classes: Class names for a single image (ndarray of str).
            raw_masks: Raw masks for a single image, or empty list when absent.
            ops: Operator chain for coordinate reversion.
            image_size: (image_h, image_w) for mask rescaling.
            mask_threshold: Binarization threshold passed to rescale_masks.
            is_seg: Whether the model is a segmentation model. Controls whether masks/segments
                keys are always present in to_dict() output, even for empty detections.
            **kwargs: Forwarded to _postprocess_masks (reads ``return_segments``).

        Returns:
            Results object for this image.
        """
        batch_masks, batch_segments = [], []
        if raw_masks is not None and len(raw_masks) > 0:
            batch_masks, batch_segments = self._postprocess_masks(raw_masks, batch_boxes, image_size, mask_threshold, **kwargs)

        segments = (
            [
                torch.tensor(s, dtype=torch.float32, device=self.device)
                if len(s) > 0
                else torch.zeros((0, 2), dtype=torch.float32, device=self.device)
                for s in batch_segments
            ]
            if batch_segments
            else None
        )
        result = Results(
            boxes=batch_boxes,
            scores=batch_scores,
            classes=batch_classes,
            masks=batch_masks if len(batch_masks) > 0 else None,
            segments=segments,
            is_seg=is_seg,
        )
        return self._apply_revert_to_result(result, ops)


@Detectron2Model.register("engine")
class Detectron2TRT(Detectron2Base):
    logger = logging.getLogger("Detectron2TRT")

    def __init__(self, model_path, **kwargs):
        self._setup_device("cuda")
        self.trt = TRTEngine(model_path, device=str(self.device))
        if len(self.trt._input_names) != 1:
            raise ValueError(f"Expected a single-input TRT engine, got inputs: {self.trt._input_names}")
        self.input_dtype = self.trt.input_dtype
        self.image_size = list(self.trt.input_shape[-2:])
        self.batch_size = self.trt.max_batch
        if not self.trt.is_dynamic:
            self.fixed_batch_size = self.trt.max_batch

        class_map = kwargs.get("class_map", None)
        if class_map is None:
            raise ValueError("class_map is required for [Detectron2TRT]")
        self._setup_class_map(class_map)

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
            input = torch.rand(self.batch_size, 3, image_h, image_w, dtype=self.input_dtype, device=self.device)
            self.forward(input)

    def preprocess(self, images: ImageBatch):
        """
        Preprocesses a batch of images for input into the model.

        Args:
            images: A list of HWC images as numpy arrays or torch tensors.

        Returns:
            torch.Tensor: A batch of preprocessed images with shape (batch_size, 3, image_h, image_w) on the model device.
        """
        tensors = []
        for img in images:
            if isinstance(img, torch.Tensor):
                tensors.append(img.permute(2, 0, 1).to(dtype=self.input_dtype, device=self.device))
            else:
                tensors.append(torch.from_numpy(img.transpose(2, 0, 1)).to(dtype=self.input_dtype, device=self.device))
        return torch.stack(tensors)

    def forward(self, inputs):
        """Run TensorRT inference.

        Args:
            inputs (torch.Tensor): BCHW input tensor.

        Returns:
            list: Output tensors from the engine.
        """
        return self.trt.infer(inputs)

    def postprocess(self, predictions, **kwargs) -> List[Results]:
        """Post-process the predictions from the TRT object detection model.

        Args:
            predictions (tuple): Tuple of (num_preds, boxes, scores, classes[, masks]).
            **kwargs:
                images (list): List of input images.
                confs: dict mapping class names to confidence thresholds.
                mask_threshold: float threshold for binarizing masks.
                operators: per-image operator chains.

        Returns:
            List of Results objects, one per image.
        """
        if len(predictions) == 0:
            return []

        images = kwargs.pop("images", [])
        confs, mask_threshold, operators = self._parse_postprocess_kwargs(kwargs, self.batch_size)
        image_h, image_w = images[0].shape[0], images[0].shape[1]

        is_seg = len(predictions) == 5
        if is_seg:
            num_preds, boxes, scores, classes, masks = predictions[:5]
        else:
            num_preds, boxes, scores, classes = predictions[:4]
            masks = None

        # classes is an int tensor — convert to numpy for class_map_func
        classes = self.class_map_func(classes.cpu().numpy() if isinstance(classes, torch.Tensor) else classes)

        scale_factors = torch.tensor([image_w, image_h, image_w, image_h], dtype=torch.float32, device=self.device)
        boxes = boxes.to(dtype=torch.float32) * scale_factors
        scores = scores.to(dtype=torch.float32)
        if masks is not None:
            masks = masks.to(dtype=torch.float32)

        results = []
        for idx in range(self.batch_size):
            n_valid = int(num_preds[idx].item())
            ops = operators[idx]
            batch_boxes, batch_scores, batch_classes, raw_masks, _ = self._apply_confidence_filter(
                scores[idx, :n_valid],
                boxes[idx, :n_valid],
                classes[idx, :n_valid],
                confs,
                masks=masks[idx, :n_valid] if masks is not None else None,
            )
            results.append(
                self._build_single_result(
                    batch_boxes,
                    batch_scores,
                    batch_classes,
                    raw_masks,
                    ops,
                    (image_h, image_w),
                    mask_threshold,
                    is_seg=is_seg,
                    **kwargs,
                )
            )
        return results


@Detectron2Model.register("pt")
class Detectron2PT(Detectron2Base):
    logger = logging.getLogger("Detectron2PT")

    def __init__(self, model_path, **kwargs):
        device = kwargs.get("device", "cuda")
        self._setup_device(device)

        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}") from e

        class_map = kwargs.get("class_map", None)
        if class_map is None:
            raise ValueError("class_map is required for [Detectron2PT]")
        self._setup_class_map(class_map)
        self.batch_size = kwargs.get("batch_size", 1)
        self.image_size = kwargs.get("image_size", [640, 640])

    def warmup(self, **kwargs):
        """
        Perform a warmup operation for the model.

        This method generates a random input tensor with the specified image size and
        passes it through the model to perform a warmup. This can be useful to initialize
        model parameters and optimize performance before actual inference.

        Args:
            **kwargs: Arbitrary keyword arguments.
                img_size (tuple, optional): A tuple specifying the height and width of the image.
                                            If not provided, the default image size is used.
        """
        image_size = kwargs.get("img_size", self.image_size)
        image_h, image_w = image_size[0], image_size[1]
        images = [np.random.rand(image_h, image_w, 3).astype(np.float32) for _ in range(self.batch_size)]
        self.forward(self.preprocess(images))

    def preprocess(self, images: ImageBatch) -> List[Dict[str, torch.Tensor]]:
        """
        Preprocesses a batch of images for input into the model.

        Args:
            images: A list of HWC images as numpy arrays or torch tensors.

        Returns:
            list: A list of dicts with key 'image' mapping to a CHW float32 tensor on self.device.
        """
        inputs = []
        for image in images:
            if isinstance(image, torch.Tensor):
                t = image.permute(2, 0, 1).to(dtype=torch.float32, device=self.device)
            else:
                t = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).to(self.device)
            inputs.append(dict(image=t))
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

    def postprocess(self, predictions, **kwargs) -> List[Results]:
        """Post-process the predictions from the TorchScript object detection model.

        Args:
            predictions (list): Per-image dicts with keys "scores", "pred_classes", "pred_boxes", "pred_masks" (optional).
            **kwargs:
                images (list): A list of input images.
                confs: dict mapping class names to confidence thresholds.
                mask_threshold: float threshold for binarizing masks.
                operators: per-image operator chains.

        Returns:
            List of Results objects, one per image.
        """
        if len(predictions) == 0:
            return []

        images = kwargs.pop("images", [])
        confs, mask_threshold, operators = self._parse_postprocess_kwargs(kwargs, len(predictions))
        is_seg = any("pred_masks" in out for out in predictions)

        results = []
        for idx, output in enumerate(predictions):
            ops = operators[idx]
            image_h, image_w = images[idx].shape[:2]
            batch_scores = output["scores"]
            batch_classes = self.class_map_func(output["pred_classes"].cpu().numpy())
            batch_boxes, batch_scores, batch_classes, raw_masks, _ = self._apply_confidence_filter(
                batch_scores, output["pred_boxes"], batch_classes, confs, masks=output.get("pred_masks")
            )
            results.append(
                self._build_single_result(
                    batch_boxes,
                    batch_scores,
                    batch_classes,
                    raw_masks,
                    ops,
                    (image_h, image_w),
                    mask_threshold,
                    is_seg=is_seg,
                    **kwargs,
                )
            )
        return results
