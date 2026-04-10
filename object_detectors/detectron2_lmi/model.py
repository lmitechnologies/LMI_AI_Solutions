import logging
from typing import Dict, List

import numpy as np
import torch
import torchvision  # noqa: F401

from lmi_utils.postprocess_utils.mask_utils import mask_to_polygon_cv2, rescale_masks
from object_detectors.od_core.model_factory import ModelFactory
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
            raw_masks: Masks as torch.Tensor or np.ndarray. 4-D tensors (N,1,H,W) are squeezed to (N,H,W).
            boxes: Bounding boxes as torch.Tensor or np.ndarray, used by rescale_masks.
            image_size: (image_h, image_w) target size for rescale_masks.
            mask_threshold: Binarization threshold for rescale_masks.
            **kwargs: Forwarded predict kwargs; reads ``return_segments``.

        Returns:
            tuple: (batch_masks, batch_segments)
        """
        batch_segments = []
        if len(raw_masks) == 0:
            return raw_masks, batch_segments

        def _to_tensor(m):
            if isinstance(m, np.ndarray):
                return torch.from_numpy(m).to(self.device)
            return m.to(self.device)

        def _to_np(m):
            return m.cpu().numpy() if isinstance(m, torch.Tensor) else m

        masks_t = _to_tensor(raw_masks)
        if masks_t.dim() == 4:
            masks_t = masks_t.squeeze(1)
        batch_masks = rescale_masks(masks_t, _to_tensor(boxes), image_size, mask_threshold)

        if kwargs.get("return_segments", False):
            batch_segments = [mask_to_polygon_cv2(_to_np(m)) for m in batch_masks]

        return batch_masks, batch_segments

    def _parse_postprocess_kwargs(self, kwargs: dict, batch_size: int) -> tuple:
        """Extract common postprocess keyword arguments.

        Returns:
            tuple: (confs, mask_threshold, process_masks, operators)
        """
        return (
            self._parse_confidence_config(kwargs.pop("configs", None), list(self.class_map.values())),
            kwargs.pop("mask_threshold", 0.5),
            kwargs.pop("process_masks", True),
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
        process_masks: bool,
        mask_threshold: float,
        orig_img,
        **kwargs,
    ) -> Results:
        """Apply mask postprocessing, assemble a Results object, and revert coordinates.

        Args:
            batch_boxes: Boxes for a single image (tensor or ndarray).
            batch_scores: Scores for a single image (tensor or ndarray).
            batch_classes: Class names for a single image (ndarray of str).
            raw_masks: Raw masks for a single image, or empty list when absent.
            ops: Operator chain for coordinate reversion.
            image_size: (image_h, image_w) for mask rescaling.
            process_masks: Whether to run mask postprocessing.
            mask_threshold: Binarization threshold passed to rescale_masks.
            orig_img: Original image used to determine tensor vs numpy output.
            **kwargs: Forwarded to _postprocess_masks (reads ``return_segments``).

        Returns:
            Results object for this image.
        """
        batch_masks, batch_segments = [], []
        if process_masks and len(raw_masks) > 0:
            batch_masks, batch_segments = self._postprocess_masks(raw_masks, batch_boxes, image_size, mask_threshold, **kwargs)
        else:
            batch_masks = raw_masks

        result = Results(
            boxes=batch_boxes if len(batch_boxes) > 0 else None,
            scores=batch_scores if len(batch_scores) > 0 else None,
            classes=batch_classes if len(batch_classes) > 0 else None,
            masks=batch_masks if len(batch_masks) > 0 else None,
            segments=[np.array(s, dtype=np.float32) if len(s) > 0 else np.zeros((0, 2), dtype=np.float32) for s in batch_segments]
            if batch_segments
            else None,
        )
        return self._apply_revert_to_result(result, orig_img, ops)


@Detectron2Model.register("engine")
class Detectron2TRT(Detectron2Base):
    logger = logging.getLogger("Detectron2TRT")

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

        import object_detectors.detectron2_lmi.utils.common_runtime as common

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
                self.fixed_batch_size = shape[0]
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
        return np.array([img.transpose(2, 0, 1) for img in images], dtype=self.input_dtype)

    def forward(self, inputs):
        """
        Perform a forward pass through the model.

        Args:
            inputs (numpy.ndarray): The input data to be processed by the model.

        Returns:
            list: A list of numpy arrays containing the model's output data.
        """
        import object_detectors.detectron2_lmi.utils.common_runtime as common

        outputs = []
        for out in self.model_outputs:
            outputs.append(np.zeros(out["shape"], dtype=out["dtype"]))
        common.memcpy_host_to_device(self.model_inputs[0]["allocation"], np.ascontiguousarray(inputs))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.model_outputs[o]["allocation"])
        return outputs

    def postprocess(self, predictions, **kwargs) -> List[Results]:
        """Post-process the predictions from the TRT object detection model.

        Args:
            predictions (tuple): Tuple of (num_preds, boxes, scores, classes[, masks]).
            **kwargs:
                images (list): List of input images.
                confs: dict mapping class names to confidence thresholds.
                mask_threshold: float threshold for binarizing masks.
                process_masks: bool indicating whether to perform mask postprocessing.
                operators: per-image operator chains.

        Returns:
            List of Results objects, one per image.
        """
        if len(predictions) == 0:
            return []

        images = kwargs.pop("images", [])
        confs, mask_threshold, process_masks, operators = self._parse_postprocess_kwargs(kwargs, self.batch_size)
        image_h, image_w = images[0].shape[0], images[0].shape[1]

        if len(predictions) == 5:
            num_preds, boxes, scores, classes, masks = predictions[:5]
        else:
            num_preds, boxes, scores, classes = predictions[:4]
            masks = None

        classes = self.class_map_func(classes)

        if len(boxes) > 0:
            scale_factors = np.array([image_w, image_h, image_w, image_h])
            boxes = (boxes * scale_factors).astype(np.int32)

        results = []
        for idx in range(self.batch_size):
            ops = operators[idx]
            batch_boxes, batch_scores, batch_classes, raw_masks, _ = self._apply_confidence_filter(
                scores[idx], boxes[idx], classes[idx], confs, masks=masks[idx] if masks is not None else None
            )
            results.append(
                self._build_single_result(
                    batch_boxes,
                    batch_scores,
                    batch_classes,
                    raw_masks,
                    ops,
                    (image_h, image_w),
                    process_masks and masks is not None,
                    mask_threshold,
                    images[idx],
                    **kwargs,
                )
            )
        return results


@Detectron2Model.register("pt")
class Detectron2PT(Detectron2Base):
    logger = logging.getLogger("Detectron2PT")

    def __init__(self, model_path, **kwargs):
        device = kwargs.get("device", "cuda")
        if not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

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
                                            If not provided, an error is logged.

        Raises:
            ValueError: If 'img_size' is not provided in kwargs.
        """
        image_size = kwargs.get("img_size", self.image_size)
        image_h, image_w = image_size[0], image_size[1]
        images = [np.random.rand(image_h, image_w, 3).astype(np.float32) for _ in range(self.batch_size)]
        self.forward(self.preprocess(images))

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
        inputs = [
            dict(image=torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).to(dtype=torch.float32).to(self.device))
            for image in images
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

    def postprocess(self, predictions, **kwargs) -> List[Results]:
        """Post-process the predictions from the TorchScript object detection model.

        Args:
            predictions (list): Per-image dicts with keys "scores", "pred_classes", "pred_boxes", "pred_masks" (optional).
            **kwargs:
                images (list): A list of input images.
                confs: dict mapping class names to confidence thresholds.
                mask_threshold: float threshold for binarizing masks.
                process_masks: bool indicating whether to perform mask postprocessing.
                operators: per-image operator chains.

        Returns:
            List of Results objects, one per image.
        """
        if len(predictions) == 0:
            return []

        images = kwargs.pop("images", [])
        confs, mask_threshold, process_masks, operators = self._parse_postprocess_kwargs(kwargs, len(predictions))
        if predictions[0]["pred_classes"].shape[0] == 0:
            return [Results() for _ in predictions]

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
                    process_masks,
                    mask_threshold,
                    images[idx],
                    **kwargs,
                )
            )
        return results
