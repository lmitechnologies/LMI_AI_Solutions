import logging
import time
from typing import Dict, List

import numpy as np
import torch
import torchvision  # noqa: F401

from lmi_utils.gadget_utils.pipeline_utils import (
    plot_one_box,
    revert_mask_to_origin,
    revert_to_origin,
)
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

    Provides common utilities: class-map setup, the predict pipeline, and image annotation.
    """

    def _setup_class_map(self, class_map: dict) -> None:
        """Initialize class_map and vectorized lookup.

        Handles both {int_id: str_name} and reversed {str_name: int_id} mappings.
        """
        try:
            self.class_map = {int(k): str(v) for k, v in class_map.items()}
        except Exception:
            self.class_map = {int(v): str(k) for k, v in class_map.items()}
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))

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

    def predict(self, images, operators=None, **kwargs):
        t0 = time.time()
        if isinstance(images, np.ndarray):
            shp = images.shape
            if len(shp) == 3:
                images = [images]
            elif len(shp) == 4 and images.shape[0] != self.batch_size:
                self.logger.error(f"Batch size mismatch: {images.shape[0]} != {self.batch_size}")
                return {}
        batch_size = len(images) if isinstance(images, list) else images.shape[0]
        operators = self._normalize_operators(operators, batch_size)
        list_results = self.postprocess(images, self.forward(self.preprocess(images)), operators=operators, **kwargs)
        t1 = time.time()
        self.logger.debug(f"proc-time {(t1 - t0) * 1000.0:.2f} ms")
        final = {"boxes": [], "scores": [], "classes": [], "masks": [], "segments": []}
        for r in list_results:
            r_dict = r.to_dict(return_tensor=False)
            for k in final:
                final[k].append(r_dict.get(k, []))
        return final

    def _postprocess_masks(self, raw_masks, boxes, image_size, mask_threshold, ops, **kwargs):
        """Rescale masks, apply revert_mask_to_origin, and optionally compute polygon segments.

        Args:
            raw_masks: Masks as torch.Tensor or np.ndarray. 4-D tensors (N,1,H,W) are squeezed to (N,H,W).
            boxes: Bounding boxes as torch.Tensor or np.ndarray, used by rescale_masks.
            image_size: (image_h, image_w) target size for rescale_masks.
            mask_threshold: Binarization threshold for rescale_masks.
            ops: List of operator dicts for revert_to_origin / revert_mask_to_origin.
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

        if len(ops) > 0:
            batch_masks = torch.stack([revert_mask_to_origin(m, ops) for m in batch_masks])

        if kwargs.get("return_segments", False):
            if len(ops) > 0:
                batch_segments = [revert_to_origin(mask_to_polygon_cv2(_to_np(m)), ops) for m in batch_masks]
            else:
                batch_segments = [mask_to_polygon_cv2(_to_np(m)) for m in batch_masks]

        return batch_masks, batch_segments

    def _parse_postprocess_kwargs(self, kwargs: dict, batch_size: int) -> tuple:
        """Extract common postprocess keyword arguments.

        Returns:
            tuple: (confs, mask_threshold, process_masks, operators)
        """
        return (
            kwargs.get("confs", {}),
            kwargs.get("mask_threshold", 0.5),
            kwargs.get("process_masks", True),
            kwargs.get("operators", [[] for _ in range(batch_size)]),
        )

    def annotate_image(self, result, image, color_map=None, **kwargs):
        for i in range(len(result.get("classes", []))):
            plot_one_box(
                result["boxes"][i],
                image,
                label=f"{result['classes'][i]}:{result['scores'][i]:.2f}",
                mask=result["masks"][i] if len(result.get("masks", [])) > 0 else None,
                color=color_map,
            )
        return image


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

    def postprocess(self, images, predictions, **kwargs) -> List[Results]:
        """Post-process the predictions from the TRT object detection model.

        Args:
            images (list): List of input images.
            predictions (tuple): Tuple of (num_preds, boxes, scores, classes[, masks]).
            **kwargs: confs, mask_threshold, process_masks, operators.

        Returns:
            List of Results objects, one per image.
        """
        if len(predictions) == 0:
            return []

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
            valid_scores = scores[idx] >= np.vectorize(confs.get)(classes[idx], 1.0)
            batch_boxes = boxes[idx][valid_scores]
            batch_scores = scores[idx][valid_scores]
            batch_classes = classes[idx][valid_scores]
            filtered_masks = masks[idx][valid_scores] if masks is not None else []
            batch_masks = []
            batch_segments = []
            if process_masks and masks is not None:
                batch_masks, batch_segments = self._postprocess_masks(
                    filtered_masks, batch_boxes, (image_h, image_w), mask_threshold, ops, **kwargs
                )
            else:
                batch_masks = filtered_masks

            batch_boxes = revert_to_origin(batch_boxes, ops)

            results.append(
                Results(
                    boxes=batch_boxes if len(batch_boxes) > 0 else None,
                    scores=batch_scores if len(batch_scores) > 0 else None,
                    classes=batch_classes if len(batch_classes) > 0 else None,
                    masks=batch_masks if len(batch_masks) > 0 else None,
                    segments=[np.array(s, dtype=np.float32) if len(s) > 0 else np.zeros((0, 2), dtype=np.float32) for s in batch_segments]
                    if batch_segments
                    else None,
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
            self.logger.exception(f"鉂?Failed to load model: {e}")

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

    def postprocess(self, images, predictions, **kwargs) -> List[Results]:
        """Post-process the predictions from the TorchScript object detection model.

        Args:
            images (list): A list of input images.
            predictions (list): Per-image dicts with keys "scores", "pred_classes", "pred_boxes", "pred_masks" (optional).
            **kwargs: confs, mask_threshold, process_masks, operators.

        Returns:
            List of Results objects, one per image.
        """
        if len(predictions) == 0:
            return []

        confs, mask_threshold, process_masks, operators = self._parse_postprocess_kwargs(kwargs, len(predictions))
        if predictions[0]["pred_classes"].shape[0] == 0:
            return [Results() for _ in predictions]

        results = []
        for idx, output in enumerate(predictions):
            ops = operators[idx]
            image_h, image_w = images[idx].shape[:2]
            batch_scores = output["scores"]
            batch_classes = self.class_map_func(output["pred_classes"].cpu().numpy())
            threshold = torch.from_numpy(np.vectorize(confs.get)(batch_classes, 1.0).astype(np.float32)).to(batch_scores.device)
            keep = batch_scores >= threshold
            batch_scores = batch_scores[keep]
            batch_classes = batch_classes[keep.cpu().numpy()]
            batch_boxes = output["pred_boxes"][keep]
            batch_masks = []
            batch_segments = []
            if "pred_masks" in output:
                batch_masks = output["pred_masks"][keep]
                if process_masks and len(batch_masks) > 0:
                    batch_masks, batch_segments = self._postprocess_masks(
                        batch_masks, batch_boxes, (image_h, image_w), mask_threshold, ops, **kwargs
                    )

            batch_boxes = revert_to_origin(batch_boxes, ops)
            masks_t = batch_masks if len(batch_masks) > 0 else None

            results.append(
                Results(
                    boxes=batch_boxes if len(batch_boxes) > 0 else None,
                    scores=batch_scores if len(batch_scores) > 0 else None,
                    classes=batch_classes if len(batch_classes) > 0 else None,
                    masks=masks_t,
                    segments=[np.array(s, dtype=np.float32) if len(s) > 0 else np.zeros((0, 2), dtype=np.float32) for s in batch_segments]
                    if batch_segments
                    else None,
                )
            )
        return results
