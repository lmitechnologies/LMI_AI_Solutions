import abc
import collections
import logging
import random

import cv2
import numpy as np
import torch

import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils

from .results import Results


class ODBase(abc.ABC):
    logger = logging.getLogger(__name__)

    @abc.abstractmethod
    def warmup(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def postprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """
        combine preprocess, forward, and postprocess
        """
        pass

    # ---- shared utilities ----

    @staticmethod
    def to_numpy(data):
        """Converts a tensor or a list to numpy arrays.

        Args:
            data (torch.Tensor | list | np.ndarray): The input data.

        Returns:
            np.ndarray: The converted numpy array.
        """
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Data type {type(data)} not supported")

    def _setup_class_map(self, class_names: dict) -> None:
        """Initialize class_map and a vectorized name-lookup.

        Args:
            class_names: A dict mapping int class index to str class name.
        """
        if class_names is None:
            raise ValueError(f"class_map is required for {self.__class__.__name__}")
        if not isinstance(class_names, dict):
            raise TypeError(f"class_map must be a dict, got {type(class_names).__name__}")
        try:
            self.class_map = {int(k): str(v) for k, v in class_names.items()}
        except (ValueError, TypeError):
            raise
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))

    def _apply_confidence_filter(self, scores, boxes, classes: np.ndarray, confs: dict, masks=None):
        """Filter predictions by per-class confidence thresholds.

        Supports both numpy arrays and torch tensors for scores, boxes, and masks.
        classes must be a numpy array of class name strings.

        Args:
            scores: 1-D array or tensor of confidence scores.
            boxes: Array or tensor of bounding boxes.
            classes: numpy array of class name strings.
            confs: Dict mapping class name to threshold.
            masks: Optional array or tensor of masks. Pass None when absent.

        Returns:
            tuple: (boxes, scores, classes, masks, keep) where keep is the boolean
                   mask matching scores type. masks is [] when not provided.
        """
        thresholds = self._compute_thresholds(classes, confs)
        if isinstance(scores, torch.Tensor):
            keep = scores >= torch.from_numpy(thresholds).to(scores.device)
            keep_np = keep.cpu().numpy()
            filtered_masks = masks[keep] if masks is not None and len(masks) > 0 else []
            return boxes[keep], scores[keep], classes[keep_np], filtered_masks, keep
        keep = scores >= thresholds
        filtered_masks = masks[keep] if masks is not None else []
        return boxes[keep], scores[keep], classes[keep], filtered_masks, keep

    @staticmethod
    def _compute_thresholds(classes: np.ndarray, confs: dict) -> np.ndarray:
        """Compute per-prediction confidence thresholds from class names and a conf dict.

        Args:
            classes: 1-D numpy array of class name strings.
            confs: Dict mapping class name to threshold. Unknown classes default to 1.0.

        Returns:
            float32 numpy array of thresholds, same length as classes.
        """
        if len(classes) == 0:
            return np.empty(0, dtype=np.float32)
        return np.vectorize(confs.get)(classes, 1.0).astype(np.float32)

    @staticmethod
    def _normalize_operators(operators, batch_size: int) -> list:
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
            configs: float/int (global threshold), or dict (per-class).
            class_names: Iterable of class name strings used when building a uniform dict.

        Returns:
            dict mapping class name -> confidence threshold.
        """
        if configs is None:
            raise ValueError("confs cannot be None, must be a float or dict")
        if isinstance(configs, (int, float)):
            return {name: float(configs) for name in class_names}
        if isinstance(configs, dict):
            return configs
        raise ValueError(f"configs must be a float or dict, got {type(configs).__name__}")

    @staticmethod
    def _aggregate_results(list_results: list[Results], return_tensor: bool = False) -> dict:
        """Aggregate a list of Results into a single dict with per-image lists.

        Args:
            list_results: List of Results objects.
            return_tensor: If True, keep tensors; if False, convert to numpy arrays.

        Returns:
            Dict with keys from Results._all_keys (boxes, scores, classes, masks, points, segments).
        """
        final = collections.defaultdict(list)
        for result in list_results:
            result_dict = result.to_dict(return_tensor=return_tensor)
            for k in result._all_keys:
                v = result_dict.get(k)
                if v is not None:
                    final[k].append(v)
                else:
                    final[k].append([])
        return dict(final)

    @staticmethod
    def annotate_image(results, image, colormap=None, line_thickness=None, hide_label=False, hide_bbox=False):
        """Annotate model results on the image.

        Args:
            results (dict): Detection results, e.g. {'boxes':[], 'classes':[], 'scores':[], 'masks':[], 'segments':[], 'points':[]}.
            image (np.ndarray | torch.Tensor): Input image.
            colormap (dict, optional): Maps class name to RGB tuple. Defaults to None (random colors).
            line_thickness (int, optional): Bounding box line thickness. Defaults to None.
            hide_label (bool): If True, suppress class/score labels.
            hide_bbox (bool): If True, suppress bounding boxes.

        Returns:
            np.ndarray: Annotated copy of the image.
        """
        boxes = results["boxes"]
        classes = results["classes"]
        scores = results["scores"]
        masks = results.get("masks", [])
        points = results.get("points", [])

        image = ODBase.to_numpy(image).copy()
        if not len(boxes):
            return image

        boxes = ODBase.to_numpy(boxes)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if len(masks):
            masks = ODBase.to_numpy(masks)
        for i in range(len(boxes)):
            label = "{}: {:.2f}".format(classes[i], scores[i])
            args = {
                "label": None if hide_label else label,
                "color": None if colormap is None else colormap.get(classes[i]),
                "line_thickness": line_thickness,
                "hide_bbox": hide_bbox,
            }
            if boxes[i].shape == (4, 2):
                pipeline_utils.plot_one_rbox(boxes[i], image, **args)
            elif boxes[i].shape == (4,):
                mask = masks[i] if len(masks) else None
                pipeline_utils.plot_one_box(boxes[i], image, mask, **args)

        if len(points):
            points = ODBase.to_numpy(points).astype(int)
            for i in range(len(points)):
                for j in range(len(points[i])):
                    cv2.circle(image, (points[i][j][0], points[i][j][1]), 4, (255, 255, 255), -1)

        segments = results.get("segments", [])
        if len(segments):
            for i, seg in enumerate(segments):
                seg = ODBase.to_numpy(seg).astype(int)
                color = None if colormap is None else colormap.get(classes[i])
                color = color or [random.randint(0, 255) for _ in range(3)]
                pts = seg.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=line_thickness or 2)

        return image
