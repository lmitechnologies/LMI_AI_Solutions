import logging
from typing import List, Union

import numpy as np
import torch
from ultralytics.utils import nms, ops
from ultralytics.utils.torch_utils import smart_inference_mode

from lmi_common.yolo_core import YoloCore
from object_detectors.od_core.object_detector_registry import ObjectDetectorRegistry
from object_detectors.od_core.od_base import ODBase
from object_detectors.od_core.results import Results


@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=["v1"],
        model_names=["yolo", "yolov8", "yolov11"],
        tasks=["od", "objectdetection"],
        frameworks=["ultralytics", "ultralytics8"],
    )
)
class Yolo(YoloCore, ODBase):
    logger = logging.getLogger("yolo")

    def __init__(self, model_path: str, device="cuda", data=None, fp16=False, **kwargs) -> None:
        """init the model
        Args:
            model_path (str): the path to the model_path file.
            device (str, optional): the device to be used, either 'cuda' or 'cpu'. Defaults to 'cuda'.
            data (str, optional): the path to dataset yaml file. Defaults to None.
            fp16 (bool, optional): Whether to use fp16. Defaults to False.
        Raises:
            FileNotFoundError: _description_
        """
        YoloCore.__init__(self, model_path, device, data, fp16, **kwargs)
        self.task = "detect"

    @smart_inference_mode()
    def _preprocess_single(self, im: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Prepares a single input image before inference.

        Args:
            im (np.ndarray | tensor): HWC image.

        Returns:
            (torch.Tensor): CHW tensor (no batch dim).
        """
        if isinstance(im, np.ndarray):
            im = self.from_numpy(im)

        im = im.to(self.device)
        # convert to HWC
        if im.ndim == 2:
            im = im.unsqueeze(-1)
        if im.shape[-1] == 1:
            im = im.expand(-1, -1, 3)

        img = im.permute((2, 0, 1))  # HWC to CHW, (3, h, w)
        img = img.contiguous()

        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    @smart_inference_mode()
    def preprocess(self, images: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]) -> torch.Tensor:
        """Prepares input image(s) before inference.

        Args:
            images: a single HWC image (np.ndarray | tensor) or a list of HWC images.
                All images in a list must have the same dimensions.

        Returns:
            (torch.Tensor): BCHW tensor.
        """
        if isinstance(images, list):
            imgs = [self._preprocess_single(im) for im in images]
            return torch.stack(imgs)
        return self._preprocess_single(images).unsqueeze(0)

    def _get_min_conf(self, conf: Union[float, dict]) -> float:
        """Get the minimum confidence level for non-maximum suppression.

        Args:
            conf (float | dict): float or dictionary of <class: confidence level>.
        """
        if isinstance(conf, float):
            min_conf = conf
        elif isinstance(conf, dict):
            min_conf = 1
            class_names = set(self.model.names.values())
            for k, v in conf.items():
                if k in class_names:
                    min_conf = min(min_conf, v)
            if min_conf == 1:
                raise ValueError("No class matches in confidence dict.")
        else:
            raise TypeError(f"Confidence type {type(conf)} not supported")
        return min_conf

    def construct_result(self, pred, img, orig_img, conf, operators=None, **kwargs):
        """Constructs the result from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (np.ndarray | torch.Tensor): Original image. If this is a tensor, this function will return tensor results.
            conf (float | dict): float or dictionary of <class: confidence level>.
            operators (list): operator chain for coordinate reversion.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        xyxy, scores, clss = pred[:, :4], pred[:, 4], pred[:, 5]
        classes = np.array([self.model.names[c.item()] for c in clss])
        confs_dict = self._parse_confidence_config(conf, list(self.model.names.values()))
        xyxy, scores, classes, _, keep = self._apply_confidence_filter(scores, xyxy, classes, confs_dict)
        result = Results(xyxy, scores, classes)
        result = self._apply_revert_to_result(result, operators, **kwargs)
        return result, keep

    def construct_results(self, preds, img, orig_imgs, conf, operators=None, **kwargs):
        """Constructs the results from the model predictions.

        Args:
            preds (list): Predictions from the model.
            img (torch.Tensor): the preprocessed image(s)
            orig_imgs (list): list of original images. If this is a list of tensors, this function will return tensor results.
            conf (float | dict): float or dictionary of <class: confidence level>.
            operators (list[list]): per-image operator chains.
        """
        ops_list = operators or [[] for _ in range(len(orig_imgs))]
        return [
            self.construct_result(pred, img, orig_img, conf, operators=op, **kwargs)[0]
            for pred, orig_img, op in zip(preds, orig_imgs, ops_list)
        ]

    def _run_nms(self, preds, conf: float, iou=0.45, agnostic=False, max_det=300):
        """runs non-maximum suppression on inference results"""
        end2end = getattr(self.model, "end2end", False)
        nc = 0 if self.task == "detect" else len(self.model.names)
        return nms.non_max_suppression(preds, conf, iou, agnostic=agnostic, max_det=max_det, nc=nc, end2end=end2end)

    @smart_inference_mode()
    def postprocess(self, preds, **kwargs):
        """Postprocesses predictions and returns a list of Results objects.

        Args:
            preds (torch.Tensor | list): Predictions from the model.
            **kwargs:
                preprocessed (torch.Tensor): the preprocessed image(s) (BCHW tensor).
                images (list): list of original images.
                configs (float | dict): confidence threshold(s).
                operators (list[list]): per-image operator chains for coordinate reversion.
                iou (float): IoU threshold for NMS. Default 0.45.
                agnostic (bool): class-agnostic NMS. Default False.
                max_det (int): max detections. Default 300.
                return_segments (bool): whether to return segments.

        Returns:
            list[Results]: one Results object per image.
        """
        img = kwargs.pop("preprocessed", None)
        orig_imgs = kwargs.pop("images", [])
        conf = kwargs.pop("configs", None)
        operators = kwargs.pop("operators", [[] for _ in range(len(orig_imgs))])
        iou = kwargs.pop("iou", 0.45)
        agnostic = kwargs.pop("agnostic", False)
        max_det = kwargs.pop("max_det", 300)

        min_conf = self._get_min_conf(conf)
        preds2 = self._run_nms(preds, min_conf, iou, agnostic, max_det)
        orig_imgs = orig_imgs if isinstance(orig_imgs, list) else [orig_imgs]
        return self.construct_results(preds2, img, orig_imgs, conf, operators=operators, **kwargs)


@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=["v1"],
        model_names=["yolo", "yolov8", "yolov11"],
        tasks=["seg", "instancesegmentation"],
        frameworks=["ultralytics", "ultralytics8"],
    )
)
class YoloSeg(Yolo):
    logger = logging.getLogger("yolo-seg")

    def __init__(self, model_path: str, device="cuda", data=None, fp16=False, **kwargs) -> None:
        super().__init__(model_path, device, data, fp16, **kwargs)
        self.task = "segment"

    def to_segments(self, masks: torch.Tensor, img_shape: tuple) -> List[np.ndarray]:
        """Convert masks to segments.

        Args:
            masks (torch.Tensor): the masks to be converted, shape (n, h, w)
            img_shape (tuple): the shape of the image, (h, w, c)

        Returns:
            (list): a list of segments, each segment is a numpy array of shape (n, 2)
        """
        segments = [ops.scale_coords(masks.shape[1:], x, img_shape, normalize=False) for x in ops.masks2segments(masks)]
        return segments

    def construct_result(self, pred, img, orig_img, conf, operators=None, proto=None, return_segments=False, **kwargs):
        """Constructs a Results object from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (np.ndarray | torch.Tensor): Original image.
            conf (float | dict): Confidence threshold for filtering predictions.
            operators (list): operator chain for coordinate reversion.
            proto (torch.Tensor): The prototype tensor for the masks.
            return_segments (bool): If True, return the segments of the masks.
        """
        if pred.shape[0] == 0:
            masks = None
        else:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with non-empty masks
            if not all(keep):
                pred, masks = pred[keep], masks[keep]

        results, M = super().construct_result(pred, img, orig_img, conf)
        results.is_seg = True
        if masks is not None:
            results.masks = masks[M]
            if return_segments:
                segments = self.to_segments(masks[M], orig_img.shape)  # list of [ (n1,2), (n2,2), ... ]
                results.segments = [self.from_numpy(x) for x in segments]
        results = self._apply_revert_to_result(results, operators, **kwargs)
        return results, M

    def construct_results(self, preds, img, orig_imgs, conf, operators=None, protos=None, return_segments=False, **kwargs):
        """Constructs the results from the model predictions.

        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image(s)
            orig_imgs (list): A list of original images. If this is a list of tensors, this function will return tensor results.
            conf (float | dict): float or dictionary of <class: confidence level>.
            operators (list[list]): per-image operator chains.
            protos (torch.Tensor): The prototype tensors for the masks.
            return_segments (bool): If True, return the segments of the masks.
        """
        ops_list = operators or [[] for _ in range(len(orig_imgs))]
        return [
            self.construct_result(pred, img, orig_img, conf, operators=op, proto=proto, return_segments=return_segments, **kwargs)[0]
            for pred, orig_img, proto, op in zip(preds, orig_imgs, protos, ops_list)
        ]

    @smart_inference_mode()
    def postprocess(self, preds, **kwargs):
        """Postprocesses predictions and returns a list of Results objects."""
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        kwargs["protos"] = protos
        return super().postprocess(preds[0], **kwargs)


@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=["v1"],
        model_names=["yolo", "yolov8", "yolov11"],
        tasks=["obb", "orientedobjectdetection"],
        frameworks=["ultralytics", "ultralytics8"],
    )
)
class YoloObb(Yolo):
    logger = logging.getLogger("yolo-obb")

    def __init__(self, model_path: str, device="cuda", data=None, fp16=False, **kwargs) -> None:
        super().__init__(model_path, device, data, fp16, **kwargs)
        self.task = "obb"

    def _run_nms(self, preds, conf: float, iou=0.45, agnostic=False, max_det=300):
        """Postprocesses predictions and returns a list of Results objects."""
        return nms.non_max_suppression(
            preds,
            conf,
            iou,
            agnostic=agnostic,
            max_det=max_det,
            nc=len(self.model.names),
            rotated=True,
            end2end=getattr(self.model, "end2end", False),
        )

    def construct_result(self, pred, img, orig_img, conf, operators=None, **kwargs):
        """Constructs the result from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (torch.Tensor): the original image
            conf (float | dict): float or dictionary of <class: confidence level>.
            operators (list): operator chain for coordinate reversion.

        Returns:
            dict: the constructed result dictionary
        """
        rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        scores, clss = pred[:, 4], pred[:, 5]
        classes = np.array([self.model.names[c.item()] for c in clss])

        # covert the boxes from xywhr to xyxyxyxy format
        rboxes = ops.xywhr2xyxyxyxy(rboxes)  # [n_obj, 4, 2]

        confs_dict = self._parse_confidence_config(conf, list(self.model.names.values()))
        rboxes, scores, classes, _, keep = self._apply_confidence_filter(scores, rboxes, classes, confs_dict)
        result = Results(rboxes, scores, classes)
        result = self._apply_revert_to_result(result, operators, **kwargs)
        return result, keep


@ObjectDetectorRegistry.register(
    metadata=dict(
        versions=["v1"],
        model_names=["yolo", "yolov8", "yolov11"],
        tasks=["pose", "keypointdetection"],
        frameworks=["ultralytics", "ultralytics8"],
    )
)
class YoloPose(Yolo):
    logger = logging.getLogger("yolo-pose")

    def __init__(self, model_path: str, device="cuda", data=None, fp16=False, **kwargs) -> None:
        super().__init__(model_path, device, data, fp16, **kwargs)
        self.task = "pose"

    def construct_result(self, pred, img, orig_img, conf, operators=None, **kwargs):
        """Constructs a Results object from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (np.ndarray | torch.Tensor): Original image.
            conf (float | dict): Confidence threshold for filtering predictions.
            operators (list): operator chain for coordinate reversion.
        """
        results, M = super().construct_result(pred, img, orig_img, conf)
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        results.points = pred_kpts[M]  # [n_obj,n_kp,3]
        results = self._apply_revert_to_result(results, operators, **kwargs)
        return results, M
