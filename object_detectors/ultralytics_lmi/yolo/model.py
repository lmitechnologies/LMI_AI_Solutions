import logging
from typing import List

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms, ops
from ultralytics.utils.torch_utils import smart_inference_mode

from lmi_common.yolo_core import YoloCore
from lmi_utils.gadget_utils.pipeline_utils import resize_image
from lmi_utils.image_utils.types import ImageLike, to_rgb
from object_detectors.od_core.od_base import ODBase
from object_detectors.od_core.results import Results

LETTERBOX_PAD = 114  # ultralytics' gray letterbox fill value


def letterbox(image: ImageLike, new_shape, pad_value: int = LETTERBOX_PAD) -> ImageLike:
    """Letterbox an HWC image to ``new_shape`` (h, w) using ultralytics' letterbox geometry.

    numpy inputs are delegated to ultralytics' ``LetterBox``.
    torch tensors only the resize interpolation backend (torch vs cv2) differs.
    """
    th, tw = int(new_shape[0]), int(new_shape[1])
    if isinstance(image, np.ndarray):
        return LetterBox(new_shape=(th, tw), auto=False, scaleup=True, center=True, padding_value=pad_value)(image=image)

    h0, w0 = image.shape[:2]
    r = min(th / h0, tw / w0)
    new_w, new_h = round(w0 * r), round(h0 * r)
    if (w0, h0) != (new_w, new_h):
        image = resize_image(image, W=new_w, H=new_h)
    dw, dh = (tw - new_w) / 2, (th - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    chw = torch.nn.functional.pad(image.permute(2, 0, 1), (left, right, top, bottom), value=pad_value)
    return chw.permute(1, 2, 0)


class Yolo(YoloCore, ODBase):
    logger = logging.getLogger("yolo")

    RESIZE_PRESERVE_ASPECT = True  # letterbox (see preprocess())
    RESIZE_PAD_VALUE = LETTERBOX_PAD

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
    def _preprocess_single(self, im: ImageLike) -> torch.Tensor:
        """Prepares a single input image before inference.

        Args:
            im (np.ndarray | tensor): HWC image.

        Returns:
            (torch.Tensor): CHW tensor (no batch dim).
        """
        if isinstance(im, np.ndarray):
            im = self.from_numpy(im)

        im = im.to(self.device)
        im = to_rgb(im)

        img = im.permute((2, 0, 1))  # HWC to CHW, (3, h, w)
        img = img.contiguous()

        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    @smart_inference_mode()
    def preprocess(self, images: List[ImageLike]) -> torch.Tensor:
        """Prepares input image(s) before inference.

        Args:
            images: a single HWC image (np.ndarray | tensor) or a list of HWC images.
                All images in a list must have the same dimensions.

        Returns:
            (torch.Tensor): BCHW tensor.
        """
        if not isinstance(images, list):
            images = [images]
        images = self._fit_to_input_size(images, resize_fn=letterbox)
        return torch.stack([self._preprocess_single(im) for im in images])

    def construct_result(self, pred, img, orig_img, confs: dict, operators=None, do_scale_boxes=True, **kwargs):
        """Constructs the result from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (np.ndarray | torch.Tensor): Original image. If this is a tensor, this function will return tensor results.
            confs (dict): per-class confidence thresholds, pre-parsed by postprocess.
            operators (list): operator chain for coordinate reversion.
            do_scale_boxes (bool): Scale boxes from network input to orig_img space. Set False when the
                caller has already scaled pred[:, :4] (e.g. YoloSeg needs scaled boxes for mask cropping).
        """
        if do_scale_boxes:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        xyxy, scores, clss = pred[:, :4], pred[:, 4], pred[:, 5]
        classes = np.array([self.model.names[c.item()] for c in clss])
        xyxy, scores, classes, _, keep = self._apply_confidence_filter(scores, xyxy, classes, confs)
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
            operators (list[dict]): per-image preprocessing history slice for this image
                (each entry's ``metadata`` is a single-image list). Built by od_base's
                ``_normalize_operators`` from the unified history passed to ``predict()``.
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
                operators (list): per-image-sliced preprocessing history (one slice per
                    image in the batch). Produced by ODBase._normalize_operators.
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

        confs = self._parse_confidence_config(conf, list(self.model.names.values()))
        preds2 = self._run_nms(preds, min(confs.values()), iou, agnostic, max_det)
        orig_imgs = orig_imgs if isinstance(orig_imgs, list) else [orig_imgs]
        return self.construct_results(preds2, img, orig_imgs, confs, operators=operators, **kwargs)


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

    def construct_result(self, pred, img, orig_img, conf, operators=None, proto=None, return_segments=True, **kwargs):
        """Constructs a Results object from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (np.ndarray | torch.Tensor): Original image.
            conf (float | dict): Confidence threshold for filtering predictions.
            operators (list[dict]): preprocessing history slice for this single image
                (each entry's ``metadata`` is a 1-element list). See the unified schema
                in PRERELEASE §9.
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

        # Boxes were already scaled above (needed for process_mask_native); don't re-scale.
        results, M = super().construct_result(pred, img, orig_img, conf, do_scale_boxes=False)
        results.is_seg = True
        if masks is not None:
            results.masks = masks[M]
            if return_segments:
                segments = self.to_segments(masks[M], orig_img.shape)  # list of [ (n1,2), (n2,2), ... ]
                results.segments = [self.from_numpy(x) for x in segments]
        results = self._apply_revert_to_result(results, operators, **kwargs)
        return results, M

    def construct_results(self, preds, img, orig_imgs, conf, operators=None, protos=None, return_segments=True, **kwargs):
        """Constructs the results from the model predictions.

        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image(s)
            orig_imgs (list): A list of original images. If this is a list of tensors, this function will return tensor results.
            conf (float | dict): float or dictionary of <class: confidence level>.
            operators (list[dict]): per-image preprocessing history slice for this image
                (each entry's ``metadata`` is a single-image list). Built by od_base's
                ``_normalize_operators`` from the unified history passed to ``predict()``.
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

    def construct_result(self, pred, img, orig_img, confs: dict, operators=None, **kwargs):
        """Constructs the result from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (torch.Tensor): the original image
            confs (dict): per-class confidence thresholds, pre-parsed by postprocess.
            operators (list[dict]): preprocessing history slice for this single image
                (each entry's ``metadata`` is a 1-element list). See the unified schema
                in PRERELEASE §9.

        Returns:
            dict: the constructed result dictionary
        """
        rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        scores, clss = pred[:, 4], pred[:, 5]
        classes = np.array([self.model.names[c.item()] for c in clss])

        # covert the boxes from xywhr to xyxyxyxy format
        rboxes = ops.xywhr2xyxyxyxy(rboxes)  # [n_obj, 4, 2]

        rboxes, scores, classes, _, keep = self._apply_confidence_filter(scores, rboxes, classes, confs)
        result = Results(rboxes, scores, classes)
        result = self._apply_revert_to_result(result, operators, **kwargs)
        return result, keep


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
            operators (list[dict]): preprocessing history slice for this single image
                (each entry's ``metadata`` is a 1-element list). See the unified schema
                in PRERELEASE §9.
        """
        results, M = super().construct_result(pred, img, orig_img, conf)
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        results.points = pred_kpts[M]  # [n_obj,n_kp,3]
        results = self._apply_revert_to_result(results, operators, **kwargs)
        return results, M
