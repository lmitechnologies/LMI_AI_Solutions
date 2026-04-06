import collections
import logging
import time
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from ultralytics.utils import nms, ops
from ultralytics.utils.torch_utils import smart_inference_mode

# import LMI AI Solutions modules
import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils
from lmi_common.yolo_core import YoloCore
from object_detectors.od_core.object_detector_registry import ObjectDetectorRegistry
from object_detectors.od_core.od_base import ODBase
from object_detectors.od_core.results import Results


@smart_inference_mode()
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
                self.logger.warning("No class matches in confidence dict, set to 1.0 for all classes.")
        else:
            raise TypeError(f"Confidence type {type(conf)} not supported")
        return min_conf

    @smart_inference_mode()
    def _get_thresholds(self, conf: Union[float, dict], num_preds: int, classes: list) -> torch.Tensor:
        """Get the thresholds for each class.

        Args:
            conf (float | dict): float or dictionary of <class: confidence level>.
            num_preds (int): the number of predictions.
            classes (list): the list of class names for each prediction.

        Returns:
            (torch.Tensor): the thresholds for each prediction.
        """
        if isinstance(conf, float):
            thres = np.array([conf] * num_preds)
        elif isinstance(conf, dict):
            # set to 1 if c is not in conf
            thres = np.array([conf.get(c, 1) for c in classes])
        else:
            raise TypeError(f"Confidence type {type(conf)} not supported")
        return self.from_numpy(thres)

    def construct_result(self, pred, img, orig_img, conf):
        """Constructs the result from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (np.ndarray | torch.Tensor): Original image. If this is a tensor, this function will return tensor results.
            conf (float | dict): float or dictionary of <class: confidence level>.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        xyxy, confs, clss = pred[:, :4], pred[:, 4], pred[:, 5]
        classes = np.array([self.model.names[c.item()] for c in clss])

        # filter based on conf
        M = confs > self._get_thresholds(conf, len(clss), classes)
        return Results(xyxy[M], confs[M], classes[M.cpu().numpy()].tolist()), M

    def construct_results(self, preds, img, orig_imgs, conf):
        """Constructs the results from the model predictions.

        Args:
            preds (list): Predictions from the model.
            img (torch.Tensor): the preprocessed image(s)
            orig_imgs (list): list of original images. If this is a list of tensors, this function will return tensor results.
            conf (float | dict): float or dictionary of <class: confidence level>.
        """
        return [self.construct_result(pred, img, orig_img, conf)[0] for pred, orig_img in zip(preds, orig_imgs)]

    def _run_nms(self, preds, conf: float, iou=0.45, agnostic=False, max_det=300):
        """runs non-maximum suppression on inference results"""
        end2end = getattr(self.model, "end2end", False)
        nc = 0 if self.task == "detect" else len(self.model.names)
        return nms.non_max_suppression(preds, conf, iou, agnostic=agnostic, max_det=max_det, nc=nc, end2end=end2end)

    @smart_inference_mode()
    def postprocess(
        self,
        preds,
        img,
        orig_imgs,
        conf: Union[float, dict],
        iou=0.45,
        agnostic=False,
        max_det=300,
        **kwargs,
    ):
        """Postprocesses predictions and returns a list of Results objects.

        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image(s)
            orig_imgs (np.ndarray | torch.Tensor | list): Original image or list of original images.
                If this is a tensor or a list of tensors, this function will return tensor results.
            conf (float | dict): float or dictionary of <class: confidence level>.
            iou (float): The IoU threshold below which boxes will be filtered out during NMS.
            max_det (int): The maximum number of detections to return. defaults to 300.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.
            kwargs (dict): Additional keyword arguments, such as return_segments, proto.
        Returns:
            (dict): the dictionary contains several keys: boxes, scores, classes, masks, and (masks, segments if use a segmentation model).
                    the shape of boxes is (B, N, 4), where B is the batch size and N is the number of detected objects.
                    the shape of classes and scores are both (B, N).
                    the shape of masks: (B, H, W, 3), where H and W are the height and width of the input image.
                    the shape of segments: [ (n1,2), (n2,2), ...]
        """
        min_conf = self._get_min_conf(conf)
        preds2 = self._run_nms(preds, min_conf, iou, agnostic, max_det)
        orig_imgs = orig_imgs if isinstance(orig_imgs, list) else [orig_imgs]
        list_results = self.construct_results(preds2, img, orig_imgs, conf, **kwargs)
        # gather final results — always include one entry per image, even if empty
        results = collections.defaultdict(list)
        for result, orig_img in zip(list_results, orig_imgs):
            use_tensor = isinstance(orig_img, torch.Tensor)
            result_dict = result.to_dict(return_tensor=use_tensor)
            for k in result._all_keys:
                v = result_dict.get(k)
                if v is not None:
                    results[k].append(v)
                else:
                    results[k].append([])
        return results

    def _revert_coordinates(self, results: Dict, operators: List[Dict], **kwargs) -> Dict:
        """Reverts prediction coordinates to the original pre-transform space for a single image."""
        if not operators:
            return results
        results["boxes"] = pipeline_utils.revert_to_origin(results["boxes"], operators, **kwargs)
        return results

    @smart_inference_mode()
    def predict(
        self,
        image: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
        configs,
        operators: Optional[Union[List[Dict], List[List[Dict]]]] = None,
        iou=0.4,
        agnostic=False,
        max_det=300,
        **kwargs,
    ):
        """Run Yolo inference, where it runs the preprocess(), forward(), and postprocess() in sequence.
        It converts the results to the original coordinates space if the operators are provided.
        Return tensors if the input image is a tensor, otherwise return numpy arrays.

        Supports both single image and batch inference. For batch inference, pass a list of images.

        Args:
            image (np.ndarray | tensor | list): a single HWC image or a list of HWC images.
                All images in a batch must have the same dimensions.
            configs (dict | float): a float or a dictionary of the confidence thresholds for each class,
                e.g., {'classA':0.5, 'classB':0.6}
            operators: operators for coordinate reversion. Accepts:
                - None: no coordinate reversion.
                - list[dict]: a single operator chain, applied to all images in the batch.
                - list[list[dict]]: per-image operator chains (length must match batch size).
            iou (float): the iou threshold for non-maximum suppression. defaults to 0.4
            agnostic (bool): If True, the model is agnostic to the number of classes,
                and all classes will be considered as one.
            max_det (int): The maximum number of detections to return. defaults to 300.
            kwargs (dict): Additional keyword arguments, such as return_segments.
        Returns:
            (results, time_info)
            results (dict): a dictionary where each value is a list of length B (batch size), e.g., {
                'boxes': [numpy or tensor, ...],    # each element shape (N_i, 4)
                'classes': [list of strings, ...],
                'scores': [numpy or tensor, ...],
                'masks': [numpy or tensor, ...],    # if applicable
                'segments': [list, ...],            # if applicable
            }
            time_info (dict): a dictionary of the time info, e.g., {'preproc':0.1, 'proc':0.2, 'postproc':0.3}
        """
        time_info = {}

        # Normalize input to list
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        batch_size = len(images)

        # Normalize operators to per-image list
        if operators is None:
            ops_list = [[] for _ in range(batch_size)]
        elif len(operators) > 0 and isinstance(operators[0], dict):
            # list[dict] — same operators for all images
            ops_list = [operators] * batch_size
        else:
            # list[list[dict]] — per-image operators
            if len(operators) != batch_size:
                raise ValueError(f"operators length ({len(operators)}) must match batch size ({batch_size})")
            ops_list = operators

        # preprocess
        t0 = time.time()
        im = self.preprocess(images if is_batch else images[0])
        time_info["preproc"] = time.time() - t0

        # infer
        t0 = time.time()
        pred = self.forward(im)
        time_info["proc"] = time.time() - t0

        # postprocess
        t0 = time.time()
        post_args = {
            "conf": configs,
            "iou": iou,
            "agnostic": agnostic,
            "max_det": max_det,
        }
        if kwargs.get("return_segments"):
            post_args["return_segments"] = kwargs["return_segments"]
        results = self.postprocess(pred, im, images, **post_args)

        # Revert coordinates per image
        final = collections.defaultdict(list)
        for i in range(batch_size):
            single = {k: v[i] for k, v in results.items()}
            single = self._revert_coordinates(single, ops_list[i], **kwargs)
            for k, v in single.items():
                final[k].append(v)

        time_info["postproc"] = time.time() - t0
        return final, time_info

    @staticmethod
    @smart_inference_mode()
    def annotate_image(
        results,
        image,
        colormap=None,
        line_thickness=None,
        hide_label=False,
        hide_bbox=False,
    ):
        """annotate model results on the image. If colormap is None, it will use the random colors.

        Args:
            results (dict): the results of the object detection, e.g., {'boxes':[], 'classes':[], 'scores':[], 'masks':[], 'segments':[]}
            image (np.ndarray | torch.Tensor): the input image
            colors (list, optional): a dictionary of colormaps, e.g., {'class-A':(0,0,255), 'class-B':(0,255,0)}. Defaults to None.
            line_thickness (int, optional): the thickness of the bounding box. Defaults to None.
            hide_bbox (bool,optional): hide the bounding box
        Returns:
            np.ndarray: the annotated image
        """
        boxes = results["boxes"]
        classes = results["classes"]
        scores = results["scores"]
        masks = results["masks"]
        points = results["points"]

        image = to_numpy(image).copy()
        if not len(boxes):
            return image

        # convert to numpy
        boxes = to_numpy(boxes)
        points = to_numpy(points)
        if len(masks):
            masks = to_numpy(masks)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # plot boxes and masks
        for i in range(len(boxes)):
            label = "{}: {:.2f}".format(classes[i], scores[i])
            args = {
                "label": None if hide_label else label,
                "color": None if colormap is None else colormap[classes[i]],
                "line_thickness": line_thickness,
                "hide_bbox": hide_bbox,
            }

            if boxes[i].shape == (4, 2):
                pipeline_utils.plot_one_rbox(boxes[i], image, **args)
            elif boxes[i].shape == (4,):
                mask = masks[i] if len(masks) else None
                pipeline_utils.plot_one_box(boxes[i], image, mask, **args)

        # plot keypoints
        points = points.astype(int)
        for i in range(len(points)):
            for j in range(len(points[i])):
                cv2.circle(image, (points[i][j][0], points[i][j][1]), 4, (255, 255, 255), -1)

        return image


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

    def construct_result(self, pred, img, orig_img, conf, proto, return_segments=True):
        """Constructs a Results object from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (np.ndarray | torch.Tensor): Original image.
            conf (float | dict): Confidence threshold for filtering predictions.
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
        if masks is not None:
            results.masks = masks[M]
            if return_segments:
                segments = self.to_segments(masks[M], orig_img.shape)  # list of [ (n1,2), (n2,2), ... ]
                results.segments = [self.from_numpy(x) for x in segments]
        return results, M

    def construct_results(self, preds, img, orig_imgs, conf, protos, return_segments=True):
        """Constructs the results from the model predictions.

        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image(s)
            orig_imgs (list): A list of original images. If this is a list of tensors, this function will return tensor results.
            conf (float | dict): float or dictionary of <class: confidence level>.
            proto (torch.Tensor): The prototype tensor for the masks.
            return_segments (bool): If True, return the segments of the masks.
        """
        return [
            self.construct_result(pred, img, orig_img, conf, proto=proto, return_segments=return_segments)[0]
            for pred, orig_img, proto in zip(preds, orig_imgs, protos)
        ]

    @smart_inference_mode()
    def postprocess(
        self,
        preds,
        img,
        orig_imgs,
        conf: Union[float, dict],
        iou=0.45,
        agnostic=False,
        max_det=300,
        return_segments=True,
    ):
        """Postprocesses predictions and returns a list of Results objects."""
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return super().postprocess(
            preds[0],
            img,
            orig_imgs,
            conf,
            iou=iou,
            agnostic=agnostic,
            max_det=max_det,
            protos=protos,
            return_segments=return_segments,
        )

    def _revert_coordinates(self, results: Dict, operators: List[Dict], **kwargs) -> Dict:
        """Reverts prediction coordinates to the original pre-transform space."""
        if not operators:
            return results
        super()._revert_coordinates(results, operators, **kwargs)
        if "masks" in results:
            results["masks"] = pipeline_utils.revert_masks_to_origin(results["masks"], operators, **kwargs)
        if "segments" in results:
            results["segments"] = [pipeline_utils.revert_to_origin(seg, operators, **kwargs) for seg in results["segments"]]

        return results


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

    def construct_result(self, pred, img, orig_img, conf):
        """Constructs the result from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (torch.Tensor): the original image
            conf (float | dict): float or dictionary of <class: confidence level>.

        Returns:
            dict: the constructed result dictionary
        """
        rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        confs, clss = pred[:, 4], pred[:, 5]
        classes = np.array([self.model.names[c.item()] for c in clss])

        # covert the boxes from xywhr to xyxyxyxy format
        rboxes = ops.xywhr2xyxyxyxy(rboxes)  # [n_obj, 4, 2]

        # filter based on conf
        M = confs > self._get_thresholds(conf, len(clss), classes)
        return Results(rboxes[M], confs[M], classes[M.cpu().numpy()].tolist()), M

    def _revert_coordinates(self, results: Dict, operators: List[Dict], **kwargs) -> Dict:
        """Reverts OBB coordinates to the original pre-transform space."""
        if not operators:
            return results
        boxes = results["boxes"]
        reverted_boxes = [pipeline_utils.revert_to_origin(box, operators, **kwargs) for box in boxes]
        results["boxes"] = torch.stack(reverted_boxes) if isinstance(boxes, torch.Tensor) else np.array(reverted_boxes)
        return results


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

    def construct_result(self, pred, img, orig_img, conf):
        """Constructs a Results object from the model prediction.

        Args:
            pred (torch.Tensor): Prediction from the model.
            img (torch.Tensor): the preprocessed image
            orig_img (np.ndarray | torch.Tensor): Original image.
            conf (float | dict): Confidence threshold for filtering predictions.
        """
        results, M = super().construct_result(pred, img, orig_img, conf)
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        results.points = pred_kpts[M]  # [n_obj,n_kp,3]
        return results, M

    def _revert_coordinates(self, results: Dict, operators: List[Dict], **kwargs) -> Dict:
        """Reverts pose coordinates to the original pre-transform space."""
        if not operators:
            return results
        super()._revert_coordinates(results, operators, **kwargs)
        if results.get("points") is not None:
            points = results["points"]
            visibility = None
            if len(points) and points.shape[-1] == 3:
                points = points[:, :, :-1]
                visibility = points[:, :, -1]
            reverted_points = [pipeline_utils.revert_to_origin(p, operators, **kwargs) for p in points]  # each iter: [n_kp,2]
            # add back the visibility if exists
            is_tensor = isinstance(points, torch.Tensor)
            if visibility is not None:
                reverted_points = [
                    torch.cat((p, v.unsqueeze(-1)), dim=-1) if is_tensor else np.hstack((p, np.expand_dims(v, -1)))
                    for p, v in zip(reverted_points, visibility)
                ]
            results["points"] = torch.stack(reverted_points) if is_tensor else np.array(reverted_points)

        return results
