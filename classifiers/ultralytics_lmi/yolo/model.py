import logging
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.data.augment import classify_transforms
from ultralytics.utils.torch_utils import smart_inference_mode

from classifiers.cls_core.classifier_registry import ClassifierRegistry
from object_detectors.ultralytics_lmi.yolo.model import Yolo


@ClassifierRegistry.register(
    metadata=dict(
        versions=["v1"],
        model_names=["yolo", "yolov8", "yolov11"],
        tasks=["classification"],
        frameworks=["ultralytics"],
    )
)
class YoloCls(Yolo):
    logger = logging.getLogger("yolo-cls")

    def __init__(self, weights: str, device="gpu", data=None, fp16=False, **kwargs) -> None:
        """init the model

        Args:
            weights (str): the path to the weights file.
            device (str, optional): _description_. Defaults to 'gpu'.
            data (str, optional): the path to dataset yaml file. Defaults to None.
            fp16 (bool, optional): use fp16 precision. Defaults to False.
            imgsz (list, optional): input image size [h,w]. Defaults to [224,224].
            crop_fraction(float, optional): crop fraction. Defaults to 1.

        Raises:
            FileNotFoundError: _description_
        """
        super().__init__(weights, device, data, fp16, **kwargs)
        self.task = "classify"
        self.image_size = kwargs.get("image_size", [224, 224])

        updated = (
            self.model.model.transforms.transforms[0].size != max(self.image_size)
            if hasattr(self.model.model, "transforms") and hasattr(self.model.model.transforms.transforms[0], "size")
            else False
        )
        self.transforms = classify_transforms(self.image_size) if updated or not self.model.pt else self.model.model.transforms

    @smart_inference_mode()
    def preprocess(self, img):
        """Prepares input image before inference."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.expand_dims(img, 0)
        if not isinstance(img, torch.Tensor):
            img = torch.stack([self.transforms(Image.fromarray(im)) for im in img], dim=0)
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    @smart_inference_mode()
    def postprocess(self, preds):
        """Postprocesses predictions and returns a list of Results objects.

        Args:
            preds (torch.Tensor | list): Predictions from the model.

        """

        results = defaultdict(list)
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        for pred in preds:
            pred = pred.cpu().numpy()
            idx = pred.argmax()
            results["scores"].append(pred[idx].item())
            results["classes"].append(self.model.names[idx])
        return results

    @smart_inference_mode()
    def predict(self, image):
        """run yolov8 classifier inference. It runs the preprocess(), forward(), and postprocess() in sequence.

        Args:
            image (np.ndarray): the input image

        Returns:
            list of [results, time info]
            results (dict): a dictionary of the results, e.g., {'classes':[], 'scores':[]}
            time_info (dict): a dictionary of the time info, e.g., {'preproc':0.1, 'proc':0.2, 'postproc':0.3}
        """
        time_info = {}

        # preprocess
        t0 = time.time()
        im = self.preprocess(image)
        time_info["preproc"] = time.time() - t0

        # infer
        t0 = time.time()
        pred = self.forward(im)
        time_info["proc"] = time.time() - t0

        # postprocess
        t0 = time.time()
        results = self.postprocess(pred)
        time_info["postproc"] = time.time() - t0

        return results, time_info
