import logging
from collections import defaultdict
from typing import List

import cv2
import torch
from PIL import Image
from ultralytics.data.augment import classify_transforms
from ultralytics.utils.torch_utils import smart_inference_mode

from classifiers.cls_core.cls_base import ClassifierBase
from lmi_common.yolo_core import YoloCore


class YoloCls(YoloCore, ClassifierBase):
    logger = logging.getLogger("yolo-cls")

    def __init__(self, model_path: str, device="cuda", data=None, fp16=False, **kwargs) -> None:
        """init the model

        Args:
            model_path (str): the path to the model_path file.
            device (str, optional): 'cuda', or 'cpu'. Defaults to 'cuda'.
            data (str, optional): the path to dataset yaml file. Defaults to None.
            fp16 (bool, optional): use fp16 precision. Defaults to False.

        """
        YoloCore.__init__(self, model_path, device, data, fp16, **kwargs)
        self.task = "classify"

        updated = (
            self.model.model.transforms.transforms[0].size != max(self.image_size)
            if hasattr(self.model.model, "transforms") and hasattr(self.model.model.transforms.transforms[0], "size")
            else False
        )
        self.transforms = classify_transforms(self.image_size) if updated or self.model.format != "pt" else self.model.model.transforms

    @smart_inference_mode()
    def preprocess(self, images: List) -> torch.Tensor:
        """Prepares a batch of input images before inference.

        Args:
            images (list): list of images, each with shape (H, W, C) or (H, W),
                as numpy arrays or torch tensors.

        Returns:
            torch.Tensor: preprocessed batch with shape (N, C, H, W)
        """
        processed = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            processed.append(self.transforms(Image.fromarray(img)))
        batch = torch.stack(processed, dim=0).to(self.model.device)
        return batch.half() if self.model.fp16 else batch.float()

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
