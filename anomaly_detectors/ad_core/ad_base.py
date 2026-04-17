import logging
from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np
import torch

import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils
from lmi_utils.image_utils.types import ImageBatch, ImageLike, normalize_image_batch, to_rgb


class ADBase(ABC):
    logger = logging.getLogger(__name__)
    _colormap_tensor = None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @property
    def colormap_tensor(self):
        """Lazily initialize and return a [256, 3] turbo colormap tensor on self.device."""
        if self._colormap_tensor is None:
            # 1. Generate a gradient from 0 to 255
            gradient = np.arange(256, dtype=np.uint8).reshape(1, 256)

            # 2. Use OpenCV to generate the Look-Up Table
            lut_bgr = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
            lut_rgb = cv2.cvtColor(lut_bgr, cv2.COLOR_BGR2RGB)

            # 3. Reshape to [256, 3] and convert to Tensor
            self._colormap_tensor = self.from_numpy(lut_rgb).squeeze(0)
        return self._colormap_tensor

    def _setup_device(self, device: str) -> None:
        """Validate and set self.device, falling back to CPU if CUDA is unavailable."""
        device = device.lower()
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"Unsupported device: {device}. Either 'cuda' or 'cpu'.")
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("GPU device unavailable. Use CPU instead.")
            self.device = torch.device("cpu")

    @torch.inference_mode()
    def from_numpy(self, x):
        """Convert numpy array or torch tensor to a tensor on self.device."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        return x

    @abstractmethod
    def warmup(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def preprocess(self, images: List[ImageLike]) -> torch.Tensor:
        """Convert a list of HWC uint8 images to a batched [N,C,H,W] float tensor."""
        pass

    @abstractmethod
    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Run raw model inference on a preprocessed [N,C,H,W] tensor."""
        pass

    @abstractmethod
    def postprocess(self, output: torch.Tensor, return_numpy: bool = True) -> List[ImageLike]:
        """Convert raw model output [N,H,W] to a list of per-image anomaly maps."""
        pass

    @torch.inference_mode()
    def predict(self, image: ImageBatch, **kwargs) -> List[ImageLike]:
        """Run the full inference pipeline: normalize → preprocess → forward → postprocess.

        Args:
            image: A single HW or HWC uint8 image, a list of HW/HWC images, or a BHWC batch.
                2D (HW) images are expanded to 3-channel RGB before preprocessing.
            **kwargs:
                batch_size (int): chunk size for mini-batch inference (default: None = all at once)

        Returns:
            List of per-image anomaly maps [H,W]. dtype mirrors input:
            numpy arrays if input was numpy, tensors if input was tensors.
        """
        images = [to_rgb(img) for img in normalize_image_batch(image)]
        use_tensor = any(isinstance(img, torch.Tensor) for img in images)

        input_batch = self.preprocess(images)

        batch_size = kwargs.get("batch_size", None)
        if batch_size is not None:
            output = self._perform_batched_inference(input_batch, batch_size)
        else:
            output = self.forward(input_batch)

        return self.postprocess(output, return_numpy=not use_tensor)

    def _perform_batched_inference(self, input_batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Run forward() in mini-batches and concatenate the results.

        Args:
            input_batch: Preprocessed [N,C,H,W] tensor.
            batch_size: Maximum number of samples per forward call.

        Returns:
            Concatenated output tensor from all mini-batches.
        """
        if not (0 < batch_size and batch_size <= input_batch.shape[0]):
            raise ValueError(f"batch_size must be positive and less than or equal to the input batch size, got {batch_size}")

        outputs = []
        for start in range(0, input_batch.shape[0], batch_size):
            mini = input_batch[start : start + batch_size]
            out = self.forward(mini)
            if out is None:
                raise RuntimeError(f"forward() returned None for mini-batch [{start}:{start + batch_size}]")
            outputs.append(out)

        if not outputs:
            raise RuntimeError(f"No outputs produced. Input batch size: {input_batch.shape[0]}, batch_size: {batch_size}")

        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def annotate(self, img, ad_scores, ad_threshold, ad_max):
        """Overlay an anomaly heatmap on the input image.

        Args:
            img: HWC image as numpy array or torch tensor.
            ad_scores: Per-pixel anomaly score map [H, W] as numpy array or torch tensor.
            ad_threshold: Score threshold below which pixels are considered normal.
            ad_max: Score value mapped to maximum colormap intensity.

        Returns:
            np.ndarray: Annotated HWC uint8 image with heatmap overlay.
        """
        # ensure that ad_max > ad_threshold
        ad_max = max(ad_max, ad_threshold + 1e-8)
        # convert to tensor
        ad_scores = self.from_numpy(ad_scores)
        img = self.from_numpy(img)
        ad_threshold = self.from_numpy(np.array(ad_threshold)).to(ad_scores.dtype)
        ad_max = self.from_numpy(np.array(ad_max)).to(ad_scores.dtype)
        # Resize AD score to match input image
        h_img, w_img = img.shape[:2]
        ad_scores = pipeline_utils.resize_image(ad_scores, H=h_img, W=w_img)
        # shrink the min-max range
        ad_scores[ad_scores < ad_threshold] = ad_threshold
        ad_scores[ad_scores > ad_max] = ad_max
        # apply colormap
        ad_norm = (ad_scores - ad_threshold) / (ad_max - ad_threshold)
        ad_gray = (ad_norm * 255).to(torch.uint8)
        residual_rgb = self.colormap_tensor[ad_gray.flatten().long()].view(*ad_gray.shape, 3).to(torch.uint8)

        # Overlay anomaly heat map with input image
        annot = img * 0.6 + residual_rgb * 0.4
        annot = annot.round().to(torch.uint8)
        img = img.to(torch.uint8)
        m = ad_gray == 0
        # replace all below-threshold pixels with input image indicating no anomaly
        annot[m] = img[m]
        return annot.cpu().numpy()
