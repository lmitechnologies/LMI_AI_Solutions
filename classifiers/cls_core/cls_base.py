import abc
import time
from typing import List

import numpy as np
import torch

from lmi_utils.image_utils.types import ImageBatch, normalize_image_batch, to_rgb


class ClassifierBase(abc.ABC):
    # Set to a positive integer in subclasses that use a fixed-batch-size model.
    fixed_batch_size: int = None

    @abc.abstractmethod
    def warmup(self, *args, **kwargs):
        pass

    def release(self) -> None:  # noqa: B027 — intentional no-op default, not abstract
        """Free resources that need deterministic teardown (e.g. TRT engine/context). Default no-op."""

    @abc.abstractmethod
    def preprocess(self, images: List, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def postprocess(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def predict(self, image: ImageBatch, **kwargs):
        """Run the full inference pipeline: preprocess → forward → postprocess.

        Supports both single image and batch inference. When ``self.fixed_batch_size``
        is set (e.g. TRT engines with a hard-coded batch dimension), the input is
        processed in chunks of that size and the last chunk is zero-padded.

        Args:
            image: A single HW or HWC image, a list of HW/HWC images, or a BHWC batch.
                Accepts both numpy arrays and torch tensors.
        kwargs:
            batch_size (int): chunk size for dynamic mini-batch inference (default: None = all at once).
                Ignored when self.fixed_batch_size is set.

        Returns:
            (results, time_info)
            results (dict): a dictionary where each value is a list of length B (batch size), e.g., {
                'classes': [str, ...],
                'scores': [float, ...],
            }
            time_info (dict): timing info with keys 'preproc', 'proc', 'postproc'.
        """
        images = [to_rgb(img) for img in normalize_image_batch(image)]

        fixed_bs = self.fixed_batch_size
        batch_size = kwargs.pop("batch_size", None)
        effective_bs = fixed_bs or batch_size

        if effective_bs:
            return self._run_batched_predict(images, effective_bs, pad_last=fixed_bs is not None, **kwargs)

        t0 = time.time()
        preprocessed = self.preprocess(images)
        time_info = {"preproc": time.time() - t0}

        t0 = time.time()
        outputs = self.forward(preprocessed)
        time_info["proc"] = time.time() - t0

        t0 = time.time()
        results = self.postprocess(outputs)
        time_info["postproc"] = time.time() - t0

        return results, time_info

    def _run_batched_predict(self, images, batch_size, pad_last=False, **kwargs):
        """Run preprocess → forward → postprocess in chunks and collect per-image results.

        Args:
            images: Flat list of HWC images (numpy or tensor).
            batch_size: Number of images per chunk.
            pad_last: If True, zero-pad the last chunk to exactly batch_size (for fixed-batch TRT engines).

        Returns:
            (results dict, time_info dict with keys 'preproc', 'proc', 'postproc').
        """
        all_results = {}
        t_preproc, t_proc, t_postproc = 0.0, 0.0, 0.0

        for start in range(0, len(images), batch_size):
            chunk_imgs = images[start : start + batch_size]
            chunk_n = len(chunk_imgs)

            if pad_last and chunk_n < batch_size:
                ref = chunk_imgs[0]
                pad = torch.zeros_like(ref) if isinstance(ref, torch.Tensor) else np.zeros_like(ref)
                chunk_imgs = chunk_imgs + [pad] * (batch_size - chunk_n)

            t0 = time.time()
            preprocessed = self.preprocess(chunk_imgs)
            t_preproc += time.time() - t0

            t0 = time.time()
            outputs = self.forward(preprocessed)
            t_proc += time.time() - t0

            t0 = time.time()
            chunk_results = self.postprocess(outputs)
            t_postproc += time.time() - t0

            for k, v in chunk_results.items():
                all_results.setdefault(k, []).extend(v[:chunk_n])

        return all_results, {"preproc": t_preproc, "proc": t_proc, "postproc": t_postproc}
