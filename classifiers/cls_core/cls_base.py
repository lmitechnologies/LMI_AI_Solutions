import abc
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from lmi_utils.image_utils.types import ImageBatch, normalize_image_batch, to_3channel


def _elapsed_since(t0: float) -> float:
    """Seconds since ``t0``, after any queued CUDA work finishes.

    Without the wait an async forward() bills its GPU time to whatever blocks first in the next phase.
    """
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return time.time() - t0


class ClassifierBase(abc.ABC):
    # Set to a positive integer in subclasses that use a fixed-batch-size model.
    fixed_batch_size: int = None

    # Largest batch a dynamic-batch model accepts; predict() splits larger inputs into chunks of it.
    max_batch_size: int = None

    def _setup_class_map(self, class_names: dict) -> None:
        """Populate self.class_map {int softmax index → str class name}. Call from __init__ of each subclass."""
        if class_names is None:
            raise ValueError(f"class_map is required for {self.__class__.__name__}")
        if not isinstance(class_names, dict):
            raise TypeError(f"class_map must be a dict, got {type(class_names).__name__}")
        self.class_map: Dict[int, str] = {int(k): str(v) for k, v in class_names.items()}

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
    def predict(self, image: ImageBatch, operators: Optional[List] = None, confidence=0, **kwargs):
        """Run the full inference pipeline: preprocess → forward → postprocess.

        Supports both single image and batch inference. When ``self.fixed_batch_size``
        is set (e.g. TRT engines with a hard-coded batch dimension), the input is
        processed in chunks of that size and the last chunk is zero-padded. When
        ``self.max_batch_size`` is set (dynamic-batch engines), chunks are at most that size
        and are not padded.

        Args:
            image: A single HW or HWC image, a list of HW/HWC images, or a BHWC batch.
                Accepts both numpy arrays and torch tensors.
            operators: Preprocessing history returned by ``Preprocessor.preprocess()``.
                When a ``TileMeta`` entry is present, per-tile results are aggregated back
                to one prediction per source image via weighted-vote over softmax scores.
            confidence: Confidence threshold (float) or per-class dict. Default 0 = no filtering.
                An image whose top score is below its class threshold is reported as no
                detection (classes[i] = scores[i] = None).
        kwargs:
            batch_size (int): chunk size for dynamic mini-batch inference, capped at self.max_batch_size
                (default: self.max_batch_size, or all at once when that is unset). Ignored when self.fixed_batch_size is set.

        Returns:
            (results, time_info)
            results (dict): a dictionary where each value is a list of length B (batch size), or one entry
                per source image when ``operators`` contains a tile entry, e.g., {
                'classes': [str, ...],
                'scores': [float, ...],
            }
            time_info (dict): timing info with keys 'preproc', 'proc', 'postproc'.
        """
        images = [to_3channel(img) for img in normalize_image_batch(image)]

        tile_meta = self._find_tile_meta(operators)
        post_kwargs = {}
        if tile_meta is not None:
            if not hasattr(self, "class_map"):
                raise RuntimeError(f"{self.__class__.__name__} must call _setup_class_map() before tiling")
            n_expected = sum(h * w for h, w in tile_meta.n_tiles)
            if n_expected != len(images):
                raise ValueError(f"tile history describes {n_expected} tiles, but {len(images)} images were passed to predict()")
            # only sent when tiling so backends without tiling support keep their plain postprocess signature
            post_kwargs["return_all_scores"] = True

        fixed_bs = self.fixed_batch_size
        batch_size = kwargs.pop("batch_size", None)
        if self.max_batch_size:
            batch_size = min(batch_size or self.max_batch_size, self.max_batch_size)
        effective_bs = fixed_bs or batch_size

        if effective_bs:
            results, time_info = self._run_batched_predict(images, effective_bs, pad_last=fixed_bs is not None, **post_kwargs)
        else:
            t0 = time.time()
            preprocessed = self.preprocess(images)
            time_info = {"preproc": _elapsed_since(t0)}

            t0 = time.time()
            outputs = self.forward(preprocessed)
            time_info["proc"] = _elapsed_since(t0)

            t0 = time.time()
            results = self.postprocess(outputs, **post_kwargs)
            time_info["postproc"] = _elapsed_since(t0)

        if tile_meta is not None:
            t0 = time.time()
            results = self._merge_tile_predictions(results, tile_meta)
            time_info["postproc"] += _elapsed_since(t0)

        class_map = getattr(self, "class_map", None)
        if class_map is not None:
            confs = self._parse_confidence_config(confidence, list(class_map.values()))
            results = self._apply_confidence_filter(results, confs)

        return results, time_info

    @staticmethod
    def _find_tile_meta(operators: Optional[List]):
        """Return the first TileMeta in operators, or None."""
        if not operators:
            return None
        from lmi_utils.preprocess_utils.ops import TileMeta

        return next((op for op in operators if isinstance(op, TileMeta)), None)

    def _merge_tile_predictions(self, tile_results: Dict, tile_meta) -> Dict:
        """Aggregate per-tile predictions back to one prediction per source image (weighted vote).

        Each source image's tiles contribute their full softmax vector weighted by their max score.
        The class with the highest accumulated weight wins; reported confidence is that weight
        normalised by the sum of all tile weights (fraction of total probability mass on winner).
        """
        all_scores_flat: List[np.ndarray] = tile_results["all_scores"]  # one (n_classes,) array per tile

        merged_classes, merged_scores = [], []
        cursor = 0
        for n_h, n_w in tile_meta.n_tiles:
            n_tiles = n_h * n_w
            tile_vecs = all_scores_flat[cursor : cursor + n_tiles]  # list of (n_classes,)
            cursor += n_tiles

            # weight each tile's softmax vector by its own peak confidence
            weights = np.array([v.max() for v in tile_vecs], dtype=np.float64)
            stacked = np.stack(tile_vecs, axis=0).astype(np.float64)  # (n_tiles, n_classes)
            weighted_sum = (stacked * weights[:, None]).sum(axis=0)  # (n_classes,)
            winner_idx = int(weighted_sum.argmax())

            total_weight = weights.sum()
            confidence = float(weighted_sum[winner_idx] / total_weight) if total_weight > 0 else 0.0

            merged_classes.append(self.class_map[winner_idx])
            merged_scores.append(confidence)

        return {"classes": merged_classes, "scores": merged_scores}

    @staticmethod
    def _parse_confidence_config(confidence, class_names):
        """Normalise the confidence threshold to a per-class dict.

        Args:
            confidence: A float (applied to every class) or a dict mapping class name → threshold.
            class_names: List of class names (values of ``self.class_map``).

        Returns:
            dict mapping every class name to its threshold.

        Raises:
            ValueError: If ``confidence`` is None.
        """
        if confidence is None:
            raise ValueError("confidence must be a float or a dict, not None")
        if isinstance(confidence, (int, float)):
            return {name: float(confidence) for name in class_names}
        if isinstance(confidence, dict):
            return confidence
        raise ValueError(f"confidence must be a float or dict, got {type(confidence).__name__}")

    @staticmethod
    def _apply_confidence_filter(results, confs):
        """Zero out predictions whose score is below the per-class threshold.

        Args:
            results: Dict with "classes" (list of str) and "scores" (list of float).
            confs: Dict mapping class name → threshold.

        Returns:
            New dict with the same list lengths; filtered entries have classes[i] = scores[i] = None.
            A class absent from ``confs`` is always filtered, matching OD.
        """
        classes = list(results["classes"])
        scores = list(results["scores"])
        for i, (cls, score) in enumerate(zip(classes, scores)):
            if cls is None or score is None:
                continue
            if score < confs.get(cls, float("inf")):
                classes[i] = None
                scores[i] = None
        return {"classes": classes, "scores": scores}

    def _run_batched_predict(self, images, batch_size, pad_last=False, **post_kwargs):
        """Run preprocess → forward → postprocess in chunks and collect per-image results.

        Args:
            images: Flat list of HWC images (numpy or tensor).
            batch_size: Number of images per chunk.
            pad_last: If True, zero-pad the last chunk to exactly batch_size (for fixed-batch TRT engines).
            post_kwargs: Passed to ``postprocess``.

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
            t_preproc += _elapsed_since(t0)

            t0 = time.time()
            outputs = self.forward(preprocessed)
            t_proc += _elapsed_since(t0)

            t0 = time.time()
            chunk_results = self.postprocess(outputs, **post_kwargs)
            t_postproc += _elapsed_since(t0)

            for k, v in chunk_results.items():
                all_results.setdefault(k, []).extend(v[:chunk_n])

        return all_results, {"preproc": t_preproc, "proc": t_proc, "postproc": t_postproc}
