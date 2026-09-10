import abc
import inspect
import logging
import time
from typing import List

import cv2
import numpy as np
import torch

import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils
from lmi_utils.image_utils.img_resize import resize_and_pad
from lmi_utils.image_utils.types import ImageBatch, ImageLike, normalize_image_batch, to_3channel

from .results import Results


class ODBase(abc.ABC):
    logger = logging.getLogger(__name__)

    # Model input size as [height, width]; subclasses must set it.
    image_size: list = None

    # Set to a positive integer in subclasses that use a fixed-batch-size model.
    fixed_batch_size: int = None

    # True = letterbox, False = stretch. Required on concrete subclasses (see __init_subclass__).
    RESIZE_PRESERVE_ASPECT: bool = None

    # Letterbox pad fill for auto-injected resize; match training (e.g. 114 for YOLO). Ignored when stretching.
    RESIZE_PAD_VALUE: int = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip abstract intermediates; their concrete leaves inherit the value.
        if not inspect.isabstract(cls) and cls.RESIZE_PRESERVE_ASPECT is None:
            raise TypeError(f"{cls.__name__} must set RESIZE_PRESERVE_ASPECT (True=letterbox, False=stretch)")

    @abc.abstractmethod
    def warmup(self, *args, **kwargs):
        pass

    def release(self) -> None:  # noqa: B027 — intentional no-op default, not abstract
        """Free resources that need deterministic teardown (e.g. TRT engine/context). Default no-op."""

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def postprocess(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def predict(self, image: ImageBatch, configs, operators=None, **kwargs):
        """Run the full inference pipeline: preprocess → forward → postprocess.

        Supports both single image and batch inference. When ``self.fixed_batch_size``
        is set (e.g. TRT engines with a hard-coded batch dimension), the input is
        processed in chunks of that size and the last chunk is zero-padded.

        Return tensors if input image are tensors, otherwise return numpy arrays.

        Args:
            image: A single HW or HWC image, a list of HW/HWC images, or a BHWC batch.
                Accepts both numpy arrays and torch tensors. 2D (HW) images are
                expanded to 3-channel RGB. All images in a batch must have the same dimensions.
            configs: Confidence threshold (float) or per-class thresholds (dict).
            operators: Unified preprocessing history for coordinate reversion, applied to the whole
                batch at once after inference. Accepts:
                - None: no coordinate reversion.
                - List of history entries matching what ``Preprocessor.preprocess()`` returns.
                  Each entry's batched fields must have length 1 (broadcast to all images) or
                  equal to the number of images that entry saw.
        kwargs:
            batch_size (int): chunk size for dynamic mini-batch inference (default: None = all at once).
                Ignored when self.fixed_batch_size is set.
            return_segments (bool): Whether to return 'segments' in the output dict when available.

        Returns:
            (results, time_info)
            results (dict): a dictionary where each value is a list with one entry per image, e.g., {
                'boxes': [numpy or tensor, ...],
                'scores': [numpy or tensor, ...],
                'classes': [numpy strings, ...],
                'masks': [numpy or tensor, ...],
                'segments': [[numpy or tensor], ...],
            }
            time_info (dict): timing info with keys 'preproc', 'proc', 'postproc'.

            The lists are as long as the input batch, except when ``operators`` contains a tile
            entry: the tiles are merged, so the results are one entry per source image.
        """
        images = [to_3channel(img) for img in normalize_image_batch(image)]
        operators = self._prepare_operators(operators, len(images))
        use_tensor = isinstance(images[0], torch.Tensor) if images else False

        fixed_bs = self.fixed_batch_size
        batch_size = kwargs.pop("batch_size", None)
        effective_bs = fixed_bs or batch_size

        if effective_bs:
            all_results, time_info = self._run_batched_predict(images, configs, effective_bs, pad_last=fixed_bs is not None, **kwargs)
        else:
            t0 = time.time()
            preprocessed = self.preprocess(images)
            time_info = {"preproc": time.time() - t0}

            t0 = time.time()
            outputs = self.forward(preprocessed)
            time_info["proc"] = time.time() - t0

            t0 = time.time()
            all_results = self.postprocess(outputs, images=images, configs=configs, preprocessed=preprocessed, **kwargs)
            time_info["postproc"] = time.time() - t0

        # revert while still on the model's device: merging tiled masks on the CPU is several times slower
        results = self._aggregate_results(all_results, return_numpy=False)
        t0 = time.time()
        results = self._revert_coordinates(results, operators)
        time_info["postproc"] += time.time() - t0
        if not use_tensor:
            results = self._results_to_numpy(results, float32=bool(operators))
        return results, time_info

    def _run_batched_predict(self, images, configs, batch_size, pad_last=False, **kwargs):
        """Run preprocess → forward → postprocess in chunks and collect per-image results.

        Coordinates stay in preprocessed space; ``predict`` reverts the whole batch at the end.

        Args:
            images: Flat list of HWC images (numpy or tensor).
            configs: Confidence threshold passed through to postprocess.
            batch_size: Number of images per chunk.
            pad_last: If True, zero-pad the last chunk to exactly batch_size (for fixed-batch TRT engines).

        Returns:
            (List[Results], time_info dict with keys 'preproc', 'proc', 'postproc').
        """
        all_results = []
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
            list_results = self.postprocess(outputs, images=chunk_imgs, configs=configs, preprocessed=preprocessed, **kwargs)
            t_postproc += time.time() - t0

            all_results.extend(list_results[:chunk_n])

        return all_results, {"preproc": t_preproc, "proc": t_proc, "postproc": t_postproc}

    def _fit_to_input_size(
        self,
        images: List[ImageLike],
        preserve_aspect: bool = True,
        resize_fn=None,
        channels_first: bool = False,
    ) -> List[ImageLike]:
        """Resize off-size images to the model input (``self.image_size``). Perform resize_and_pad by default.

        Args:
            ``images``: images (numpy or tensor) to resize if they don't match the model input.
            ``preserve_aspect``: ``True`` for letterbox, ``False`` for stretch. Only affects the
                default resize_and_pad and the warning wording (ignored when ``resize_fn`` is given).
            ``resize_fn``: optional resize function ``(image, (th, tw)) -> image`` to use. Overrides the default resize_and_pad when given.
            ``channels_first``: ``True`` if images are CHW, else HWC.

        """
        target = getattr(self, "image_size", None)
        if not target:
            raise ValueError(
                f"{type(self).__name__}.image_size is not set; cannot fit inputs to model size. "
                "Set self.image_size = [h, w] in the backend's __init__."
            )

        th, tw = int(target[0]), int(target[1])
        out, mismatched = [], None
        for im in images:
            h, w = (im.shape[1], im.shape[2]) if channels_first else im.shape[:2]
            if (h, w) != (th, tw):
                mismatched = mismatched or (h, w)
                if resize_fn is not None:
                    im = resize_fn(im, (th, tw))
                else:
                    im = resize_and_pad(im, width=tw, height=th, preserve_aspect=preserve_aspect)
            out.append(im)

        if mismatched is not None:
            mh, mw = mismatched
            mode = "letterbox" if preserve_aspect else "stretch"
            self.logger.warning(
                f"Input size {mh}x{mw} != model input {th}x{tw}; auto-resizing ({mode}) to fit. "
                "Add a matching resize to your preprocessing pipeline to silence this and avoid the extra resize."
            )
        return out

    @staticmethod
    def _to_numpy(data):
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

    def _setup_device(self, device: str) -> None:
        """Set up the computation device (CPU or GPU).

        Args:
            device (str): The device to be used, either 'cpu' or 'cuda'.
        """
        if device.lower() not in ["cpu", "cuda"]:
            raise ValueError(f'Invalid device: {device}. Supported devices are "cpu" and "cuda".')

        self.device = torch.device("cpu")
        if device.lower() == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.logger.warning("GPU not available, falling back to CPU")

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
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)), otypes=[np.str_])

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
        filtered_masks = masks[keep] if masks is not None and len(masks) > 0 else []
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
    def _prepare_operators(operators, batch_size: int) -> list:
        """Validate a typed preprocessing history and broadcast length-1 records to the batch.

        The history is a list of typed ``Meta`` records (struct-of-arrays). Walking it backwards
        (revert order) tracks how many results each record will be handed: most records leave the
        count alone, while a tile record folds its tiles back onto one entry per source image.

        Returns:
            The history with every batched field sized to what its record sees during a single
            Reconstructor pass over the whole batch.
        """
        from dataclasses import fields

        from lmi_utils.preprocess_utils.operation import Meta
        from lmi_utils.preprocess_utils.ops import TileMeta

        if not operators:
            return []
        if not isinstance(operators, list) or not all(isinstance(e, Meta) for e in operators):
            raise ValueError("operators must be a list of typed Meta records.")

        prepared = []
        count = batch_size
        for entry in reversed(operators):
            if isinstance(entry, TileMeta):
                n_tiles = sum(h * w for h, w in entry.n_tiles)
                if n_tiles != count:
                    raise ValueError(f"tile history entry describes {n_tiles} tiles, but {count} images were passed to predict().")
                count = len(entry.n_tiles)
                prepared.append(entry)
                continue

            entry_fields = fields(entry)
            list_field = next((f for f in entry_fields if isinstance(getattr(entry, f.name), list)), None)
            n = len(getattr(entry, list_field.name)) if list_field else 1
            if n not in (1, count):
                raise ValueError(f"history entry '{type(entry).__name__}' batch size {n} is not 1 (broadcast) or {count} (per-image).")
            if n == count:
                prepared.append(entry)
                continue
            broadcast = type(entry).__new__(type(entry))
            for f in entry_fields:
                v = getattr(entry, f.name)
                object.__setattr__(broadcast, f.name, [v[0]] * count if isinstance(v, list) else v)
            prepared.append(broadcast)

        prepared.reverse()
        return prepared

    def _revert_coordinates(self, results: dict, operators: list, round: bool = True) -> dict:
        """Revert a batch of predictions to the original pre-transform space.

        Reverts boxes (regular and OBB), masks, segments, and points (with optional
        visibility column) in a single Reconstructor pass over ``operators``. No-op
        when operators is empty.

        Args:
            results: Aggregated batch dict as built by ``_aggregate_results`` — every value
                is a list with one entry per image.
            operators: History whose batched fields are all at ``len(results)`` (see ``_prepare_operators``).
            round: Round and clamp the reverted point-like coords (boxes, segments,
                and the xy of points) to non-negative integers, matching
                ``revert_to_origin``. Masks are always left as resampled floats.

        Returns:
            The reverted batch dict. Entries are keyed by source image, so the length can differ
            from the input when an operator maps many inputs onto one image (e.g. tiling).
        """
        if not operators or not results:
            return results

        mask_dtype = next((m.dtype for m in results.get("masks") or [] if m is not None and len(m)), None)
        reverted = pipeline_utils._reconstructor().reconstruct_coordinates(results, operators)

        if round and "boxes" in reverted:  # (N,4) xyxy or (N,4,2) OBB
            reverted["boxes"] = [self._round_clamp_coords(b) if b is not None and len(b) else b for b in reverted["boxes"]]

        if round and "segments" in reverted:
            reverted["segments"] = [
                [self._round_clamp_coords(s) if len(s) else s for s in segs] if segs is not None else segs for segs in reverted["segments"]
            ]

        if round and "points" in reverted:  # (N,K,2) or (N,K,3) with a trailing visibility column
            reverted["points"] = [self._round_clamp_points(p) for p in reverted["points"]]

        if mask_dtype is not None and "masks" in reverted:  # resampled, not rounded; restore the original dtype
            reverted["masks"] = [self._cast_masks(m, mask_dtype) for m in reverted["masks"]]

        return reverted

    @staticmethod
    def _round_clamp_coords(value):
        """Round coords to nearest int and clamp to non-negative; mirrors ``revert_to_origin(round=True)``."""
        if torch.is_tensor(value):
            return value.round().clamp(min=0)
        return np.clip(np.round(value), 0, None)

    @classmethod
    def _round_clamp_points(cls, pts):
        """Round only the xy of keypoints; a trailing visibility column is left untouched."""
        if pts is None or not len(pts):
            return pts
        if pts.shape[-1] == 3:
            xy, vis = cls._round_clamp_coords(pts[..., :2]), pts[..., 2:]
            return torch.cat((xy, vis), dim=-1) if torch.is_tensor(pts) else np.concatenate((xy, vis), axis=-1)
        return cls._round_clamp_coords(pts)

    @staticmethod
    def _cast_masks(masks, dtype):
        if masks is None or not len(masks):
            return masks
        return masks.to(dtype) if torch.is_tensor(masks) else masks.astype(dtype)

    @staticmethod
    def _results_to_numpy(results: dict, float32: bool) -> dict:
        """Move an aggregated batch dict to numpy.

        ``float32`` casts boxes, scores, points and segments, matching what a revert on numpy input returned.
        """
        cast = {"boxes", "scores", "points", "segments"} if float32 else set()

        def to_numpy(key, v):
            if not isinstance(v, torch.Tensor):
                return v
            a = v.cpu().numpy()
            return a.astype(np.float32) if key in cast else a

        out = {}
        for k, vs in results.items():
            out[k] = [[to_numpy(k, s) for s in v] if k == "segments" and isinstance(v, list) else to_numpy(k, v) for v in vs]
        return out

    @staticmethod
    def _aggregate_results(list_results: List[Results], return_numpy: bool = True) -> dict:
        """Aggregate a list of Results into a single dict with per-image lists.

        Args:
            list_results: List of Results objects.
            return_numpy: If True, convert tensors to numpy arrays; if False, keep as-is.

        Returns:
            Dict with keys from Results._all_keys (boxes, scores, classes, masks, points, segments).
        """
        if not list_results:
            return {}
        all_dicts = [result.to_dict(return_numpy=return_numpy) for result in list_results]
        active_keys = [k for k in Results._all_keys if any(k in d for d in all_dicts)]
        result = {}
        for k in active_keys:
            result[k] = [d.get(k) for d in all_dicts]
        return result

    @staticmethod
    def annotate_image(results, image: ImageLike, colormap=None, **kwargs) -> np.ndarray:
        """Annotate model results on a single image.

        Args:
            results (dict): Detection results with the keys: boxes, classes, scores, masks, segments, points.
            image (np.ndarray | torch.Tensor): Input image.
            colormap (dict, optional): Maps class name to RGB tuple. Defaults to None (random colors).

        kwargs:
            line_thickness (int, optional): Bounding box line thickness.
            hide_label (bool, optional): If True, suppress class/score labels.
            hide_bbox (bool, optional): If True, suppress bounding boxes.
            plot_segments (bool, optional): If True, plot segments when available.

        Returns:
            np.ndarray: Annotated copy of the image.
        """
        boxes = results.get("boxes", [])
        classes = results.get("classes", [])
        scores = results.get("scores", [])
        masks = results.get("masks", [])
        points = results.get("points", [])

        image = ODBase._to_numpy(image).copy()
        if not len(boxes):
            return image

        hide_label = kwargs.get("hide_label", False)
        hide_bbox = kwargs.get("hide_bbox", False)
        line_thickness = kwargs.get("line_thickness", None)

        boxes = ODBase._to_numpy(boxes)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if len(masks):
            masks = ODBase._to_numpy(masks)
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
            points = ODBase._to_numpy(points).astype(int)
            for i in range(len(points)):
                for j in range(len(points[i])):
                    cv2.circle(image, (points[i][j][0], points[i][j][1]), 4, (255, 255, 255), -1)

        segments = results.get("segments", [])
        if len(segments) and kwargs.get("plot_segments", False):
            for seg in segments:
                seg = ODBase._to_numpy(seg).astype(int)
                color = (255, 255, 255)
                pts = seg.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=line_thickness or 2)

        return image
