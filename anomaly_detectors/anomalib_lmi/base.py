import json
import logging
import os
import shutil
from typing import Any, Iterable, List

import numpy as np
import torch
from torchvision.transforms import v2

from anomaly_detectors.ad_core.ad_base import ADBase
from lmi_common.onnx_engine import ONNXEngine
from lmi_common.trt_convert import onnx_to_trt
from lmi_common.trt_engine import TRTEngine
from lmi_utils.image_utils.types import ImageLike


def to_list(data) -> List:
    """convert to a two element list

    Args:
        data (int | list): a int or a two element list

    """
    if isinstance(data, int):
        return [data] * 2
    if len(data) != 2:
        raise Exception(f"Must be a two element list, but got {data}")
    return list(data)


class Anomalib_Base(ADBase):
    """Shared base for Anomalib AD backends.

    Concrete inference behavior lives in the per-format subclasses (`AnomalibTRT`,
    `AnomalibONNX`, `AnomalibPT`). This base owns preprocess/postprocess/warmup,
    the `export_onnx`/`export_trt` methods, and shared helpers.

    Public entry points are the per-version factories in `v1/model.py` and
    `v2/model.py`, each subclassing `ModelFactory` to dispatch on file extension.
    """

    logger = logging.getLogger("Anomalib Base")

    def _init_common(self, model_path: str, **kwargs: Any) -> None:
        """Shared __init__ setup for all backends."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Cannot find the model file: {model_path}")
        self.model_path = model_path
        self._setup_device(kwargs.get("device", "cuda"))
        self.image_size = kwargs.get("image_size", [224, 224])
        self.fp16 = False
        self.logger.info(f"Loading model on {self.device}: {model_path}")

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Run inference on the input batch. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement forward()")

    @staticmethod
    def _pick_score_output_idx(names) -> int | None:
        """Return output index for a recognized scalar-score output name, else None."""
        for name in ("pred_score", "pred_scores", "anomaly_score"):
            if name in names:
                return names.index(name)
        return None

    def _pick_anomaly_output_idx(self, names, buffers) -> int:
        if "anomaly_map" in names:
            return names.index("anomaly_map")
        h, w = self.image_size
        for i, name in enumerate(names):
            buf = buffers[name] if isinstance(buffers, dict) else buffers[i]
            if buf.ndim >= 2 and tuple(buf.shape[-2:]) == (h, w):
                return i
        return 0

    @torch.inference_mode()
    def preprocess(self, images: List[ImageLike]) -> torch.Tensor:
        """Convert a list of HWC uint8 images to a batched [N,C,H,W] float tensor.

        Args:
            images (List[ImageLike]): List of uint8 numpy arrays or torch tensors [H,W,C] or [H,W]

        Returns:
            Preprocessed tensor [N,C,H,W] float32 (or float16 if fp16)

        Raises:
            ValueError: If batch size exceeds TensorRT engine limit
        """
        tensors = []
        for image in images:
            img = self.from_numpy(image).float()
            tensors.append(img.permute((2, 0, 1)))  # [C,H,W]

        img = torch.stack(tensors) / 255.0  # [N,C,H,W]

        batch = img.shape[0]
        is_engine = isinstance(self, _AnomalibEngine)
        if is_engine and batch > self.batch_size:
            raise ValueError(f"Batch size {batch} exceeds {type(self).__name__} engine max batch size {self.batch_size}")

        if is_engine and (img.shape[2] != self.image_size[0] or img.shape[3] != self.image_size[1]):
            img = v2.Resize(self.image_size, antialias=False)(img)

        img = img.contiguous()
        return img.half() if self.fp16 else img

    def postprocess(self, output: torch.Tensor, return_numpy: bool = True) -> List[ImageLike]:
        """Convert raw model output to a list of per-image anomaly maps.

        Args:
            output: Model output tensor [N,H,W] or [N,1,H,W].
            return_numpy: If True, return numpy arrays; otherwise return tensors.

        Returns:
            List of anomaly maps [H,W], one per input image.
        """
        output = output.squeeze(1) if output.ndim == 4 else output
        if return_numpy:
            output_np = output.cpu().numpy()
            return [np.squeeze(output_np[i]) for i in range(output_np.shape[0])]
        return [output[i].squeeze() for i in range(output.shape[0])]

    def warmup(self, input_hw=None):
        """Warm up model using a dummy zeros array.

        Args:
            input_hw: Input height and width as int (h==w) or [h, w]. Defaults to the model's built-in shape.
        """
        if input_hw is None:
            input_hw = self.image_size
        input_hw = to_list(input_hw)
        zeros = np.zeros(input_hw + [3], dtype=np.uint8)
        self.logger.info(f"Warming up model with input shape: {zeros.shape}")
        self.predict([zeros])

    def export_onnx(self, export_path, opset_version=14):
        """Export the loaded PT model to ONNX. Requires self.pt_model."""
        if not hasattr(self, "pt_model"):
            raise TypeError(f"{type(self).__name__} has no PT model loaded; load a .pt file first")

        # Write sidecar metadata.json next to the .onnx output, if available.
        if hasattr(self, "pt_metadata"):
            json_file = os.path.join(os.path.dirname(export_path), "metadata.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(self.pt_metadata, f, ensure_ascii=False, indent=4)

        h, w = self.image_size
        torch.onnx.export(
            self.pt_model,
            torch.zeros((1, 3, h, w)).to(self.device),
            export_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
        )
        self.logger.info(f"ONNX model saved at {export_path}")

    def export_trt(self, export_path, fp16=True, workspace_gb=4, min_batch=1, opt_batch=None, max_batch=1):
        """Export to a TRT engine in `export_path`.

        PT-loaded instance: chains PT → ONNX → TRT.
        ONNX-loaded instance: reuses the source .onnx file.
        Engine-loaded instance: nothing to convert.
        """
        if os.path.isfile(export_path):
            raise Exception("Export path should be a directory.")
        os.makedirs(export_path, exist_ok=True)
        trt_path = os.path.join(export_path, "model.engine")

        if hasattr(self, "pt_model"):
            onnx_path = os.path.join(export_path, "model.onnx")
            self.export_onnx(onnx_path)
        elif self.model_path.endswith(".onnx"):
            onnx_path = self.model_path
        else:
            raise TypeError(f"{type(self).__name__} cannot export to TRT; load a .pt or .onnx model first")

        onnx_to_trt(
            onnx_path,
            trt_path,
            fp16=fp16,
            workspace_gb=workspace_gb,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
        )

        sidecar = os.path.join(os.path.dirname(onnx_path), "metadata.json")
        if os.path.isfile(sidecar) and os.path.dirname(sidecar) != export_path:
            shutil.copyfile(sidecar, os.path.join(export_path, "metadata.json"))

    def test(self, *args, **kwargs):
        """Run evaluation on a directory of images. See `anomalib_lmi.evaluate.evaluate` for arguments."""
        from anomaly_detectors.anomalib_lmi.evaluate import evaluate

        return evaluate(self, *args, **kwargs)


class _AnomalibEngine(Anomalib_Base):
    """Shared base for hardware-engine backends (TRT, ONNX).

    Subclasses set `_engine_cls` to the engine wrapper class.
    """

    _engine_cls = None  # subclass sets

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self._init_common(model_path, **kwargs)
        engine = self._engine_cls(model_path, device=str(self.device))
        if len(engine._input_names) != 1:
            raise ValueError(f"Expected a single-input {type(self).__name__} engine, got inputs: {engine._input_names}")
        self.engine = engine
        self.image_size = list(engine.input_shape[-2:])
        self.batch_size = engine.max_batch
        self.fp16 = engine.fp16
        if not engine.is_dynamic:
            self.fixed_batch_size = engine.max_batch

        self._anomaly_output_idx = self._pick_anomaly_output_idx(engine._output_names, engine._output_buffers)
        self._score_output_idx = self._pick_score_output_idx(engine._output_names)
        self.logger.info(
            f"{type(self).__name__} anomaly-map output: index {self._anomaly_output_idx} "
            f"('{engine._output_names[self._anomaly_output_idx]}')"
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.engine.infer(input_batch)[self._anomaly_output_idx]

    def _forward_with_scores(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.engine.infer(input_batch)
        score = outputs[self._score_output_idx] if self._score_output_idx is not None else None
        return outputs[self._anomaly_output_idx], score

    def release(self) -> None:
        self.engine.release()


class AnomalibTRT(_AnomalibEngine):
    """TensorRT engine backend."""

    _engine_cls = TRTEngine


class AnomalibONNX(_AnomalibEngine):
    """ONNX Runtime backend."""

    _engine_cls = ONNXEngine


class AnomalibPT(Anomalib_Base):
    """PyTorch / TorchScript backend.

    Loads either a TorchScript artifact or a full Anomalib checkpoint with metadata.
    """

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self._init_common(model_path, **kwargs)
        try:
            # Try loading as TorchScript model
            self.pt_model = torch.jit.load(model_path, map_location=self.device)
            self.logger.info(f"Loaded TorchScript model with shape: {self.image_size}")
        except Exception:
            # Fall back to loading as checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.pt_model = checkpoint["model"]

            if "metadata" in checkpoint:
                self.pt_metadata = checkpoint["metadata"]
                self.logger.info(f"Model metadata: {self.pt_metadata}")

            model_shape = None
            for transform in self._get_pt_transforms():
                if type(transform).__name__ == "Resize":
                    model_shape = to_list(transform.size)
                    self.logger.info(f"Model shape from transforms: {model_shape}")
                    break
            if model_shape is not None and model_shape != list(self.image_size):
                raise ValueError(f"Model input shape {model_shape} does not match the provided image_size {self.image_size}") from None

        self.pt_model.eval()

    def _get_pt_transforms(self) -> Iterable:
        """Return iterable of preprocessor transforms on the loaded pt_model."""
        return self.pt_model.transform.transforms

    def _extract_pt_output(self, preds: Any) -> torch.Tensor:
        """Extract the anomaly map tensor from the framework-specific PT model output."""
        raise NotImplementedError

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self._extract_pt_output(self.pt_model(input_batch))

    def _forward_with_scores(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Subclasses override to also extract native image-level scores from PT output."""
        return self.forward(input_batch), None


def register_backends(factory_cls, pt_cls) -> None:
    """Register the standard set of file-extension backends on a factory.

    `pt_cls` is the per-version PT backend (e.g. AnomalibPTv1); TRT and ONNX
    backends are version-agnostic and shared across factories.
    """
    factory_cls.register("engine")(AnomalibTRT)
    factory_cls.register("onnx")(AnomalibONNX)
    for ext in ("pt", "ts", "torchscript"):
        factory_cls.register(ext)(pt_cls)
