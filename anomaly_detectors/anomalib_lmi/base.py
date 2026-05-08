import json
import logging
import os
import subprocess
from typing import Any, Iterable, List

import numpy as np
import torch
from torchvision.transforms import v2

from anomaly_detectors.ad_core.ad_base import ADBase
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
    logger = logging.getLogger("Anomalib Base")

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        """Initialize the AnomalyModel.

        Args:
            model_path: Path to the model file (either .pt or .engine)
            **kwargs:
                device (str): Device to run on ('cuda' or 'cpu')
                image_size (List[int]): Input image size [h, w]
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Cannot find the model file: {model_path}")

        self._setup_device(kwargs.get("device", "cuda"))
        self.image_size = kwargs.get("image_size", [224, 224])
        self.fp16 = False

        self.logger.info(f"Loading model on {self.device}: {model_path}")
        ext = os.path.splitext(model_path)[1]
        if ext == ".engine":
            self._load_tensorrt_model(model_path)
        elif ext in [".pt", ".torchscript", ".ts"]:
            self._load_pytorch_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {ext}. Expected '.pt', '.torchscript', '.ts', or '.engine'")

    def _load_pytorch_model(self, model_path: str) -> None:
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
        self.inference_mode = "PT"

    def _get_pt_transforms(self) -> Iterable:
        """Return iterable of preprocessor transforms on the loaded pt_model.

        Subclasses override to point at the framework-version-specific transform location.
        """
        return self.pt_model.transform.transforms

    def _extract_pt_output(self, preds: Any) -> torch.Tensor:
        """Extract the anomaly map tensor from the framework-specific PT model output."""
        raise NotImplementedError

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Run inference on the input batch.

        Args:
            input_batch: Preprocessed input tensor [B,C,H,W]

        Returns:
            Model output anomaly map tensor.
        """
        if self.inference_mode == "TRT":
            return self.trt.infer(input_batch)[0]
        if self.inference_mode == "PT":
            return self._extract_pt_output(self.pt_model(input_batch))
        raise ValueError(f"Unknown inference mode: {self.inference_mode}")

    def _load_tensorrt_model(self, model_path: str) -> None:
        self.trt = TRTEngine(model_path, device=str(self.device))
        if len(self.trt._input_names) != 1:
            raise ValueError(f"Expected a single-input TRT engine, got inputs: {self.trt._input_names}")
        self.image_size = list(self.trt.input_shape[-2:])
        self.batch_size = self.trt.max_batch
        self.fp16 = self.trt.fp16
        self.inference_mode = "TRT"
        if not self.trt.is_dynamic:
            self.fixed_batch_size = self.trt.max_batch

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
        if self.inference_mode == "TRT" and batch > self.batch_size:
            raise ValueError(f"Batch size {batch} exceeds TensorRT engine max batch size {self.batch_size}")

        if self.inference_mode == "TRT" and (img.shape[2] != self.image_size[0] or img.shape[3] != self.image_size[1]):
            img = v2.Resize(self.image_size, antialias=True)(img)

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

    def convert_to_onnx(self, export_path, opset_version=14):
        """
        Desc: Convert existing .pt file to onnx
        Args:
            - path to output .onnx file
            - opset_version: onnx version ID
        """
        # write metadata to export path
        json_file = os.path.join(os.path.dirname(export_path), "metadata.json")
        if hasattr(self, "pt_metadata"):
            with open(json_file, "w", encoding="utf-8") as metadata_file:
                json.dump(self.pt_metadata, metadata_file, ensure_ascii=False, indent=4)

        b, c = 1, 3
        h, w = self.image_size
        torch.onnx.export(
            self.pt_model,
            torch.zeros((b, c, h, w)).to(self.device),
            export_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
        )

    def convert_trt(self, onnx_path, out_engine_path, fp16, workspace=4096):
        """
        Desc: Convert an onnx to trt engine
        Args:
            - onnx_path: input file path
            - out_engine_path: output file path
            - fp16: set fixed point width
            - workspace: conversion memory size in MB
        """
        if not out_engine_path.endswith(".engine"):
            raise Exception("trt engine file must end with '.engine'")

        out_dir = os.path.dirname(out_engine_path)
        os.makedirs(out_dir, exist_ok=True)

        # run convert cmd
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={out_engine_path}",
            f"--memPoolSize=workspace:{workspace}",
        ]
        if fp16:
            cmd.append("--fp16")
        subprocess.run(cmd, check=True)

        # check if metadata.json exists in the same directory as onnx_path
        onnx_dir = os.path.dirname(onnx_path)
        if os.path.isfile(f"{onnx_dir}/metadata.json"):
            cmd2 = [f"cp -sf {onnx_dir}/metadata.json {out_dir}"]
            subprocess.run(cmd2, shell=True)
        else:
            self.logger.warning(f"metadata.json not found in {onnx_dir}")

    def convert(self, model_path, export_path, fp16=True, convert_type="trt"):
        """
        Desc: Converts .onnx or .pt file to ONNX or TensorRT engine.

        Args:
            - model_path: model file path (.pt or .onnx)
            - export_path: output directory
            - fp16: use half precision for TRT conversion
            - convert_type: "onnx" to export ONNX only, "trt" to export TensorRT engine
        """
        if os.path.isfile(export_path):
            raise Exception("Export path should be a directory.")
        ext = os.path.splitext(model_path)[1]

        if convert_type == "onnx":
            if ext != ".pt":
                raise ValueError(f"ONNX export requires a .pt input, got {ext}")
            self.logger.info("Converting pt to onnx...")
            onnx_path = os.path.join(export_path, "model.onnx")
            self.convert_to_onnx(onnx_path)
            self.logger.info(f"ONNX model saved at {onnx_path}")
        elif convert_type == "trt":
            if ext not in (".pt", ".onnx"):
                raise ValueError(f"TRT export requires a .pt or .onnx input, got {ext}")
            onnx_path = model_path
            if ext == ".pt":
                self.logger.info("Converting pt to onnx...")
                onnx_path = os.path.join(export_path, "model.onnx")
                self.convert_to_onnx(onnx_path)
                self.logger.info(f"ONNX model saved at {onnx_path}")
            self.logger.info("Converting onnx to trt engine...")
            trt_path = os.path.join(export_path, "model.engine")
            self.convert_trt(onnx_path, trt_path, fp16)
        else:
            raise ValueError(f"Unknown convert_type: {convert_type!r}. Expected 'onnx' or 'trt'")

    def test(self, *args, **kwargs):
        """Run evaluation on a directory of images. See `anomalib_lmi.evaluate.evaluate` for arguments."""
        from anomaly_detectors.anomalib_lmi.evaluate import evaluate

        return evaluate(self, *args, **kwargs)
