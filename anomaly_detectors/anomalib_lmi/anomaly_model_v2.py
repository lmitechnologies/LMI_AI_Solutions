import logging
import os
from collections import OrderedDict, namedtuple
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from anomaly_detectors.ad_core.anomaly_detector_registry import AnomalyDetectorRegistry
from lmi_utils.image_utils.tiler import OverlapMode, ScaleMode, Tiler

from .base import Anomalib_Base, to_list

MINIMUM_QUANT = 1e-12
Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

# TensorRT binding names
TRT_INPUT_NAME = "input"
TRT_OUTPUT_NAME = "anomaly_map"


@AnomalyDetectorRegistry.register(
    metadata=dict(
        frameworks=["anomalib2"],
        model_names=["patchcore", "padim", "efficientad"],
        tasks=["anomalydetection", "seg"],
        versions=["v2"],
    )
)
class AnomalyModel_V2(Anomalib_Base):
    """
    Desc: Class used for AD model inference.
    """

    logger = logging.getLogger("AnomalyModel v2")

    def __init__(
        self,
        model_path: str,
        tile: Optional[Union[int, List[int]]] = None,
        stride: Optional[Union[int, List[int]]] = None,
        tile_mode: str = "padding",
        **kwargs: Any,
    ) -> None:
        """Initialize the AnomalyModel_V2.

        Args:
            model_path: Path to the model file (either .pt or .engine)
            tile: Tile size [h,w]. Required if using tiling
            stride: Stride size [h,w]. Required if using tiling
            tile_mode: Tiling mode, either 'padding' or 'resize'
            **kwargs: Additional keyword arguments
                device (str): Device to run on ('cuda' or 'cpu')
                image_size (List[int]): Input image size [h, w]

        Raises:
            FileNotFoundError: If model file does not exist
            ValueError: If device is unsupported or tile/stride mismatch
            RuntimeError: If model loading fails

        Attributes:
            device: Device to run model on
            fp16: Flag for half precision
            model_shape: Model input shape (h,w)
            inference_mode: Model inference mode ('TRT' or 'PT')
            tiler: Tiling object (if tiling is enabled)
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Cannot find the model file: {model_path}")

        # set device
        device = kwargs.get("device", "cuda").lower()
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"Unsupported device: {device}. Choose either 'cuda' or 'cpu'.")
        self.device = torch.device(device)
        if device == "cuda" and torch.cuda.is_available() is False:
            self.logger.warning("GPU device unavailable. Use CPU instead.")
            self.device = torch.device("cpu")

        self.image_size = kwargs.get("image_size", [224, 224])
        self.fp16 = False

        _, ext = os.path.splitext(model_path)
        self.logger.info(f"Loading model on {self.device}: {model_path}")

        if ext == ".engine":
            self._load_tensorrt_model(model_path)
        elif ext == ".pt":
            self._load_pytorch_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {ext}. Expected .pt or .engine")

        # Initialize tiler
        self._init_tiler(tile, stride, tile_mode)

    def _load_tensorrt_model(self, model_path: str) -> None:
        """Load TensorRT engine model.

        Args:
            model_path: Path to .engine file

        Raises:
            RuntimeError: If TensorRT model loading fails
        """
        import tensorrt as trt

        with open(model_path, "rb") as f:
            with trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())

        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []

        for i in range(model.num_bindings):
            name = model.get_tensor_name(i)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            shape = tuple(self.context.get_tensor_shape(name))
            self.logger.info(f"Binding {name} ({dtype}) with shape {shape}")

            if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_shape = shape
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            im = self.from_numpy(np.empty(shape, dtype=dtype))
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.model_shape = list(input_shape[-2:])
        self.image_size = self.model_shape
        self.batch_size = input_shape[0]
        self.inference_mode = "TRT"

    def _load_pytorch_model(self, model_path: str) -> None:
        """Load PyTorch model (TorchScript or checkpoint).

        Args:
            model_path: Path to .pt file

        Raises:
            RuntimeError: If PyTorch model loading fails
        """
        try:
            # Try loading as TorchScript model
            self.pt_model = torch.jit.load(model_path, map_location=self.device)
            self.pt_model.eval()
            self.model_shape = self.image_size
            self.logger.info(f"Loaded TorchScript model with shape: {self.model_shape}")
            # self.model_shape = self.image_size
        except Exception:
            # Fall back to loading as checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.pt_model = checkpoint["model"]

            if "metadata" in checkpoint:
                self.pt_metadata = checkpoint["metadata"]
                self.logger.info(f"Model metadata: {self.pt_metadata}")

            # Extract model shape from preprocessor transforms
            for transform in self.pt_model.pre_processor.transform.transforms:
                if type(transform).__name__ == "Resize":
                    self.model_shape = to_list(transform.size)
                    self.image_size = to_list(transform.size)
                    self.logger.info(f"Model shape from transforms: {self.model_shape}")
                    break

        self.pt_model.eval()
        self.inference_mode = "PT"

    def _init_tiler(
        self,
        tile: Optional[Union[int, List[int]]],
        stride: Optional[Union[int, List[int]]],
        tile_mode: str,
    ) -> None:
        """Initialize tiling configuration.

        Args:
            tile: Tile size [h,w]
            stride: Stride size [h,w]
            tile_mode: Tiling mode ('padding' or 'resize')

        Raises:
            ValueError: If stride is missing or tile shape mismatches model shape
        """
        self.tiler: Optional[Tiler] = None
        self.tile_mode: Optional[ScaleMode] = None

        if tile is not None:
            self.logger.info("Tiling is enabled.")

            if stride is None:
                raise ValueError("Must provide stride when using tiling")

            tile = to_list(tile)
            if self.model_shape != tile:
                raise ValueError(f"Tile shape {tile} mismatch with model expected shape: {self.model_shape}")

            self.tiler = Tiler(tile, stride)
            self.tile_mode = ScaleMode.PADDING if tile_mode == "padding" else ScaleMode.INTERPOLATION
            self.logger.info(f"Initialized tiler with tile={tile}, stride={stride}, mode={self.tile_mode}")

    @torch.inference_mode()
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image for model inference.

        Args:
            image: Input image as numpy array [H,W,C] or [H,W]

        Returns:
            Preprocessed image tensor [B,C,H,W]

        Raises:
            ValueError: If batch size doesn't match TensorRT model expectations
        """
        img = self.from_numpy(image).float()

        # Convert grayscale to RGB
        if img.ndim == 2:
            img = img.unsqueeze(-1).repeat(1, 1, 3)

        # Convert to [B,C,H,W] and normalize
        img = img.permute((2, 0, 1)).unsqueeze(0)
        img = img / 255.0

        # Apply tiling if configured
        if self.tiler is not None:
            self.logger.info(f"Applying tiling to input image with shape {img.shape}")
            img = self.tiler.tile(img, self.tile_mode)

        # Validate batch size for TensorRT
        batch = img.shape[0]
        if self.inference_mode == "TRT" and batch != self.batch_size:
            raise ValueError(f"Batch size mismatch for TensorRT model. Got {batch}, expected {self.batch_size}")

        img = img.contiguous()
        return img.half() if self.fp16 else img

    def _infer(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Run inference on the input batch.

        Args:
            input_batch: Preprocessed input tensor [B,C,H,W]

        Returns:
            Model output anomaly map tensor

        Raises:
            RuntimeError: If TensorRT execution fails
        """
        if self.inference_mode == "TRT":
            self.binding_addrs[TRT_INPUT_NAME] = int(input_batch.data_ptr())
            success = self.context.execute_v2(list(self.binding_addrs.values()))
            if not success:
                raise RuntimeError("TensorRT inference execution failed")
            output_tensor = self.bindings[TRT_OUTPUT_NAME].data

        elif self.inference_mode == "PT":
            preds = self.pt_model(input_batch)
            if isinstance(preds, tuple):
                output_tensor = preds[2]
            else:
                output_tensor = preds.anomaly_map
            self.logger.debug(f"PT model output shape: {output_tensor.shape}, input shape: {input_batch.shape}")
        else:
            raise ValueError(f"Unknown inference mode: {self.inference_mode}")

        return output_tensor

    def _setup_tiling_settings(self, kwargs: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """Setup tiling settings from kwargs.

        Args:
            kwargs: Keyword arguments containing tiling_settings
            verbose: Whether to log tiling configuration

        Returns:
            Tiling settings with overlap_mode and scale_mode configured

        Raises:
            ValueError: If overlap mode is invalid
        """
        tiling_settings = kwargs.get("tiling_settings", {})
        overlap_mode_str = tiling_settings.get("overlap_mode", "average")

        try:
            overlap_mode = OverlapMode(overlap_mode_str)
        except (ValueError, KeyError) as e:
            valid_modes = ", ".join([mode.value for mode in OverlapMode])
            raise ValueError(f"Invalid overlap mode '{overlap_mode_str}'. Valid options: {valid_modes}") from e

        tiling_settings["overlap_mode"] = overlap_mode
        tiling_settings["scale_mode"] = self.tile_mode

        if verbose:
            self.logger.info(f"Using overlap mode: {overlap_mode_str}")

        return tiling_settings

    def _perform_batched_inference(self, input_batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Perform inference on input batch in smaller chunks.

        Args:
            input_batch: Preprocessed input tensor [N, C, H, W]
            batch_size: Maximum batch size for each inference call

        Returns:
            Concatenated output from all mini-batches

        Raises:
            RuntimeError: If inference fails for any mini-batch
            ValueError: If batch_size is invalid or output type is not supported
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        num_samples = input_batch.shape[0]
        mini_batch_outputs: List[torch.Tensor] = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            mini_batch = input_batch[start_idx:end_idx]

            output = self._infer(mini_batch)

            if output is None:
                raise RuntimeError(f"Inference failed for mini-batch [{start_idx}:{end_idx}]. Model returned None.")

            mini_batch_outputs.append(output)

        if not mini_batch_outputs:
            raise RuntimeError(f"Batched inference produced no outputs. Input batch size: {num_samples}, batch size: {batch_size}")

        return torch.cat(mini_batch_outputs, dim=0)

    def _convert_to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array.

        Args:
            tensor: Torch tensor or numpy array

        Returns:
            Numpy array representation

        Raises:
            TypeError: If tensor type is not supported
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            raise TypeError(f"Cannot convert type {type(tensor).__name__} to numpy array. Expected torch.Tensor or np.ndarray")

    @torch.inference_mode()
    def predict(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Perform model prediction on input image(s).

        Args:
            image: Input image as numpy array [H,W,C] or [N,H,W,C]
            **kwargs: Additional keyword arguments
                tiling_settings (dict): Settings for tiling
                    overlap_mode (str): "average", "max", "cosine", "linear", "gaussian" (default: "average")
                inference_settings (dict): Settings for inference
                    inference_batch_size (int): Batch size for chunked inference (default: None = process all at once)
                verbose (bool): Enable verbose logging (default: False)

        Returns:
            Prediction output with singleton dimensions removed.
            If tiling is used, returns the untiled output.

        Raises:
            ValueError: If tiling settings are invalid
            RuntimeError: If inference fails
            TypeError: If output type conversion fails

        Note:
            This method calls preprocess() internally to prepare the input.
        """
        verbose = kwargs.get("verbose", False)

        # Setup tiling if enabled
        tiling_settings: Optional[Dict[str, Any]] = None
        if self.tiler is not None:
            tiling_settings = self._setup_tiling_settings(kwargs, verbose)

        # Preprocess input
        input_batch = self.preprocess(image)

        # Perform inference (batched or single)
        inference_settings = kwargs.get("inference_settings", {})
        batch_size = inference_settings.get("inference_batch_size")

        if verbose:
            self.logger.info(f"Input batch shape: {input_batch.shape}, mini-batch size: {batch_size}")

        if batch_size is not None and batch_size > 0:
            output = self._perform_batched_inference(input_batch, batch_size)
        else:
            output = self._infer(input_batch)

        if output is None:
            raise RuntimeError(f"Model inference failed to produce output. Input shape: {input_batch.shape}, Mode: {self.inference_mode}")

        # Untile if tiling was used
        if self.tiler is not None and tiling_settings is not None:
            output = self.tiler.untile(output, **tiling_settings)

        # Convert to numpy and squeeze
        output_numpy = self._convert_to_numpy(output)

        return np.squeeze(output_numpy)

    def warmup(self, input_hw: Optional[Union[int, List[int]]] = None) -> None:
        """Warm up model using a dummy zeros array.

        Args:
            input_hw: Input height and width. Can be:
                - int: Uses same value for h and w
                - List[int]: [h, w]
                - None: Uses model's built-in shape (default)
                Must be specified if using tiling.

        Note:
            This method helps initialize CUDA/TensorRT contexts for better performance.
        """
        if input_hw is None:
            input_hw = self.model_shape
        input_hw = to_list(input_hw)

        # Create dummy RGB image [H, W, 3]
        zeros = np.zeros(input_hw + [3], dtype=np.uint8)

        self.logger.info(f"Warming up model with input shape: {zeros.shape}")
        self.predict(zeros, verbose=True)

    def _cleanup_resources(self) -> None:
        """Internal helper to cleanup resources."""
        if hasattr(self, "context") and self.context is not None:
            del self.context
        if hasattr(self, "bindings"):
            self.bindings.clear()

    def __del__(self) -> None:
        """Cleanup resources when object is destroyed."""
        self._cleanup_resources()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup_resources()
        return False


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest="action", required=True, help="Action modes: test or convert")

    test_ap = subs.add_parser("test", help="test model")
    test_ap.add_argument("-i", "--model_path", default="/app/model/model.pt", help="Input model file path.")
    test_ap.add_argument("-d", "--data_dir", default="/app/data", help="Data file directory.")
    test_ap.add_argument(
        "-o",
        "--annot_dir",
        default="/app/annotation_results",
        help="Annot file directory.",
    )
    test_ap.add_argument("-g", "--generate_stats", action="store_true", help="generate the data stats")
    test_ap.add_argument("-p", "--plot", action="store_true", help="plot the annotated images")
    test_ap.add_argument("-t", "--ad_threshold", type=float, default=None, help="AD patch threshold.")
    test_ap.add_argument("-m", "--ad_max", type=float, default=None, help="AD patch max anomaly.")
    test_ap.add_argument("--tile", type=int, nargs=2, default=None, help="tile size (h,w)")
    test_ap.add_argument("--stride", type=int, nargs=2, default=None, help="stride size (h,w)")
    test_ap.add_argument("--resize", action="store_true", help="use resize for tiling")
    test_ap.add_argument(
        "-om",
        "--overlap_mode",
        default="gaussian",
        help='overlap mode for tiling, can be "average", "max", "cosine", "linear", "gaussian"',
    )
    test_ap.add_argument(
        "-device",
        "--device",
        type=str,
        default="cuda",
        help="device for inference. cuda",
    )

    convert_ap = subs.add_parser("convert", help="convert model to trt engine")
    convert_ap.add_argument("-i", "--model_path", default="/app/model/model.pt", help="Input model file path.")
    convert_ap.add_argument("-o", "--export_dir", default="/app/export")
    convert_ap.add_argument(
        "-c",
        "--convert_type",
        default="trt",
        type=str,
        choices=["trt", "onnx"],
        help="convert type: trt or onnx",
    )
    convert_ap.add_argument(
        "--hw",
        type=int,
        nargs=2,
        default=None,
        help="input image shape (h,w). Muse be provided if using tiling",
    )
    convert_ap.add_argument("--tile", type=int, nargs=2, default=None, help="tile size (h,w)")
    convert_ap.add_argument("--stride", type=int, nargs=2, default=None, help="stride size (h,w)")
    convert_ap.add_argument("--resize", action="store_true", help="use resize for tiling, otherwise pad zeros")
    args = vars(ap.parse_args())

    action = args["action"]
    model_path = args["model_path"]

    mode = "resize" if args["resize"] else "padding"
    ad = AnomalyModel_V2(model_path, args["tile"], args["stride"], mode, device=args["device"])

    if action == "convert":
        export_dir = args["export_dir"]
        os.makedirs(export_dir, exist_ok=True)
        if args["convert_type"] == "onnx":
            onnx_path = os.path.join(export_dir, "model.onnx")
            ad.convert_to_onnx(onnx_path, args["hw"])
        if args["convert_type"] == "trt":
            ad.convert(model_path, export_dir, args["hw"])
    elif action == "test":
        os.makedirs(args["annot_dir"], exist_ok=True)
        ad.test(
            args["data_dir"],
            args["annot_dir"],
            args["generate_stats"],
            args["plot"],
            args["ad_threshold"],
            args["ad_max"],
            args["overlap_mode"],
        )
