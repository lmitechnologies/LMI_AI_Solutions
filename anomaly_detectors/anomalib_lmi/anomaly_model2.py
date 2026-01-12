import logging
import os
from collections import OrderedDict, namedtuple
from collections.abc import Sequence

import numpy as np
import torch
from ad_core.anomaly_detector_registry import AnomalyDetectorRegistry
from image_utils.tiler import OverlapMode, ScaleMode, Tiler
from torchvision.transforms import v2

from .base import Anomalib_Base, to_list

logging.basicConfig()


MINIMUM_QUANT = 1e-12
Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))


@AnomalyDetectorRegistry.register(
    metadata=dict(
        frameworks=["anomalib1"],
        model_names=["patchcore", "padim", "efficientad"],
        tasks=["anomalydetection", "seg"],
        versions=["v1"],
    )
)
class AnomalyModel2(Anomalib_Base):
    """
    Desc: Class used for AD model inference.
    """

    logger = logging.getLogger("AnomalyModel v1")
    logger.setLevel(logging.INFO)

    def __init__(self, model_path, tile=None, stride=None, tile_mode="padding", **kwargs):
        """_summary_

        Args:
            model_path (str): the path to the model file, either a pt or trt engine file
            tile (int | list, optional): tile size [h,w]. Must provide if using tiling
            stride (int | list, optional): stride size [h,w]. Must provide if using tiling
            tile_mode (str, optional): 'padding' or 'resize'. Defaults to 'padding'
        attributes:
            - self.device: device to run model on
            - self.fp16: flag for half precision
            - self.model_shape: model input shape (h,w)
            - self.inference_mode: model inference mode (TRT or PT)
            - self.tiler: tiling object
        """
        if not os.path.isfile(model_path):
            raise Exception(f"Cannot find the model file: {model_path}")

        # set device
        device = kwargs.get("device", "cuda").lower()
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"Unsupported device: {device}. Choose either 'cuda' or 'cpu'.")
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("GPU device unavailable. Use CPU instead.")
            self.device = torch.device("cpu")

        self.image_size = kwargs.get("image_size", [224, 224])

        _, ext = os.path.splitext(model_path)
        self.fp16 = False
        self.logger.info(f"Loading model using {self.device}: {model_path}")
        if ext == ".engine":
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
                self.logger.info(f"binding {name} ({dtype}) with shape {shape}")
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
        elif ext == ".pt":
            try:
                # try loading the model using torchscript
                self.pt_model = torch.jit.load(model_path, map_location=self.device)
                self.model_shape = self.image_size
                self.logger.info(f"Traced model shape: {self.model_shape}")
            except Exception:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.pt_model = checkpoint["model"]
                self.pt_metadata = checkpoint["metadata"]
                self.logger.info(f"Model metadata: {self.pt_metadata}")
                for d in self.pt_model.transform.transforms:
                    if isinstance(d, v2.Resize):
                        self.model_shape = to_list(d.size)
                        self.image_size = to_list(d.size)
                        self.logger.info(f"Model shape: {self.model_shape}")

            self.pt_model.eval()
            self.inference_mode = "PT"
        else:
            raise Exception(f"Unknown model format: {ext}")

        # init tiler
        if tile is not None:
            self.logger.info("Tiling is enabled.")
            if stride is None:
                raise Exception("Must provide stride using tiling")

            tile = to_list(tile)
            if self.model_shape != tile:
                raise Exception(f"tile shape {tile} mismatch with model expected shape: {self.model_shape}")

            self.tiler = Tiler(tile, stride)
            self.tile_mode = ScaleMode.PADDING if tile_mode == "padding" else ScaleMode.INTERPOLATION
            self.logger.info(f"init tiler with tile={tile}, stride={stride}, mode={self.tile_mode}")

    @torch.inference_mode()
    def preprocess(self, image, **kwargs):
        """
        Desc: Preprocess input image.
        args:
            - image: numpy array [H,W,Ch]
        """

        verbose = kwargs.get("verbose", False)
        img = self.from_numpy(image).float()

        # grayscale to rgb
        if img.ndim == 2:
            img = img.unsqueeze(-1).repeat(1, 1, 3)

        img = img.permute((2, 0, 1)).unsqueeze(0)  # to [B,Ch,H,W]
        img = img / 255.0

        if self.tiler is not None:
            img = self.tiler.tile(img, self.tile_mode)

        batch = img.shape[0]
        if self.inference_mode == "TRT" and batch != self.batch_size:
            raise Exception(
                f"Batch size mismatch when using tensorRT model.Got input batch size of {batch}, but tensorRT expects {self.batch_size}."
            )

        # resize inputs. Although resize baked into the original pt model, other model types (trt, ts) may not have it
        if self.tiler is None and (img.shape[2] != self.model_shape[0] or img.shape[3] != self.model_shape[1]):
            if verbose:
                self.logger.info(
                    f"Input image shape mismatch when using non-tiling mode."
                    f"Got input image shape of {img.shape[2:]}, resizing to {self.model_shape}."
                )
            img = v2.Resize(self.model_shape, antialias=True)(img)

        img = img.contiguous()
        return img.half() if self.fp16 else img

    def _infer(self, input_batch):
        """
        Desc: Run inference on the input batch.
        Args:
            - input_batch: preprocessed input batch
        Returns:
            - output: model output tensor
        """
        if self.inference_mode == "TRT":
            self.binding_addrs["input"] = int(input_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            output_tensor = self.bindings["output"].data

        elif self.inference_mode == "PT":
            preds = self.pt_model(input_batch)
            if isinstance(preds, torch.Tensor):
                output_tensor = preds
            elif isinstance(preds, dict):
                output_tensor = preds["anomaly_map"]
            elif isinstance(preds, Sequence):
                output_tensor = preds[1]
            else:
                raise Exception(f"Unknown prediction type: {type(preds)}")

        return output_tensor

    def _setup_tiling_settings(self, kwargs, verbose=False):
        """
        Setup tiling settings from kwargs.

        Args:
            kwargs: Keyword arguments containing tiling_settings
            verbose: Whether to log tiling configuration

        Returns:
            dict: Tiling settings with overlap_mode and scale_mode configured
        """
        tiling_settings = kwargs.get("tiling_settings", {})
        overlap_mode_str = tiling_settings.get("overlap_mode", "average")

        try:
            overlap_mode = OverlapMode(overlap_mode_str)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid overlap mode '{overlap_mode_str}'. Valid options: average, max, cosine, linear, gaussian") from e

        tiling_settings["overlap_mode"] = overlap_mode
        tiling_settings["scale_mode"] = self.tile_mode

        if verbose:
            self.logger.info(f"Using overlap mode: {overlap_mode_str}")

        return tiling_settings

    def _perform_batched_inference(self, input_batch, batch_size):
        """
        Perform inference on input batch in smaller chunks.

        Args:
            input_batch: Preprocessed input tensor [N, C, H, W]
            batch_size: Maximum batch size for each inference call

        Returns:
            torch.Tensor: Concatenated output from all mini-batches

        Raises:
            RuntimeError: If inference fails for any mini-batch
            ValueError: If output type is not supported for aggregation
        """
        num_samples = input_batch.shape[0]
        mini_batch_outputs = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            mini_batch = input_batch[start_idx:end_idx]

            output = self._infer(mini_batch)

            if output is None:
                raise RuntimeError(f"Inference failed for mini-batch [{start_idx}:{end_idx}]. Model returned None.")

            mini_batch_outputs.append(output)

        if not mini_batch_outputs:
            raise RuntimeError(
                f"Batched inference completed but no outputs were collected. Input batch size: {num_samples}, batch size: {batch_size}"
            )

        # Aggregate outputs
        if isinstance(mini_batch_outputs[0], torch.Tensor):
            return torch.cat(mini_batch_outputs, dim=0)
        else:
            raise ValueError(f"Cannot aggregate outputs of type {type(mini_batch_outputs[0]).__name__}. Expected torch.Tensor.")

    def _convert_to_numpy(self, tensor):
        """
        Convert tensor to numpy array.

        Args:
            tensor: torch.Tensor or np.ndarray

        Returns:
            np.ndarray: Numpy array representation

        Raises:
            TypeError: If tensor type is not supported
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            raise TypeError(f"Cannot convert type {type(tensor).__name__} to numpy array. Expected torch.Tensor or np.ndarray.")

    @torch.inference_mode()
    def predict(self, image, **kwargs):
        """
        Perform model prediction on input image(s).

        Args:
            image: Input image as numpy array [H,W,Ch] or [N,H,W,Ch]
            **kwargs: Additional keyword arguments
                tiling_settings (dict): Settings for tiling
                    overlap_mode (str): "average", "max", "cosine", "linear", "gaussian". Default 'average'
                inference_settings (dict): Settings for inference
                    inference_batch_size (int): Batch size for chunked inference. If None, process all at once
                verbose (bool): Enable verbose logging. Default False

        Returns:
            np.ndarray: Prediction output, squeezed to remove singleton dimensions.
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
        tiling_settings = None
        if self.tiler is not None:
            tiling_settings = self._setup_tiling_settings(kwargs, verbose)

        # Preprocess input
        input_batch = self.preprocess(image, verbose=verbose)

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
        if self.tiler is not None:
            output = self.tiler.untile(output, **tiling_settings)

        # Convert to numpy and squeeze
        output_numpy = self._convert_to_numpy(output)

        return np.squeeze(output_numpy)

    def warmup(self, input_hw=None):
        """
        Desc:
            Warm up model using a np zeros array with shape matching model input size.
        Args:
            input_hw(int | list, optional): a int if h equals to w, or a list of [h,w]. Need to specify this if using tiling.
                Otherwise, use model's built-in shape.
        """
        if input_hw is None:
            input_hw = self.model_shape
        input_hw = to_list(input_hw)
        zeros = np.zeros(
            input_hw
            + [
                3,
            ]
        )
        self.logger.info(f"Warming up model with input shape: {zeros.shape}")
        self.predict(zeros)


if __name__ == "__main__":
    import argparse

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
    ad = AnomalyModel2(model_path, args["tile"], args["stride"], mode)

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
