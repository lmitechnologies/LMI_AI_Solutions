import logging
import os
from collections.abc import Sequence

import torch
from torchvision.transforms import v2

from anomaly_detectors.ad_core.anomaly_detector_registry import AnomalyDetectorRegistry

from ..base import Anomalib_Base, to_list


@AnomalyDetectorRegistry.register(
    metadata=dict(
        frameworks=["anomalib1"],
        model_names=["patchcore", "padim", "efficientad"],
        tasks=["anomalydetection", "seg"],
        versions=["v1"],
    )
)
class AnomalyModel(Anomalib_Base):
    """
    Desc: Class used for AD model inference.
    """

    logger = logging.getLogger("AnomalyModel v1")

    def __init__(self, model_path, **kwargs):
        """
        Args:
            model_path (str): the path to the model file, either a pt or trt engine file
            **kwargs: Additional keyword arguments
                    device (str): Device to run on ('cuda' or 'cpu')
                    image_size (List[int]): Input image size [h, w]

        Attributes:
            - device: device to run model on
            - fp16: flag for half precision
            - image_size: model input shape (h,w)
            - inference_mode: model inference mode (TRT or PT)
        """
        if not os.path.isfile(model_path):
            raise Exception(f"Cannot find the model file: {model_path}")

        # set device
        device = kwargs.get("device", "cuda")
        self._setup_device(device)

        self.image_size = kwargs.get("image_size", [224, 224])
        self.fp16 = False

        self.logger.info(f"Loading model using {self.device}: {model_path}")
        _, ext = os.path.splitext(model_path)
        if ext == ".engine":
            self._load_tensorrt_model(model_path)
        elif ext == ".pt":
            self._load_pytorch_model(model_path)
        else:
            raise Exception(f"Unknown model format: {ext}. Expected .pt or .engine")

    def _load_pytorch_model(self, model_path: str) -> None:
        try:
            # Try loading as TorchScript model
            self.pt_model = torch.jit.load(model_path, map_location=self.device)
            self.logger.info(f"Traced model shape: {self.image_size}")
        except Exception:
            # Fall back to loading as checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.pt_model = checkpoint["model"]

            if "metadata" in checkpoint:
                self.pt_metadata = checkpoint["metadata"]
                self.logger.info(f"Model metadata: {self.pt_metadata}")

            # Extract model shape from preprocessor transforms
            model_shape = None
            for d in self.pt_model.transform.transforms:
                if isinstance(d, v2.Resize):
                    model_shape = to_list(d.size)
                    self.logger.info(f"Model shape from transforms: {model_shape}")
                    break
            if model_shape is not None and model_shape != list(self.image_size):
                raise Exception(f"Model input shape {model_shape} does not match the provided image_size {self.image_size}") from None

        self.pt_model.eval()
        self.inference_mode = "PT"

    def forward(self, input_batch):
        """
        Desc: Run inference on the input batch.
        Args:
            - input_batch: preprocessed input batch
        Returns:
            - output: model output tensor
        """
        if self.inference_mode == "TRT":
            output_tensor = self.trt.infer(input_batch)[0]

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


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest="action", required=True, help="Action modes: test or convert")

    test_ap = subs.add_parser("test", help="test model")
    test_ap.add_argument("-i", "--model_path", default="/app/model/model.pt", help="Input model file path.")
    test_ap.add_argument("-d", "--data_dir", default="/app/data", help="Data file directory.")
    test_ap.add_argument("-o", "--annot_dir", default="/app/annotation_results", help="Annot file directory.")
    test_ap.add_argument("-g", "--generate_stats", action="store_true", help="generate the data stats")
    test_ap.add_argument("-p", "--plot", action="store_true", help="plot the annotated images")
    test_ap.add_argument("-t", "--ad_threshold", type=float, default=None, help="AD patch threshold.")
    test_ap.add_argument("-m", "--ad_max", type=float, default=None, help="AD patch max anomaly.")
    test_ap.add_argument("--tile", type=int, nargs=2, default=None, help="tile size (h,w)")
    test_ap.add_argument("--stride", type=int, nargs=2, default=None, help="stride size (h,w)")
    test_ap.add_argument(
        "-om",
        "--overlap_mode",
        default="gaussian",
        help='overlap blending mode for tiling: "average", "max", "cosine", "linear", "gaussian"',
    )
    test_ap.add_argument(
        "--scale_mode", default="padding", choices=["padding", "interpolation"], help="tile scaling mode: padding or interpolation"
    )
    test_ap.add_argument("--limit", type=int, default=None, help="process only the first N images")

    convert_ap = subs.add_parser("convert", help="convert model to trt engine")
    convert_ap.add_argument("-i", "--model_path", default="/app/model/model.pt", help="Input model file path.")
    convert_ap.add_argument("-o", "--export_dir", default="/app/export")
    convert_ap.add_argument("-c", "--convert_type", default="trt", choices=["trt", "onnx"], help="convert type: trt or onnx")
    args = vars(ap.parse_args())

    action = args["action"]
    model_path = args["model_path"]

    ad = AnomalyModel(model_path)

    if action == "convert":
        export_dir = args["export_dir"]
        os.makedirs(export_dir, exist_ok=True)
        ad.convert(model_path, export_dir, convert_type=args["convert_type"])
    elif action == "test":
        os.makedirs(args["annot_dir"], exist_ok=True)
        ad.test(
            args["data_dir"],
            args["annot_dir"],
            args["generate_stats"],
            args["plot"],
            args["ad_threshold"],
            args["ad_max"],
            args["tile"],
            args["stride"],
            args["overlap_mode"],
            args["scale_mode"],
            args["limit"],
        )
