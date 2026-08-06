import logging
import os

import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import smart_inference_mode


class YoloCore:
    logger = logging.getLogger("yolo-core")
    task = ""

    def __init__(self, model_path: str, device="cuda", data=None, fp16=False, **kwargs) -> None:
        """init the model
        Args:
            model_path (str): the path to the model_path file.
            device (str, optional): the device to be used, either 'cuda' or 'cpu'. Defaults to 'cuda'.
            data (str, optional): the path to dataset yaml file. Defaults to None.
            fp16 (bool, optional): Whether to use fp16. Defaults to False.
        Raises:
            FileNotFoundError: the model_path file does not exist
        """
        image_size = kwargs.get("image_size", None)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"File not found: {model_path}")

        self._setup_device(device)

        # load model
        self.model = AutoBackend(model_path, self.device, data=data, fp16=fp16, fuse=False)
        if model_path.endswith(".pt") and hasattr(self.model.model, "fuse"):
            self.model.model.fuse()
        self.model.eval()

        self._resolve_image_size(image_size)

        # class map < id: class name >
        self.names = self.model.names

    def _resolve_image_size(self, image_size):
        """Set self.image_size from the caller's request, reporting where the model contradicts it."""
        trained = self._infer_image_size()
        long_side = self._infer_long_side()
        if image_size is not None:
            self.image_size = [int(image_size[0]), int(image_size[1])]
            if trained is not None and self.image_size != trained:
                self.logger.warning(
                    f"Provided image_size {self.image_size} != model's trained imgsz {trained}; "
                    "inference may be less accurate and differ from how the model was trained."
                )
            elif long_side is not None and max(self.image_size) != long_side:
                self.logger.warning(
                    f"Provided image_size {self.image_size} does not have the long side {long_side} this model "
                    "was trained at with rectangular batches; objects will be scaled differently than in training."
                )
        elif trained is not None:
            self.image_size = trained
            self.logger.info(f"image_size not specified; using model's trained imgsz {trained}")
        else:
            self.image_size = [640, 640]
            self.logger.warning("image_size not specified and trained imgsz unavailable; using [640, 640]")

    def _training_arg(self, name):
        args = getattr(getattr(self.model, "model", None), "args", None)
        return args.get(name) if isinstance(args, dict) else getattr(args, name, None)

    def _infer_image_size(self):
        """Return the model's input size as [h, w], or None when the model cannot express one.

        An exported model records the real pair in its metadata. A .pt has only its training args, where a
        rectangular run stores just the long side -- a scalar that is not a shape and must not be read as a
        square.
        """
        metadata = getattr(self.model, "metadata", None)
        imgsz = metadata.get("imgsz") if isinstance(metadata, dict) else None
        if isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2:
            return [int(imgsz[0]), int(imgsz[1])]

        imgsz = self._training_arg("imgsz")
        if isinstance(imgsz, (list, tuple)):
            return [int(imgsz[0]), int(imgsz[1])] if len(imgsz) >= 2 else [int(imgsz[0]), int(imgsz[0])]
        if isinstance(imgsz, int):
            return None if self._training_arg("rect") else [imgsz, imgsz]
        return None

    def _infer_long_side(self):
        """Return the long side a rectangular .pt was trained at -- the only size such a model records."""
        imgsz = self._training_arg("imgsz")
        return int(imgsz) if isinstance(imgsz, int) and self._training_arg("rect") else None

    @smart_inference_mode()
    def from_numpy(self, x: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def _setup_device(self, device):
        """set up the computation device (CPU or GPU).

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

    @smart_inference_mode()
    def forward(self, im: torch.Tensor):
        return self.model(im)

    @smart_inference_mode()
    def warmup(self, imgsz=None):
        """
        Warm up the model by running one forward pass with a dummy input.
        Args:
            imgsz(list): list of [h,w], default to None
        Returns:
            (None): This method runs the forward pass and don't return any value
        """
        if imgsz is None:
            imgsz = self.image_size

        if isinstance(imgsz, tuple):
            imgsz = list(imgsz)

        imgsz = [1, 3] + imgsz
        im = torch.empty(*imgsz, dtype=torch.half if self.model.fp16 else torch.float, device=self.device)
        self.forward(im)
