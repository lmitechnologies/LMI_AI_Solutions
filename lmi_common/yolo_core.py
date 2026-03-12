import logging
import os

import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import smart_inference_mode


class YoloCore:
    logger = logging.getLogger("yolo-core")
    task = ""

    def __init__(self, model_path: str, device="gpu", data=None, fp16=False, **kwargs) -> None:
        """init the model
        Args:
            model_path (str): the path to the model_path file.
            device (str, optional): the device to be used, either 'gpu' or 'cpu'. Defaults to 'gpu'.
            data (str, optional): the path to dataset yaml file. Defaults to None.
            fp16 (bool, optional): Whether to use fp16. Defaults to False.
        Raises:
            FileNotFoundError: the model_path file does not exist
        """
        image_size = kwargs.get("image_size", None)
        if image_size is None:
            image_size = [640, 640]
            self.logger.warning("image_size not specified, using default value of [640, 640]")
        self.image_size = image_size

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"File not found: {model_path}")

        self._setup_device(device)

        # load model
        self.model = AutoBackend(model_path, self.device, data=data, fp16=fp16, fuse=False)
        if model_path.endswith(".pt") and hasattr(self.model.model, "fuse"):
            self.model.model.fuse()
        self.model.eval()

        # class map < id: class name >
        self.names = self.model.names

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
            device (str): The device to be used, either 'cpu' or 'gpu'.
        """
        if device.lower() not in ["cpu", "gpu"]:
            raise ValueError(f'Invalid device: {device}. Supported devices are "cpu" and "gpu".')

        self.device = torch.device("cpu")
        if device.lower() in ["gpu", "cuda"]:
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
        im = torch.empty(
            *imgsz,
            dtype=torch.half if self.model.fp16 else torch.float,
            device=self.device,
        )
        self.forward(im)
