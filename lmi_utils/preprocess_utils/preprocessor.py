import numpy as np
import torch
from image_utils.img_resize import resize_and_pad
from image_utils.tiler import Tiler


class Preprocessor:
    """
    A class to run a dynamic pipeline of preprocessing steps on an image.

    Handlers (processing functions) are registered with the instance and
    called based on a list of processing steps.
    """

    def __init__(self):
        """
        Initializes the preprocessor and registers default handlers.
        """
        self._handlers = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Registers the built-in processing functions."""
        self.register_handler("resize", self._resize_wrapper)
        self.register_handler("tile", self._tile_wrapper)

    def register_handler(self, name: str, handler_func):
        """
        Registers a new handler function or overwrites an existing one.

        The handler_func must accept 'image' as its first argument,
        followed by keyword arguments matching its configuration.
        """
        if not callable(handler_func):
            raise TypeError(f"Handler for '{name}' must be a callable function.")
        self._handlers[name] = handler_func

    def preprocess(self, image, processing_steps):
        """
        Runs the preprocessing pipeline.

        Args:
            image (np.ndarray | torch.Tensor): Input image in format (H, W, C).
            processing_steps (list): List of config dictionaries.

        Returns:
            current_images (list[np.ndarray | torch.Tensor]): Processed images (H, W, C).
            ops (list[dict]): Metadata chain for reconstruction.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        is_numpy = isinstance(image, np.ndarray)
        current_images = [image]
        ops = []
        required_keys = {"type", "configuration"}
        for step in processing_steps:
            if not required_keys.issubset(step.keys()):
                raise ValueError(f"Each processing step must contain keys: {required_keys}")

            op_name = step["type"]
            config = step["configuration"]
            if op_name not in self._handlers:
                raise ValueError(f"Handler for '{op_name}' is not registered.")

            handler = self._handlers[op_name]
            parent_shapes = [img.shape[0:2] for img in current_images]
            new_images, metadata = handler(current_images, config)

            # Save Metadata
            step_record = {"op": op_name, "metadata": {"config": config, "parent_shapes": parent_shapes, **metadata}}
            ops.append(step_record)
            current_images = new_images

        # Convert back to numpy if needed
        if is_numpy:
            current_images = [img.cpu().numpy() if isinstance(img, torch.Tensor) else img for img in current_images]

        return current_images, ops

    def _tile_wrapper(self, images, config):
        """
        Wraps Tiler.
        Input: List of (H, W, C) numpy arrays or tensors.
        Output: List of (H, W, C) tiles.
        """
        output_images = []
        instances = []

        for img in images:
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)

            tiler = Tiler(**config)

            img_batch = img.permute(2, 0, 1).unsqueeze(0)  # Convert to [1, C, H, W]
            tiles_batch = tiler.tile(img_batch)  # Returns [N, C, H, W]

            # Convert back to List of (H, W, C)
            tiles_list_chw = list(torch.unbind(tiles_batch, dim=0))
            tiles_list_hwc = [t.permute(1, 2, 0) for t in tiles_list_chw]

            output_images.extend(tiles_list_hwc)
            instances.append(tiler)

        return output_images, {"tiler_instances": instances}

    def _resize_wrapper(self, images, config):
        """
        Wraps the user's custom 'resize_and_pad' function.

        Args:
            images (list[np.ndarray | torch.Tensor]): List of input images (H, W, C).
            config (dict): Configuration for resize_and_pad.
        """
        # Map config keys to function arguments
        dt = {
            "width": config.get("width"),
            "height": config.get("height"),
            "preserve_aspect": config.get("preserve_aspect", False),
            "mode": config.get("mode", "bilinear"),
        }

        output_images = []
        image_ops_list = []
        for img in images:
            # convert to tensor
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)

            processed, ops = resize_and_pad(
                img,
                return_operators=True,
                **dt,
            )

            output_images.append(processed)
            image_ops_list.append(ops)

        return output_images, {"ops": image_ops_list}
