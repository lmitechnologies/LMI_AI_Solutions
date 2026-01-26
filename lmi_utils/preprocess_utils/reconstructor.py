from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch
from gadget_utils.pipeline_utils import revert_mask_to_origin
from image_utils.tiler import Tiler


class Reconstructor:
    def __init__(self):
        self._undo_handlers = {}
        self.register_default_undo_handlers()

    def register_default_undo_handlers(self):
        """Registers built-in undo handlers."""
        self.register_undo_handler("tile", self._undo_tile)
        self.register_undo_handler("resize", self._undo_resize)

    def register_undo_handler(self, name: str, undo_func: Callable) -> None:
        """Register an undo handler with validation"""
        if not callable(undo_func):
            raise TypeError(f"Undo handler for '{name}' must be callable.")
        self._undo_handlers[name] = undo_func

    def reconstruct(
        self, processed_images: List[Union[torch.Tensor, np.ndarray]], history: List[Dict[str, Any]]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        reconstructs the original image from processed images and metadata.

        Args:
            processed_images: List of (H, W, C) tensors or numpy arrays.
            history: List of metadata dicts.

        Returns:
            torch.Tensor | np.ndarray: The reconstructed image (H, W, C).
        """
        if not processed_images:
            raise ValueError("No input images provided for reconstruction.")

        if not isinstance(processed_images, list):
            raise TypeError("input images must be a list.")

        first_type = type(processed_images[0])
        if first_type not in (torch.Tensor, np.ndarray):
            raise TypeError("Images must be torch.Tensors or np.ndarrays")
        if not all(isinstance(img, first_type) for img in processed_images):
            raise TypeError("All images must be the same type")

        # convert to tensors if needed
        is_numpy = isinstance(processed_images[0], np.ndarray)
        current_images = processed_images
        if is_numpy:
            current_images = [torch.from_numpy(img) for img in processed_images]

        # Iterate BACKWARDS through history
        required_keys = {"op", "metadata"}
        for step in reversed(history):
            if not required_keys.issubset(step.keys()):
                raise ValueError(f"Each operation step must contain keys: {required_keys}")

            op_name = step["op"]
            meta = step["metadata"]

            if op_name in self._undo_handlers:
                undo_func = self._undo_handlers[op_name]
                current_images = undo_func(current_images, meta)
            else:
                raise ValueError(f"No undo handler for {op_name}")

        if len(current_images) != 1:
            raise RuntimeError(f"Reconstruction expected returning 1 image, got {len(current_images)}")

        # Return the single root image
        if is_numpy:
            return current_images[0].cpu().numpy()
        return current_images[0]

    def _undo_tile(self, images: List[torch.Tensor], meta: Dict[str, Any]) -> List[torch.Tensor]:
        """
        undoes the 'tile' operation.
        Args:
            images (list[torch.Tensor]): List of input tiles (H, W, C).
            meta (dict): Metadata containing 'tiler_metadata'.
        """
        if not images:
            raise ValueError("No input images provided for untile operation.")

        if "tiler_metadata" not in meta:
            raise KeyError(f"Metadata missing required key 'tiler_metadata'. Got keys: {list(meta.keys())}")

        tiler_meta_list = meta["tiler_metadata"]
        restored_images = []
        cursor = 0

        for tiler_meta in tiler_meta_list:
            tiler = Tiler.from_dict(tiler_meta)
            count = tiler_meta["n_tiles"][0] * tiler_meta["n_tiles"][1]

            # Slice the batch
            batch_slice_hwc = images[cursor : cursor + count]
            cursor += count

            # Integrity Check
            if len(batch_slice_hwc) != count:
                raise RuntimeError(f"Expected {count} tiles, found {len(batch_slice_hwc)}")

            # Prepare for Untile
            batch_hwc = torch.stack(batch_slice_hwc)  # Stack -> [N, H, W, C]
            batch_chw = batch_hwc.permute(0, 3, 1, 2)  # Permute -> [N, C, H, W]

            # Untile the batch -> [1, C, H, W]
            scale_mode = tiler_meta.get("scale_mode", "padding")
            overlap_mode = tiler_meta.get("overlap_mode", "average")
            restored_batch = tiler.untile(batch_chw, scale_mode=scale_mode, overlap_mode=overlap_mode)

            # Convert back to (H, W, C)
            restored_img = restored_batch.squeeze(0).permute(1, 2, 0)
            restored_images.append(restored_img)

        if cursor != len(images):
            raise RuntimeError(f"Tile reconstruction mismatch: processed {cursor} images, but received {len(images)}")

        return restored_images

    def _undo_resize(self, images: List[torch.Tensor], meta: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Reverses the composite 'resize_and_pad' operation.
        Args:
            images (list[torch.Tensor]): List of input images (H, W, C).
            meta (dict): Metadata containing 'ops' for each image.
        """
        if not images:
            raise ValueError("No input images provided for undo resize operation.")

        if "ops" not in meta:
            raise KeyError("Metadata missing required key 'ops'")

        image_ops_list = meta["ops"]
        if len(images) != len(image_ops_list):
            raise ValueError(f"Image count ({len(images)}) doesn't match ops count ({len(image_ops_list)})")

        output_images = [revert_mask_to_origin(image, ops) for image, ops in zip(images, image_ops_list)]
        return output_images
