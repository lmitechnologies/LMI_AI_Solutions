import numpy as np
import torch
from gadget_utils.pipeline_utils import revert_mask_to_origin


class Reconstructor:
    def __init__(self):
        self._undo_handlers = {"tile": self._undo_tile, "resize": self._undo_resize}

    def reconstruct(self, final_images: list, ops: list) -> torch.Tensor:
        """
        Reverses the pipeline.

        Args:
            final_images: List of (H, W, C) tensors or numpy arrays.
            ops: List of metadata dicts.

        Returns:
            torch.Tensor | np.ndarray: The reconstructed image (H, W, C).
        """
        if not len(final_images):
            return

        is_numpy = any(isinstance(img, np.ndarray) for img in final_images)

        current_images = final_images
        if is_numpy:
            current_images = [torch.from_numpy(img) if isinstance(img, np.ndarray) else img for img in final_images]

        # Iterate BACKWARDS through ops
        for step in reversed(ops):
            op_name = step["op"]
            meta = step["metadata"]

            if op_name in self._undo_handlers:
                undo_func = self._undo_handlers[op_name]
                current_images = undo_func(current_images, meta)
            else:
                raise ValueError(f"No undo handler for {op_name}")

        # Return the single root image
        if is_numpy:
            return current_images[0].cpu().numpy()
        return current_images[0]

    def _undo_tile(self, images, meta):
        """
        Input: List of (H, W, C) tiles.
        Output: List of (H, W, C) restored parents.
        """
        tiler_instances = meta["tiler_instances"]

        restored_images = []
        cursor = 0

        for tiler in tiler_instances:
            # 1. Derive count from Tiler state
            # n_tiles is likely [rows, cols]
            count = tiler.n_tiles[0] * tiler.n_tiles[1]

            # 2. Slice the batch
            batch_slice_hwc = images[cursor : cursor + count]
            cursor += count

            # Integrity Check
            if len(batch_slice_hwc) != count:
                raise RuntimeError(f"Expected {count} tiles, found {len(batch_slice_hwc)}")

            # 3. Prepare for Untile (Stack & Permute)
            # Stack -> [N, H, W, C]
            batch_hwc = torch.stack(batch_slice_hwc)
            # Permute -> [N, C, H, W]
            batch_chw = batch_hwc.permute(0, 3, 1, 2)

            # 4. Untile using the specific instance
            # Returns [1, C, H, W]
            restored_batch = tiler.untile(batch_chw)

            # 5. Convert back to (H, W, C)
            restored_img = restored_batch.squeeze(0).permute(1, 2, 0)
            restored_images.append(restored_img)

        return restored_images

    def _undo_resize(self, images, meta):
        """
        Reverses the composite 'resize_and_pad' operation.
        """
        image_ops_list = meta["ops"]
        output_images = []
        for image, ops in zip(images, image_ops_list):
            image = revert_mask_to_origin(image, ops)
            output_images.append(image)
        return output_images
