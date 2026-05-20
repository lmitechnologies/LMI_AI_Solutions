from typing import Any, Dict, List, Optional, Tuple

import torch

from lmi_utils.gadget_utils.pipeline_utils import fit_im_to_size, resize_image

from .._coords import apply_coord_transform
from ..operation import Operation


class ResizeOperation(Operation):
    """Resize each image to a target size; emit one ``resize`` history entry per image.

    Configuration:
        width (int, optional): target width. Defaults to current width.
        height (int, optional): target height. Defaults to current height.
        preserve_aspect (bool, optional): if True, scale-to-fit preserving aspect ratio
            then pad to target size (letterbox). Default False (free stretch to target).
        mode (str, optional): interpolation mode passed to ``resize_image``. Default 'bilinear'.

    Metadata schema (per image)::

        {"src_size": [w0, h0], "dst_size": [w, h], "pad"?: [L, R, T, B]}

    ``dst_size`` is the size after scaling but before padding. ``pad`` is omitted when
    no padding was applied (preserve_aspect=False, or aspect already matched).
    """

    name = "resize"

    @classmethod
    def build_step(
        cls,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        preserve_aspect: bool = False,
        mode: str = "bilinear",
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"preserve_aspect": preserve_aspect, "mode": mode}
        if width is not None:
            cfg["width"] = width
        if height is not None:
            cfg["height"] = height
        return cls._finalize_step(cfg, id=id)

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        width = config.get("width")
        height = config.get("height")
        preserve_aspect = config.get("preserve_aspect", False)
        mode = config.get("mode", "bilinear")

        out_images: List[torch.Tensor] = []
        meta_list: List[Dict[str, Any]] = []
        for img in images:
            h0, w0 = img.shape[:2]
            tw = width if width is not None else w0
            th = height if height is not None else h0

            if tw == w0 and th == h0:
                out_images.append(img)
                meta_list.append({"src_size": [w0, h0], "dst_size": [w0, h0]})
                continue

            if preserve_aspect:
                scale = min(th / h0, tw / w0)
                w1 = int(scale * w0)
                h1 = int(scale * h0)
                scaled = resize_image(img, W=w1, H=h1, mode=mode)
                entry: Dict[str, Any] = {"src_size": [w0, h0], "dst_size": [w1, h1]}
                if w1 != tw or h1 != th:
                    padded, pad_L, pad_R, pad_T, pad_B = fit_im_to_size(scaled, W=tw, H=th)
                    entry["pad"] = [pad_L, pad_R, pad_T, pad_B]
                    out_images.append(padded)
                else:
                    out_images.append(scaled)
                meta_list.append(entry)
            else:
                scaled = resize_image(img, W=tw, H=th, mode=mode)
                out_images.append(scaled)
                meta_list.append({"src_size": [w0, h0], "dst_size": [tw, th]})

        return out_images, meta_list

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> List[torch.Tensor]:
        _check_len(images, metadata, "resize")
        return [self._revert_image_single(img, m) for img, m in zip(images, metadata)]

    @staticmethod
    def _revert_image_single(img: torch.Tensor, m: Dict[str, Any]) -> torch.Tensor:
        pad = m.get("pad")
        if pad is not None:
            pL, pR, pT, pB = pad
            from lmi_utils.gadget_utils.pipeline_utils import fit_im

            img = fit_im(img, [-pL, -pR, -pT, -pB])
        src_w, src_h = m["src_size"]
        dst_w, dst_h = m["dst_size"]
        if (src_w, src_h) == (dst_w, dst_h):
            return img
        return resize_image(img, W=src_w, H=src_h)

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        _check_len(results, metadata, "resize")
        return [_apply_resize(r, m, forward=False) for r, m in zip(results, metadata)]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        _check_len(results, metadata, "resize")
        return [_apply_resize(r, m, forward=True) for r, m in zip(results, metadata)]


def _check_len(items, metadata, name: str) -> None:
    if len(items) != len(metadata):
        raise ValueError(f"{name}: input length ({len(items)}) != metadata length ({len(metadata)})")


def _apply_resize(result: Dict[str, Any], m: Dict[str, Any], *, forward: bool) -> Dict[str, Any]:
    """Apply or revert a resize+optional-pad metadata to one image's coords.

    forward=True: original-space -> preprocessed-space (scale, then add pad offset).
    forward=False: preprocessed-space -> original-space (subtract pad offset, then unscale).
    """
    src_w, src_h = m["src_size"]
    dst_w, dst_h = m["dst_size"]
    pad = m.get("pad", [0, 0, 0, 0])
    pad_L, _pad_R, pad_T, _pad_B = pad

    sx = dst_w / src_w
    sy = dst_h / src_h

    def xy_fn(xy: torch.Tensor) -> torch.Tensor:
        device = xy.device
        scale = torch.tensor([sx, sy], dtype=torch.float32, device=device)
        offset = torch.tensor([pad_L, pad_T], dtype=torch.float32, device=device)
        xy = xy.float()
        if forward:
            return xy * scale + offset
        return (xy - offset) / scale

    def mask_fn(masks: torch.Tensor) -> torch.Tensor:
        # masks live in preprocessed space. Forward and revert are symmetric image-resamples.
        # We don't implement mask resampling here because callers go through revert_images
        # for whole-image masks; instance masks reverted alongside boxes use this path.
        return _resample_masks(masks, [src_w, src_h], [dst_w, dst_h], pad_L, pad_T, pad_after=forward)

    return apply_coord_transform(result, xy_fn=xy_fn, mask_fn=mask_fn)


def _resample_masks(
    masks: torch.Tensor, src_size: List[int], dst_size: List[int], pad_L: int, pad_T: int, *, pad_after: bool
) -> torch.Tensor:
    """Resample instance masks between original space (src) and preprocessed space (dst+pad).

    pad_after=True: src -> dst -> pad (forward).
    pad_after=False: strip pad -> dst -> src (revert).
    """
    import torch.nn.functional as F

    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if pad_after:
        # forward: resize to (dst_h, dst_w), then pad to (dst_h + pT+pB, dst_w + pL+pR)
        resized = F.interpolate(masks.float().unsqueeze(1), size=(dst_h, dst_w), mode="nearest").squeeze(1)
        return resized  # Note: padding masks isn't strictly needed for coord-only forward;
        # callers don't typically forward-apply masks. Kept minimal here.
    # revert: strip pad first (if any), then resize back to (src_h, src_w)
    if pad_L or pad_T:
        # mask shape: (N, H, W). Strip pad from a (dst_h+pT+pB, dst_w+pL+pR) canvas.
        n, h, w = masks.shape
        # h == dst_h + pT + pB; w == dst_w + pL + pR
        masks = masks[:, pad_T : pad_T + dst_h, pad_L : pad_L + dst_w]
    return F.interpolate(masks.float().unsqueeze(1), size=(src_h, src_w), mode="nearest").squeeze(1)
