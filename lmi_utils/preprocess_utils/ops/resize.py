from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from lmi_utils.gadget_utils.pipeline_utils import fit_im_to_size, resize_image

from .._coords import apply_coord_transform
from ..operation import Config, Meta, Operation


@dataclass
class ResizeConfig(Config):
    """Resize each image to a target size.

    width: target width. Defaults to current width.
    height: target height. Defaults to current height.
    preserve_aspect: scale-to-fit preserving aspect ratio then pad (letterbox).
    pad_value: fill value for letterbox padding (only used when preserve_aspect). 0 is black; use 114 to match YOLO.
    mode: interpolation mode passed to ``resize_image``.
    """

    width: Optional[int] = None
    height: Optional[int] = None
    preserve_aspect: bool = False
    pad_value: int = 0
    mode: str = "bilinear"


@dataclass
class ResizeMeta(Meta):
    """Batched resize metadata.

    src_sizes / dst_sizes: [W, H] per image; dst_size is the post-scale, pre-pad size.
    pads: per-image [L, R, T, B] (zeros when no padding was applied).
    """

    src_sizes: List[List[int]] = field(default_factory=list)
    dst_sizes: List[List[int]] = field(default_factory=list)
    pads: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.src_sizes)
        if not (len(self.dst_sizes) == n and len(self.pads) == n):
            raise ValueError(
                f"ResizeMeta: field lengths must match — "
                f"src_sizes={len(self.src_sizes)} dst_sizes={len(self.dst_sizes)} pads={len(self.pads)}"
            )


class ResizeOperation(Operation[ResizeConfig, ResizeMeta]):
    config_cls = ResizeConfig
    meta_cls = ResizeMeta

    def __init__(self, image_mode: str = "bilinear"):
        """image_mode: ``revert_images`` interpolation. ``"nearest"`` for binary/label masks."""
        self.image_mode = image_mode

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: ResizeConfig) -> Tuple[List[torch.Tensor], ResizeMeta]:
        out_images: List[torch.Tensor] = []
        src_sizes: List[List[int]] = []
        dst_sizes: List[List[int]] = []
        pads: List[List[int]] = []

        for img in images:
            h0, w0 = img.shape[:2]
            tw = config.width if config.width is not None else w0
            th = config.height if config.height is not None else h0

            if tw == w0 and th == h0:
                out_images.append(img)
                src_sizes.append([w0, h0])
                dst_sizes.append([w0, h0])
                pads.append([0, 0, 0, 0])
                continue

            if config.preserve_aspect:
                scale = min(th / h0, tw / w0)
                w1 = int(scale * w0)
                h1 = int(scale * h0)
                scaled = resize_image(img, W=w1, H=h1, mode=config.mode)
                src_sizes.append([w0, h0])
                dst_sizes.append([w1, h1])
                if w1 != tw or h1 != th:
                    padded, pL, pR, pT, pB = fit_im_to_size(scaled, W=tw, H=th, value=config.pad_value)
                    pads.append([pL, pR, pT, pB])
                    out_images.append(padded)
                else:
                    pads.append([0, 0, 0, 0])
                    out_images.append(scaled)
            else:
                scaled = resize_image(img, W=tw, H=th, mode=config.mode)
                out_images.append(scaled)
                src_sizes.append([w0, h0])
                dst_sizes.append([tw, th])
                pads.append([0, 0, 0, 0])

        return out_images, ResizeMeta(src_sizes=src_sizes, dst_sizes=dst_sizes, pads=pads)

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], meta: ResizeMeta) -> List[torch.Tensor]:
        _check_len(images, meta.src_sizes, "resize")
        return [
            _revert_image_single(img, src, dst, pad, mode=self.image_mode)
            for img, src, dst, pad in zip(images, meta.src_sizes, meta.dst_sizes, meta.pads)
        ]

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], meta: ResizeMeta) -> List[Dict[str, Any]]:
        _check_len(results, meta.src_sizes, "resize")
        return [
            _apply_resize(r, src, dst, pad, forward=False) for r, src, dst, pad in zip(results, meta.src_sizes, meta.dst_sizes, meta.pads)
        ]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], meta: ResizeMeta) -> List[Dict[str, Any]]:
        _check_len(results, meta.src_sizes, "resize")
        return [
            _apply_resize(r, src, dst, pad, forward=True) for r, src, dst, pad in zip(results, meta.src_sizes, meta.dst_sizes, meta.pads)
        ]


def _check_len(items, sizes, name: str) -> None:
    if len(items) != len(sizes):
        raise ValueError(f"{name}: input length ({len(items)}) != metadata length ({len(sizes)})")


def _revert_image_single(img: torch.Tensor, src: List[int], dst: List[int], pad: List[int], mode: str = "bilinear") -> torch.Tensor:
    pL, pR, pT, pB = pad
    if pL or pR or pT or pB:
        from lmi_utils.gadget_utils.pipeline_utils import fit_im

        img = fit_im(img, [-pL, -pR, -pT, -pB])
    src_w, src_h = src
    dst_w, dst_h = dst
    if (src_w, src_h) == (dst_w, dst_h):
        return img
    return resize_image(img, W=src_w, H=src_h, mode=mode)


def _apply_resize(result: Dict[str, Any], src: List[int], dst: List[int], pad: List[int], *, forward: bool) -> Dict[str, Any]:
    src_w, src_h = src
    dst_w, dst_h = dst
    pad_L, pad_R, pad_T, pad_B = pad

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
        return _resample_masks(masks, [src_w, src_h], [dst_w, dst_h], [pad_L, pad_R, pad_T, pad_B], pad_after=forward)

    return apply_coord_transform(result, xy_fn=xy_fn, mask_fn=mask_fn)


def _resample_masks(masks: torch.Tensor, src_size: List[int], dst_size: List[int], pad: List[int], *, pad_after: bool) -> torch.Tensor:
    """Resample instance masks between original space (src) and preprocessed space (dst+pad).

    pad_after=True: src -> dst -> pad (forward).
    pad_after=False: strip pad -> dst -> src (revert).
    """
    import torch.nn.functional as F

    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    pad_L, pad_R, pad_T, pad_B = pad
    if pad_after:
        # forward: resize to (dst_h, dst_w), then pad to (dst_h + pT+pB, dst_w + pL+pR)
        resized = F.interpolate(masks.float().unsqueeze(1), size=(dst_h, dst_w), mode="nearest").squeeze(1)
        if pad_L or pad_R or pad_T or pad_B:
            n = resized.shape[0]
            canvas_h = dst_h + pad_T + pad_B
            canvas_w = dst_w + pad_L + pad_R
            canvas = torch.zeros((n, canvas_h, canvas_w), dtype=resized.dtype, device=resized.device)
            canvas[:, pad_T : pad_T + dst_h, pad_L : pad_L + dst_w] = resized
            return canvas
        return resized
    # revert: strip pad first (if any), then resize back to (src_h, src_w)
    if pad_L or pad_T:
        # mask shape: (N, H, W). Strip pad from a (dst_h+pT+pB, dst_w+pL+pR) canvas.
        masks = masks[:, pad_T : pad_T + dst_h, pad_L : pad_L + dst_w]
    return F.interpolate(masks.float().unsqueeze(1), size=(src_h, src_w), mode="nearest").squeeze(1)
