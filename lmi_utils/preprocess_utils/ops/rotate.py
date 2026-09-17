import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .._coords import apply_coord_transform
from ..operation import Config, Meta, Operation

Affine = Tuple[float, float, float, float, float, float]


@dataclass
class RotateConfig(Config):
    """Rotate each image by ``angle`` degrees (positive = clockwise, image y-down).

    The canvas expands to fit, so no content is clipped. Multiples of 90 are exact and lossless.
    Points, segments, masks and OBB boxes round-trip; axis-aligned xyxy boxes grow at other
    angles, because the rotated box is re-fit to an axis-aligned one in each direction.
    """

    angle: float = 0.0


@dataclass
class RotateMeta(Meta):
    """Batched rotate metadata.

    angles: per-image rotation in degrees.
    src_sizes: [W, H] before rotation.
    dst_sizes: [new_W, new_H] after rotation (canvas expanded to fit).
    """

    angles: List[float] = field(default_factory=list)
    src_sizes: List[List[int]] = field(default_factory=list)
    dst_sizes: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.angles)
        if not (len(self.src_sizes) == n and len(self.dst_sizes) == n):
            raise ValueError("RotateMeta: field lengths must match")


class RotateOperation(Operation[RotateConfig, RotateMeta]):
    config_cls = RotateConfig
    meta_cls = RotateMeta

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: RotateConfig) -> Tuple[List[torch.Tensor], RotateMeta]:
        angle_deg = float(config.angle)
        src_sizes = [[img.shape[1], img.shape[0]] for img in images]
        dst_sizes = [list(_expand_size(W, H, angle_deg)) for W, H in src_sizes]
        out_images = [
            _rotate_image(img, W=W, H=H, nW=nW, nH=nH, angle_deg=angle_deg, forward=True)
            for img, (W, H), (nW, nH) in zip(images, src_sizes, dst_sizes)
        ]
        return out_images, RotateMeta(angles=[angle_deg] * len(images), src_sizes=src_sizes, dst_sizes=dst_sizes)

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], meta: RotateMeta) -> List[torch.Tensor]:
        _check_count(len(images), meta, "image")
        return [
            _rotate_image(img, W=src[0], H=src[1], nW=dst[0], nH=dst[1], angle_deg=angle, forward=False)
            for img, angle, src, dst in zip(images, meta.angles, meta.src_sizes, meta.dst_sizes)
        ]

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], meta: RotateMeta) -> List[Dict[str, Any]]:
        _check_count(len(results), meta, "result")
        return [
            _apply_rotate(r, angle, src, dst, forward=False)
            for r, angle, src, dst in zip(results, meta.angles, meta.src_sizes, meta.dst_sizes)
        ]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], meta: RotateMeta) -> List[Dict[str, Any]]:
        _check_count(len(results), meta, "result")
        return [
            _apply_rotate(r, angle, src, dst, forward=True)
            for r, angle, src, dst in zip(results, meta.angles, meta.src_sizes, meta.dst_sizes)
        ]


def _check_count(n: int, meta: RotateMeta, what: str) -> None:
    if n != len(meta.angles):
        raise ValueError(f"rotate: {what} count ({n}) != meta count ({len(meta.angles)})")


def _expand_size(W: int, H: int, angle_deg: float) -> Tuple[int, int]:
    """Bounding-box size of the rotated rectangle, rounded up so no content is clipped."""
    theta = math.radians(angle_deg)
    abs_cos = abs(math.cos(theta))
    abs_sin = abs(math.sin(theta))
    # The eps absorbs float error at multiples of 90, where a zero term computes as ~1e-15 and ceil would add a pixel.
    new_W = math.ceil(H * abs_sin + W * abs_cos - 1e-6)
    new_H = math.ceil(H * abs_cos + W * abs_sin - 1e-6)
    return new_W, new_H


def rotate_transform(width: int, height: int, angle_deg: float) -> Tuple[Affine, int, int]:
    """Affine taking src pixel coords onto the expanded canvas, plus that canvas size.

    Args:
        width: source image width.
        height: source image height.
        angle_deg: rotation in degrees, positive = clockwise (image y-down).

    Returns:
        ``((a, b, tx, c, d, ty), new_width, new_height)`` — the affine in row-major order.
    """
    new_W, new_H = _expand_size(width, height, angle_deg)
    return _affine(width, height, new_W, new_H, angle_deg), new_W, new_H


def _rot90_k(angle_deg: float) -> Optional[int]:
    """``k`` for ``torch.rot90(dims=(0, 1))`` when the angle is a multiple of 90, else None."""
    if angle_deg % 90 != 0:
        return None
    return -int(angle_deg // 90) % 4  # rot90 is counter-clockwise; our angle is clockwise


def _affine(W: int, H: int, new_W: int, new_H: int, angle_deg: float) -> Affine:
    """Map pixel coords in a W x H canvas onto a new_W x new_H one: rotate about the center, re-center."""
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    a, b = cos_t, -sin_t
    c, d = sin_t, cos_t
    # Centers are (S-1)/2, not S/2: pixel centers sit at integer coords, so S/2 skews the result by up to a pixel.
    tx = -a * (W - 1) / 2 - b * (H - 1) / 2 + (new_W - 1) / 2
    ty = -c * (W - 1) / 2 - d * (H - 1) / 2 + (new_H - 1) / 2
    return (a, b, tx, c, d, ty)


def _rotate_image(img: torch.Tensor, *, W: int, H: int, nW: int, nH: int, angle_deg: float, forward: bool) -> torch.Tensor:
    """Rotate one HW/HWC image src->dst (``forward``) or dst->src. W,H is src size; nW,nH is dst size."""
    k = _rot90_k(angle_deg)
    if k is not None:  # exact and allocation-free; also makes angle 0 a true no-op
        return torch.rot90(img, k if forward else -k, dims=(0, 1)).contiguous()
    # grid_sample pulls, so it needs the map running opposite to the direction being travelled.
    if forward:
        return _warp(img, in_W=W, in_H=H, out_W=nW, out_H=nH, sample_M=_affine(nW, nH, W, H, -angle_deg))
    return _warp(img, in_W=nW, in_H=nH, out_W=W, out_H=H, sample_M=_affine(W, H, nW, nH, angle_deg))


def _sample_grid(*, in_W: int, in_H: int, out_W: int, out_H: int, sample_M: Affine, device: torch.device) -> torch.Tensor:
    """(1, out_H, out_W, 2) grid_sample grid where output[i, j] samples the input at sample_M @ [j, i, 1]."""
    a, b, tx, c, d, ty = sample_M
    # affine_grid works in normalized coords, where align_corners=False puts pixel p of a size-S axis at
    # 2p/S + 1/S - 1. Rewriting sample_M under that substitution gives theta; affine_grid then builds the
    # grid in one kernel instead of materializing the x/y pixel ramps separately.
    kx = a * (out_W / 2 - 0.5) + b * (out_H / 2 - 0.5) + tx
    ky = c * (out_W / 2 - 0.5) + d * (out_H / 2 - 0.5) + ty
    theta = torch.tensor(
        [
            [a * out_W / in_W, b * out_H / in_W, (2 * kx + 1) / in_W - 1],
            [c * out_W / in_H, d * out_H / in_H, (2 * ky + 1) / in_H - 1],
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    return F.affine_grid(theta, (1, 1, out_H, out_W), align_corners=False)


def _warp(img: torch.Tensor, *, in_W: int, in_H: int, out_W: int, out_H: int, sample_M: Affine) -> torch.Tensor:
    """Bilinear warp of an HW or HWC image. ``out[i, j] = img[sample_M @ [j, i, 1]]``, zero outside."""
    grid = _sample_grid(in_W=in_W, in_H=in_H, out_W=out_W, out_H=out_H, sample_M=sample_M, device=img.device)
    orig_dtype = img.dtype
    round_int = not orig_dtype.is_floating_point  # truncating the float result biases integer pixels down
    hw_only = img.dim() == 2
    if hw_only:
        inp = img.unsqueeze(0).unsqueeze(0).float()
    else:
        # Channels ride the batch dim: grid_sample's CPU kernel threads over batch only, so (1, C, H, W) runs serial.
        inp = img.permute(2, 0, 1).unsqueeze(1).float()
        grid = grid.expand(inp.shape[0], out_H, out_W, 2)
    out = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = out[0, 0] if hw_only else out.squeeze(1).permute(1, 2, 0)
    return (out.round_() if round_int else out).to(orig_dtype)  # out is grid_sample's fresh buffer, safe to round in place


def _apply_rotate(result: Dict[str, Any], angle: float, src: List[int], dst: List[int], *, forward: bool) -> Dict[str, Any]:
    W, H = src
    nW, nH = dst
    # Coords travel with the requested direction; the mask warp pulls, so it samples with the opposite map.
    if forward:
        travel, pull = _affine(W, H, nW, nH, angle), _affine(nW, nH, W, H, -angle)
        in_W, in_H, out_W, out_H = W, H, nW, nH
    else:
        travel, pull = _affine(nW, nH, W, H, -angle), _affine(W, H, nW, nH, angle)
        in_W, in_H, out_W, out_H = nW, nH, W, H

    def xy_fn(xy: torch.Tensor) -> torch.Tensor:
        a, b, tx, c, d, ty = travel
        x, y = xy[..., 0].float(), xy[..., 1].float()
        return torch.stack([a * x + b * y + tx, c * x + d * y + ty], dim=-1)

    def box_fn(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.ndim == 3:  # OBB (N, 4, 2)
            return xy_fn(boxes.reshape(-1, 2)).reshape(boxes.shape)
        # xyxy: rotate the 4 corners, then re-fit an axis-aligned box around them.
        corners = xy_fn(boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)).reshape(-1, 4, 2)
        return torch.cat([corners.amin(dim=1), corners.amax(dim=1)], dim=-1)

    def mask_fn(masks: torch.Tensor) -> torch.Tensor:
        k = _rot90_k(angle)
        if k is not None:
            return torch.rot90(masks, k if forward else -k, dims=(1, 2)).contiguous()
        grid = _sample_grid(in_W=in_W, in_H=in_H, out_W=out_W, out_H=out_H, sample_M=pull, device=masks.device)
        inp = masks.float().unsqueeze(1)  # (N, 1, in_H, in_W)
        grid = grid.expand(masks.shape[0], out_H, out_W, 2)
        out = F.grid_sample(inp, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        return out.squeeze(1).to(masks.dtype)

    return apply_coord_transform(result, xy_fn=xy_fn, box_fn=box_fn, mask_fn=mask_fn)
