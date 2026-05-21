import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .._coords import apply_coord_transform
from ..operation import Operation

# 2x3 affine in row-major order (a, b, tx, c, d, ty) — maps [x, y, 1] -> [a*x + b*y + tx, c*x + d*y + ty].
Affine = Tuple[float, float, float, float, float, float]


class RotateOperation(Operation):
    """Rotate each image by ``angle`` degrees, expanding the canvas to fit the rotated extent.

    Configuration:
        angle (float): rotation in degrees. Positive = clockwise (image y-down).

    Metadata schema (per image)::

        {"angle": float, "src_size": [W, H], "dst_size": [new_W, new_H]}

    Coordinate revert is lossless up to float precision. Image revert is **lossy**
    for non-90° angles: bilinear resampling softens edges on each pass and the
    expanded canvas's zero-fill borders bleed back into the rotated rectangle on revert.

    To reproduce the same rotation with OpenCV::

        import cv2
        W, H = m["src_size"]; nW, nH = m["dst_size"]
        M = cv2.getRotationMatrix2D((W // 2, H // 2), -m["angle"], 1.0)
        M[0, 2] += nW / 2 - W // 2
        M[1, 2] += nH / 2 - H // 2
        out = cv2.warpAffine(img, M, (nW, nH))
    """

    name = "rotate"

    @classmethod
    def build_step(cls, *, angle: float, id: Optional[str] = None) -> Dict[str, Any]:
        return cls._finalize_step({"angle": float(angle)}, id=id)

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        angle_deg = float(config["angle"])
        out_images: List[torch.Tensor] = []
        meta_list: List[Dict[str, Any]] = []
        for img in images:
            H, W = img.shape[:2]
            new_W, new_H = _expand_size(W, H, angle_deg)
            M = _forward_affine(W, H, new_W, new_H, angle_deg)
            # Sample original at M_inv(dst) for each rotated pixel.
            out_images.append(_warp(img, in_W=W, in_H=H, out_W=new_W, out_H=new_H, sample_M=_invert(M), mode="bilinear"))
            meta_list.append({"angle": angle_deg, "src_size": [W, H], "dst_size": [new_W, new_H]})
        return out_images, meta_list

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> List[torch.Tensor]:
        if len(images) != len(metadata):
            raise ValueError(f"rotate: image count ({len(images)}) != metadata count ({len(metadata)})")
        out: List[torch.Tensor] = []
        for img, m in zip(images, metadata):
            W, H = m["src_size"]
            nW, nH = m["dst_size"]
            M = _forward_affine(W, H, nW, nH, m["angle"])
            # Sample rotated at M(dst) for each original pixel.
            out.append(_warp(img, in_W=nW, in_H=nH, out_W=W, out_H=H, sample_M=M, mode="bilinear"))
        return out

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [_apply_rotate(r, m, forward=False) for r, m in zip(results, metadata)]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [_apply_rotate(r, m, forward=True) for r, m in zip(results, metadata)]


def _expand_size(W: int, H: int, angle_deg: float) -> Tuple[int, int]:
    theta = math.radians(angle_deg)
    abs_cos = abs(math.cos(theta))
    abs_sin = abs(math.sin(theta))
    new_W = int(H * abs_sin + W * abs_cos)
    new_H = int(H * abs_cos + W * abs_sin)
    return new_W, new_H


def _forward_affine(W: int, H: int, new_W: int, new_H: int, angle_deg: float) -> Affine:
    """Matches ``cv2.getRotationMatrix2D((W//2, H//2), -angle_deg, 1.0)`` + expand translation."""
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cx = W // 2
    cy = H // 2
    a, b = cos_t, -sin_t
    c, d = sin_t, cos_t
    tx = -a * cx - b * cy + new_W / 2.0
    ty = -c * cx - d * cy + new_H / 2.0
    return (a, b, tx, c, d, ty)


def _invert(M: Affine) -> Affine:
    a, b, tx, c, d, ty = M
    det = a * d - b * c
    ia = d / det
    ib = -b / det
    ic = -c / det
    id_ = a / det
    itx = -(ia * tx + ib * ty)
    ity = -(ic * tx + id_ * ty)
    return (ia, ib, itx, ic, id_, ity)


def _warp(img: torch.Tensor, *, in_W: int, in_H: int, out_W: int, out_H: int, sample_M: Affine, mode: str) -> torch.Tensor:
    """Warp ``img`` so output[i, j] samples img at sample_M @ [j, i, 1]."""
    a, b, tx, c, d, ty = sample_M
    device = img.device
    ys = torch.arange(out_H, device=device, dtype=torch.float32).view(-1, 1).expand(out_H, out_W)
    xs = torch.arange(out_W, device=device, dtype=torch.float32).view(1, -1).expand(out_H, out_W)
    sx = a * xs + b * ys + tx
    sy = c * xs + d * ys + ty
    # align_corners=False matches cv2.warpAffine's pixel-center-at-integer-index +
    # half-pixel-border convention. Pixel centers at integer coords; valid range is [-0.5, W-0.5].
    norm_x = (sx + 0.5) / in_W * 2.0 - 1.0
    norm_y = (sy + 0.5) / in_H * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, out_H, out_W, 2)

    orig_dtype = img.dtype
    if img.dim() == 2:
        inp = img.unsqueeze(0).unsqueeze(0).float()
        out = F.grid_sample(inp, grid, mode=mode, padding_mode="zeros", align_corners=False)
        return out[0, 0].to(orig_dtype)
    inp = img.permute(2, 0, 1).unsqueeze(0).float()
    out = F.grid_sample(inp, grid, mode=mode, padding_mode="zeros", align_corners=False)
    return out[0].permute(1, 2, 0).to(orig_dtype)


def _apply_affine_xy(M: Affine, xy: torch.Tensor) -> torch.Tensor:
    a, b, tx, c, d, ty = M
    xy = xy.float()
    x = xy[..., 0]
    y = xy[..., 1]
    return torch.stack([a * x + b * y + tx, c * x + d * y + ty], dim=-1)


def _apply_rotate(result: Dict[str, Any], m: Dict[str, Any], *, forward: bool) -> Dict[str, Any]:
    W, H = m["src_size"]
    nW, nH = m["dst_size"]
    M_fwd = _forward_affine(W, H, nW, nH, m["angle"])
    M_apply = M_fwd if forward else _invert(M_fwd)

    def xy_fn(xy: torch.Tensor) -> torch.Tensor:
        return _apply_affine_xy(M_apply, xy)

    def box_fn(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.ndim == 3:  # OBB (N, 4, 2)
            n = boxes.shape[0]
            return xy_fn(boxes.reshape(-1, 2)).reshape(n, 4, 2)
        # xyxy: rotate 4 corners, take axis-aligned bbox of the rotated corners.
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        corners = torch.stack(
            [
                torch.stack([x1, y1], dim=-1),
                torch.stack([x2, y1], dim=-1),
                torch.stack([x2, y2], dim=-1),
                torch.stack([x1, y2], dim=-1),
            ],
            dim=1,
        )  # (N, 4, 2)
        rotated = xy_fn(corners.reshape(-1, 2)).reshape(-1, 4, 2)
        nx1 = rotated[..., 0].amin(dim=1)
        ny1 = rotated[..., 1].amin(dim=1)
        nx2 = rotated[..., 0].amax(dim=1)
        ny2 = rotated[..., 1].amax(dim=1)
        return torch.stack([nx1, ny1, nx2, ny2], dim=-1)

    def mask_fn(masks: torch.Tensor) -> torch.Tensor:
        if forward:
            in_W, in_H, out_W, out_H = W, H, nW, nH
            sample_M = _invert(M_fwd)
        else:
            in_W, in_H, out_W, out_H = nW, nH, W, H
            sample_M = M_fwd
        a, b, tx, c, d, ty = sample_M
        device = masks.device
        ys = torch.arange(out_H, device=device, dtype=torch.float32).view(-1, 1).expand(out_H, out_W)
        xs = torch.arange(out_W, device=device, dtype=torch.float32).view(1, -1).expand(out_H, out_W)
        sx = a * xs + b * ys + tx
        sy = c * xs + d * ys + ty
        norm_x = (sx + 0.5) / in_W * 2.0 - 1.0
        norm_y = (sy + 0.5) / in_H * 2.0 - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).expand(masks.shape[0], out_H, out_W, 2)
        inp = masks.float().unsqueeze(1)  # (N, 1, in_H, in_W)
        out = F.grid_sample(inp, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        return out.squeeze(1).to(masks.dtype)

    return apply_coord_transform(result, xy_fn=xy_fn, box_fn=box_fn, mask_fn=mask_fn)
