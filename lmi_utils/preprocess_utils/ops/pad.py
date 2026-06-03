from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from lmi_utils.gadget_utils.pipeline_utils import fit_im, fit_im_to_size

from .._coords import apply_coord_transform
from ..operation import Config, Meta, Operation


@dataclass
class PadConfig(Config):
    """Pad or crop each image to a target size.

    Either ``pad`` is supplied directly as [L, R, T, B] (positive => pad, negative => crop),
    or ``width`` / ``height`` are used to compute it (``fit_im_to_size`` semantics:
    target smaller than input center-crops on that axis).
    """

    width: Optional[int] = None
    height: Optional[int] = None
    pad: Optional[List[int]] = None
    value: int = 0

    def __post_init__(self):
        if self.pad is not None and len(self.pad) != 4:
            raise ValueError(f"pad: 'pad' must be [L, R, T, B], got {self.pad!r}")


@dataclass
class PadMeta(Meta):
    """Batched pad metadata.

    pads: per-image [L, R, T, B] actually applied.
    """

    pads: List[List[int]] = field(default_factory=list)


class PadOperation(Operation[PadConfig, PadMeta]):
    config_cls = PadConfig
    meta_cls = PadMeta

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: PadConfig) -> Tuple[List[torch.Tensor], PadMeta]:
        out_images: List[torch.Tensor] = []
        pads: List[List[int]] = []
        for img in images:
            if config.pad is not None:
                out_images.append(fit_im(img, config.pad, value=config.value))
                pads.append(list(config.pad))
            else:
                padded, pL, pR, pT, pB = fit_im_to_size(img, W=config.width, H=config.height, value=config.value)
                out_images.append(padded)
                pads.append([pL, pR, pT, pB])
        return out_images, PadMeta(pads=pads)

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], meta: PadMeta) -> List[torch.Tensor]:
        if len(images) != len(meta.pads):
            raise ValueError(f"pad: image count ({len(images)}) != meta count ({len(meta.pads)})")
        return [fit_im(img, [-p[0], -p[1], -p[2], -p[3]]) for img, p in zip(images, meta.pads)]

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], meta: PadMeta) -> List[Dict[str, Any]]:
        return [_apply_pad(r, p, forward=False) for r, p in zip(results, meta.pads)]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], meta: PadMeta) -> List[Dict[str, Any]]:
        return [_apply_pad(r, p, forward=True) for r, p in zip(results, meta.pads)]


def _apply_pad(result: Dict[str, Any], pad: List[int], *, forward: bool) -> Dict[str, Any]:
    pad_L, pad_R, pad_T, pad_B = pad

    def xy_fn(xy: torch.Tensor) -> torch.Tensor:
        offset = torch.tensor([pad_L, pad_T], dtype=torch.float32, device=xy.device)
        xy = xy.float()
        return xy + offset if forward else xy - offset

    def mask_fn(masks: torch.Tensor) -> torch.Tensor:
        pads = (pad_L, pad_R, pad_T, pad_B) if forward else (-pad_L, -pad_R, -pad_T, -pad_B)
        return F.pad(masks, pads)

    return apply_coord_transform(result, xy_fn=xy_fn, mask_fn=mask_fn)
