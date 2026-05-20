from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from lmi_utils.gadget_utils.pipeline_utils import fit_im, fit_im_to_size

from .._coords import apply_coord_transform
from ..operation import Operation


class PadOperation(Operation):
    """Pad or crop each image to a target size; emit one ``pad`` history entry per image.

    Configuration (one of):
        width (int) and/or height (int): pad/crop to target H/W (positive => pad, negative => crop strip).
            The op uses ``fit_im_to_size`` semantics: when the target dim is smaller than the input,
            the image is center-cropped on that axis.
        pad (list[int], optional): explicit ``[L, R, T, B]`` to apply (positive => pad, negative => crop).

    Either ``pad`` is supplied directly, or ``width``/``height`` are used to compute it.

    Metadata schema (per image)::

        {"pad": [L, R, T, B]}
    """

    name = "pad"

    @classmethod
    def build_step(
        cls,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        pad: Optional[List[int]] = None,
        value: int = 0,
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"value": value}
        if pad is not None:
            cfg["pad"] = pad
        if width is not None:
            cfg["width"] = width
        if height is not None:
            cfg["height"] = height
        return cls._finalize_step(cfg, id=id)

    @torch.inference_mode()
    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        explicit = config.get("pad")
        width = config.get("width")
        height = config.get("height")
        value = config.get("value", 0)

        out_images: List[torch.Tensor] = []
        meta_list: List[Dict[str, Any]] = []
        for img in images:
            if explicit is not None:
                if len(explicit) != 4:
                    raise ValueError(f"pad: 'pad' must be [L, R, T, B], got {explicit!r}")
                out_images.append(fit_im(img, explicit, value=value))
                meta_list.append({"pad": list(explicit)})
            else:
                padded, pL, pR, pT, pB = fit_im_to_size(img, W=width, H=height, value=value)
                out_images.append(padded)
                meta_list.append({"pad": [pL, pR, pT, pB]})

        return out_images, meta_list

    @torch.inference_mode()
    def revert_images(self, images: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> List[torch.Tensor]:
        if len(images) != len(metadata):
            raise ValueError(f"pad: image count ({len(images)}) != metadata count ({len(metadata)})")
        return [fit_im(img, [-m["pad"][0], -m["pad"][1], -m["pad"][2], -m["pad"][3]]) for img, m in zip(images, metadata)]

    @torch.inference_mode()
    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [_apply_pad(r, m, forward=False) for r, m in zip(results, metadata)]

    @torch.inference_mode()
    def apply_coords(self, results: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [_apply_pad(r, m, forward=True) for r, m in zip(results, metadata)]


def _apply_pad(result: Dict[str, Any], m: Dict[str, Any], *, forward: bool) -> Dict[str, Any]:
    pad_L, pad_R, pad_T, pad_B = m["pad"]

    def xy_fn(xy: torch.Tensor) -> torch.Tensor:
        offset = torch.tensor([pad_L, pad_T], dtype=torch.float32, device=xy.device)
        xy = xy.float()
        return xy + offset if forward else xy - offset

    def mask_fn(masks: torch.Tensor) -> torch.Tensor:
        # masks are (N, H, W); F.pad accepts negative values to crop strips.
        pads = (pad_L, pad_R, pad_T, pad_B) if forward else (-pad_L, -pad_R, -pad_T, -pad_B)
        return F.pad(masks, pads)

    return apply_coord_transform(result, xy_fn=xy_fn, mask_fn=mask_fn)
