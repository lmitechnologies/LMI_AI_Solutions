from typing import Any, Dict, List, Tuple

import torch

from ..operation import Operation


class CropToLabelOperation(Operation):
    """Macro op: rewrites itself to a concrete `crop` step using runtime-supplied boxes.

    Manifest configuration:
        label: declared name of the upstream class/region this crop targets
               (e.g. "BOTTLE-BBOX"). Used as a contract for tooling; the runtime
               channel supplies the actual per-image boxes.

    Runtime payload (caller -> Preprocessor.preprocess):
        boxes: list of [x1, y1, x2, y2] in original-image space, one per image.

    `forward` / `revert_*` are never called — `bind` always rewrites this step
    to `type: "crop"` before forward dispatch.
    """

    name = "crop-to-label"

    def bind(self, step: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
        configuration = step.get("configuration") or {}
        label = configuration.get("label")
        if not label:
            raise ValueError("crop-to-label: 'label' is required in configuration")

        if not runtime:
            raise ValueError(
                f"crop-to-label (label='{label}'): no runtime payload provided. "
                f"Pass runtime=[{{'type': 'crop-to-label', 'runtime': {{'boxes': [...]}}}}] to preprocess()."
            )

        boxes = runtime.get("boxes")
        if not isinstance(boxes, list) or len(boxes) == 0:
            raise ValueError(f"crop-to-label (label='{label}'): runtime['boxes'] must be a non-empty list, got {boxes!r}")

        resolved: Dict[str, Any] = {
            "type": "crop",
            "configuration": {"boxes": boxes},
        }
        if step.get("instance") is not None:
            resolved["instance"] = step["instance"]
        return resolved

    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Any]]:
        raise RuntimeError("crop-to-label must be resolved via bind() before forward")
