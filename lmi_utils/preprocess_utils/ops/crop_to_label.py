from dataclasses import dataclass
from typing import Any, ClassVar, Dict

from ..operation import Config
from .crop import CropConfig


@dataclass
class CropToLabelConfig(Config):
    """Macro config: rewrites itself to a CropConfig using runtime-supplied boxes.

    label: declared name of the upstream class/region this crop targets
           (e.g. "BOTTLE-BBOX"). Used as a contract for tooling; the runtime
           channel supplies the actual per-image boxes.

    Runtime value (caller -> Preprocessor.preprocess):
        boxes: list of [x1, y1, x2, y2] in original-image space, one per image.
    """

    label: str = ""
    is_runtime: ClassVar[bool] = True

    def __post_init__(self):
        if not self.label:
            raise ValueError("CropToLabelConfig: 'label' is required")

    def bind(self, runtime: Dict[str, Any]) -> CropConfig:
        if not runtime:
            raise ValueError(
                f"crop-to-label (label='{self.label}'): no runtime value provided. "
                f"Set 'id' and pass runtime={{<id>: {{'boxes': [...]}}}} to preprocess()."
            )
        boxes = runtime.get("boxes")
        if not isinstance(boxes, list) or len(boxes) == 0:
            raise ValueError(f"crop-to-label (label='{self.label}'): runtime['boxes'] must be a non-empty list, got {boxes!r}")
        return CropConfig(boxes=boxes, id=self.id)
