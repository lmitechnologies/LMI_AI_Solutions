"""Bridge between JSON manifests (legacy dict-shape) and typed Configs.

External callers that load preprocessing manifests from JSON pass the resulting
list of dicts through ``parse_steps`` to get a ``List[Config]`` suitable for
``Preprocessor.preprocess``. In-process callers should construct Configs directly
(via ``lmi_utils.preprocess_utils.steps``) and skip this layer.
"""

from typing import Any, Dict, List, Type

from .operation import Config
from .ops import (
    CropConfig,
    FlipConfig,
    PadConfig,
    ResizeConfig,
    RotateConfig,
    TileConfig,
)

# Manifest "type" string -> Config dataclass.
STEP_TYPES: Dict[str, Type[Config]] = {
    "crop": CropConfig,
    "flip": FlipConfig,
    "pad": PadConfig,
    "resize": ResizeConfig,
    "rotate": RotateConfig,
    "tile": TileConfig,
}


def parse_steps(steps: List[Dict[str, Any]]) -> List[Config]:
    """Convert a list of manifest dicts into typed Configs.

    Each entry must have ``type`` and ``configuration`` keys; optional ``id`` is
    forwarded as the Config's ``id`` field.
    """
    if not isinstance(steps, list):
        raise TypeError(f"steps must be a list, got {type(steps)}")
    out: List[Config] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise TypeError(f"steps[{i}] must be a dict, got {type(step)}")
        if "type" not in step or "configuration" not in step:
            raise ValueError(f"steps[{i}] must contain keys 'type' and 'configuration', got {step!r}")
        cfg_cls = STEP_TYPES.get(step["type"])
        if cfg_cls is None:
            raise ValueError(f"steps[{i}]: unknown type {step['type']!r}. Known: {sorted(STEP_TYPES)}")
        kwargs = dict(step.get("configuration") or {})
        if step.get("id") is not None:
            kwargs["id"] = step["id"]
        out.append(cfg_cls(**kwargs))
    return out
