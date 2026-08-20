"""Building an rfdetr model from the variant a checkpoint records."""

import inspect
import math
from typing import Any, Dict, List, Optional

import torch


def _trust_kwarg(from_checkpoint: Any) -> Dict[str, bool]:
    """trust_checkpoint if this rfdetr takes it; rfdetr < 1.9 forwards unknown kwargs into the model config, which rejects them."""
    if "trust_checkpoint" in inspect.signature(from_checkpoint).parameters:
        # Allows the pickle fallback for our own training checkpoints, whose "args" can hold objects rfdetr's safe-load misses.
        return {"trust_checkpoint": True}
    return {}


def _find_tensor(weights: Dict[str, Any], suffix: str) -> Optional[Any]:
    for key, value in weights.items():
        if key.endswith(suffix) and isinstance(value, torch.Tensor):
            return value
    return None


def _pe_tracks_resolution(model_name: Any) -> bool:
    """Whether rfdetr sizes this variant's position grid from its resolution — what _resolution_from_weights assumes.

    Mirrors rfdetr's own sync rule (config.ModelConfig._sync_pe_with_resolution): true when the variant's default
    grid is its default resolution divided by its patch size. A variant that pins the grid instead (the deprecated
    RFDETRBase: 37 x 14 != 560) would give a wrong resolution. An unresolvable variant is assumed to follow the rule.
    """
    from rfdetr import variants

    config_cls = getattr(getattr(variants, str(model_name), None), "_model_config_class", None)
    fields = getattr(config_cls, "model_fields", None)
    if not isinstance(fields, dict):
        return True
    defaults = [getattr(fields.get(name), "default", None) for name in ("positional_encoding_size", "patch_size", "resolution")]
    if not all(isinstance(default, int) for default in defaults):
        return True
    grid, patch, resolution = defaults
    return grid * patch == resolution


def _resolution_from_weights(weights: Dict[str, Any]) -> Optional[int]:
    """The input size the backbone's position grid was trained for: grid side x patch size.

    Returns None when the grid is not square (an unexpected backbone layout), leaving the resolution to rfdetr.
    """
    position = _find_tensor(weights, "embeddings.position_embeddings")
    patch = _find_tensor(weights, "patch_embeddings.projection.weight")
    if position is None or patch is None or position.ndim != 3 or patch.ndim != 4:
        return None
    grid = math.isqrt(position.shape[1] - 1)  # one non-patch token: the cls token
    if grid * grid != position.shape[1] - 1:
        return None
    return grid * patch.shape[-1]


def resolution_from_checkpoint(model_path: str) -> Optional[int]:
    """The square resolution a checkpoint was trained at, or None when it cannot be read.

    Callers must pass it back in, or the model runs at the variant's default resolution: rfdetr < 1.9 ignores the
    checkpoint's model_config, naming the variant explicitly skips it on every version, and rfdetr strips
    model_config out of checkpoint_best_total.pth altogether — hence the fallback to the weights themselves.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return None
    model_config = checkpoint.get("model_config")
    resolution = model_config.get("resolution") if isinstance(model_config, dict) else None
    if isinstance(resolution, int):
        return resolution
    weights = checkpoint.get("model")
    if not isinstance(weights, dict) or not _pe_tracks_resolution(checkpoint.get("model_name")):
        return None
    return _resolution_from_weights(weights)


def load_from_checkpoint(model_path: str, supported: List[str], **kwargs: Any) -> Any:
    """Build the rfdetr model whose variant the checkpoint names, turning rfdetr's failures into an actionable error.

    num_classes is deliberately not a parameter: rfdetr reads it off the checkpoint's detection head, and passing it
    would pin the head as a user override.

    Args:
        model_path: Path to an rfdetr checkpoint.
        supported: Variant names the caller accepts, listed in the error when the checkpoint names none.
        **kwargs: Forwarded to the rfdetr model constructor (device, resolution, ...).

    Returns:
        An rfdetr model instance of the recorded variant.

    Raises:
        ValueError: If the checkpoint records no variant.
    """
    from rfdetr import RFDETR

    try:
        return RFDETR.from_checkpoint(model_path, **_trust_kwarg(RFDETR.from_checkpoint), **kwargs)
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"Could not read the RF-DETR variant from {model_path} ({type(e).__name__}: {e}). "
            f"rfdetr records it since 1.7.0; older checkpoints and Roboflow's published starter weights do not. "
            f"Specify model_type explicitly, one of: {', '.join(supported)}."
        ) from e
