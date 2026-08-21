"""Building an rfdetr model from the variant a checkpoint records."""

import inspect
from typing import Any, Dict, List


def _trust_kwarg(from_checkpoint: Any) -> Dict[str, bool]:
    """trust_checkpoint if this rfdetr takes it; rfdetr < 1.9 forwards unknown kwargs into the model config, which rejects them."""
    if "trust_checkpoint" in inspect.signature(from_checkpoint).parameters:
        # Allows the pickle fallback for our own training checkpoints, whose "args" can hold objects rfdetr's safe-load misses.
        return {"trust_checkpoint": True}
    return {}


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
