"""Generic optional model hooks used by the shared training entry point."""

from __future__ import annotations

from pathlib import Path

import anomalib.models as ad_models


def prepare_model_training_config(cfg: dict, config_path: str | Path) -> dict:
    """Let a model class translate concise config before generic construction.

    Standard Anomalib models do not implement ``prepare_training_config`` and
    therefore pass through unchanged.
    """
    model_cfg = cfg.get("model", {}) or {}
    class_name = model_cfg.get("class_name")
    model_class = getattr(ad_models, class_name, None) if class_name else None
    hook = getattr(model_class, "prepare_training_config", None)
    if callable(hook):
        prepared = hook(cfg, config_path=Path(config_path))
        if prepared is not None:
            return prepared
    return cfg


def notify_model_exported(model, cfg: dict) -> None:
    """Let a model publish optional model-owned artifacts after export."""
    hook = getattr(model, "on_training_exported", None)
    if callable(hook):
        hook(cfg)
