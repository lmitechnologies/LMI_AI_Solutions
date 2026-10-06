from __future__ import annotations

import torch

from .base import BaseAnomalibMemoryEstimator
from .common import (
    FeatureProfile,
    MemoryBudget,
    MemoryEstimate,
    TileConfig,
    dtype_from_precision,
)
from .padim import PadimMemoryEstimate, PadimMemoryEstimator
from .patchcore import PatchCoreMemoryEstimate, PatchCoreMemoryEstimator

__all__ = [
    "BaseAnomalibMemoryEstimator",
    "FeatureProfile",
    "MemoryBudget",
    "MemoryEstimate",
    "TileConfig",
    "dtype_from_precision",
    "PadimMemoryEstimate",
    "PadimMemoryEstimator",
    "PatchCoreMemoryEstimate",
    "PatchCoreMemoryEstimator",
    "make_memory_estimator",
]


def make_memory_estimator(
    model,
    tile_config: TileConfig,
    *,
    precision: str | torch.dtype = "float32",
    profiling_device: str = "cuda",
    profiling_dtype: torch.dtype | None = None,
    stats_dtype: torch.dtype = torch.float32,
) -> BaseAnomalibMemoryEstimator:
    """Build the memory estimator matching the anomalib model type."""
    name = type(model).__name__.lower()

    if "patchcore" in name:
        return PatchCoreMemoryEstimator(
            model=model,
            tile_config=tile_config,
            precision=precision,
            profiling_device=profiling_device,
            profiling_dtype=profiling_dtype,
        )

    if "padim" in name:
        return PadimMemoryEstimator(
            model=model,
            tile_config=tile_config,
            precision=precision,
            profiling_device=profiling_device,
            profiling_dtype=profiling_dtype,
            stats_dtype=stats_dtype,
        )

    raise NotImplementedError(f"No memory estimator for model type: {type(model).__name__}. Supported: patchcore and padim.")
