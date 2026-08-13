from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


def bytes_to_mib(x: float) -> float:
    return x / 1024**2


def mib_to_bytes(x: float) -> int:
    return int(x * 1024**2)


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def dtype_from_precision(precision, fallback: torch.dtype = torch.float32) -> torch.dtype:
    """Resolve a torch dtype from a precision tag, dtype, or finalized dtype string.

    Accepts Lightning-style tags ("16-mixed", "bf16"), plain names ("float16"),
    and already-finalized ``str(dtype)`` values ("torch.float16").
    """
    if isinstance(precision, torch.dtype):
        return precision

    p = str(precision).lower().removeprefix("torch.")

    if p in {"16", "16-mixed", "fp16", "float16", "half"}:
        return torch.float16

    if p in {"bf16", "bf16-mixed", "bfloat16"}:
        return torch.bfloat16

    if p in {"32", "32-true", "fp32", "float32", "full"}:
        return torch.float32

    return fallback


@dataclass
class MemoryBudget:
    memory_limit_mib: float
    fixed_overhead_mib: float = 0.0
    safety_fraction: float = 1.0
    reserve_mib: float = 0.0

    @property
    def usable_mib(self) -> float:
        return self.memory_limit_mib * self.safety_fraction - self.reserve_mib

    @property
    def variable_budget_mib(self) -> float:
        return self.usable_mib - self.fixed_overhead_mib


@dataclass
class TileConfig:
    image_size: tuple[int, int]
    batch_size: int
    tile_size: tuple[int, int] | None = None
    stride: tuple[int, int] | None = None
    channels: int = 3

    @property
    def uses_tiling(self) -> bool:
        return self.tile_size is not None

    @property
    def effective_tile_size(self) -> tuple[int, int]:
        return self.tile_size or self.image_size

    @property
    def effective_stride(self) -> tuple[int, int]:
        return self.stride or self.effective_tile_size

    @property
    def n_tiles_h(self) -> int:
        if not self.uses_tiling:
            return 1

        H, _ = self.image_size
        Th, _ = self.effective_tile_size
        Sh, _ = self.effective_stride

        return max(1, math.ceil((H - Th) / Sh) + 1)

    @property
    def n_tiles_w(self) -> int:
        if not self.uses_tiling:
            return 1

        _, W = self.image_size
        _, Tw = self.effective_tile_size
        _, Sw = self.effective_stride

        return max(1, math.ceil((W - Tw) / Sw) + 1)

    @property
    def tiles_per_image(self) -> int:
        return self.n_tiles_h * self.n_tiles_w

    @property
    def effective_tile_batch(self) -> int:
        return self.batch_size * self.tiles_per_image


@dataclass
class FeatureProfile:
    feature_shapes: dict[str, tuple[int, int, int, int]]
    feature_dtypes: dict[str, str]
    embedding_shape_per_tile: tuple[int, int, int, int]
    embedding_dtype: str
    stitched_embedding_shape: tuple[int, int, int, int]

    @property
    def embedding_dim(self) -> int:
        return self.embedding_shape_per_tile[1]

    @property
    def stitched_h(self) -> int:
        return self.stitched_embedding_shape[-2]

    @property
    def stitched_w(self) -> int:
        return self.stitched_embedding_shape[-1]

    @property
    def patches_per_image(self) -> int:
        return self.stitched_h * self.stitched_w


@dataclass
class MemoryEstimate:
    max_train_images: int | None
    requested_train_images: int | None = None
    fits_requested_train_images: bool | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    per_image: dict[str, Any] = field(default_factory=dict)
    requested: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    tiling: dict[str, Any] = field(default_factory=dict)
    feature_profile: dict[str, Any] = field(default_factory=dict)

    @property
    def budget_mib_after_fixed(self) -> float | None:
        return self.budget.get("effective_budget_mib")

    @property
    def per_image_peak_mib(self) -> float | None:
        return self.per_image.get("peak_bank_per_image_mib")

    @property
    def details(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "per_image": self.per_image,
            "requested": self.requested,
            "model": self.model,
            "tiling": self.tiling,
            "feature_profile": self.feature_profile,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
