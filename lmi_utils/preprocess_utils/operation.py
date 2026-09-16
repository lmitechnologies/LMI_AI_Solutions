from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import torch


@dataclass
class Config:
    """Base for all forward-step configs. Subclass + @dataclass per op."""

    id: Optional[str] = None


@dataclass
class Meta:
    """Base for all per-step metadata. Fields are batched (struct-of-arrays)."""


CfgT = TypeVar("CfgT", bound=Config)
MetaT = TypeVar("MetaT", bound=Meta)


class Operation(ABC, Generic[CfgT, MetaT]):
    """Forward + revert-images + revert-coords + apply-coords for one op.

    Subclasses declare ``config_cls`` and ``meta_cls`` and implement ``forward``.
    The three revert handlers default to identity (correct for image-space-only ops).

    Device contract:
        Output tensors live on the same device as the input tensors. Always
        allocate scratch with ``device=<input>.device``.
    """

    config_cls: ClassVar[Type[Config]]
    meta_cls: ClassVar[Type[Meta]]

    #: Coord handlers may drop instances, so an emptied coord field is expected, not a lost one.
    filters_instances: ClassVar[bool] = False

    @abstractmethod
    def forward(self, images: List[torch.Tensor], config: CfgT) -> Tuple[List[torch.Tensor], MetaT]:
        """Run the op. Returns (processed_images, batched_meta)."""

    def revert_images(self, images: List[torch.Tensor], meta: MetaT) -> List[torch.Tensor]:
        """Default identity. Override for ops that change image geometry."""
        return images

    def revert_coords(self, results: List[Dict[str, Any]], meta: MetaT) -> List[Dict[str, Any]]:
        """Default identity. Override for ops that change coordinate space."""
        return results

    def apply_coords(self, results: List[Dict[str, Any]], meta: MetaT) -> List[Dict[str, Any]]:
        """Forward complement of ``revert_coords``. Default identity."""
        return results
