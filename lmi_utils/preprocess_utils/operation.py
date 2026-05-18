from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch


class Operation(ABC):
    """
    Bundles the forward + revert-images + revert-coords for one preprocessing op.

    Pair this with `Preprocessor.register(op)` and `Reconstructor.register(op)` so
    a single object owns all three sides of the contract — no more chance of
    forgetting to register a revert handler under a matching name.

    Subclasses must set `name` (the key used in processing_steps and history)
    and implement `forward`. `revert_images` and `revert_coords` default to
    no-ops, which is the right behavior for image-space-only ops (normalize,
    clahe, denoise, ...).

    Device contract:
        Implementations MUST preserve the input device — output tensors live
        on the same device as inputs. When allocating constants or scratch
        tensors, always pass `device=<input>.device` (never default to CPU
        and `.to(...)` later — that's a per-call H2D transfer).
    """

    name: str = ""

    def bind(self, step: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve a caller-supplied runtime patch into a concrete step before forward.

        Args:
            step: The manifest step `{type, configuration, id?}`.
            runtime: The value for this step keyed from the caller's
                     `runtime={id: value}` dict. Empty dict when no value
                     targeted this step.

        Returns:
            A resolved step. May rewrite `type` (macro ops like crop-to-label →
            crop) and/or extend `configuration` with runtime-derived fields.
            Default implementation returns the step unchanged — appropriate for
            any op that does not consume caller runtime data.
        """
        return step

    @abstractmethod
    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Any]]:
        """
        Run the op.

        Returns:
            (images, metadata_list): processed images and a per-source-image
            metadata list. The metadata is fed back unchanged to revert_images
            and revert_coords.
        """

    def revert_images(self, images: List[torch.Tensor], metadata: List[Any]) -> List[torch.Tensor]:
        """Default: identity. Override for ops that change image geometry."""
        return images

    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Any]) -> List[Dict[str, Any]]:
        """Default: identity. Override for ops that change coordinate space."""
        return results
