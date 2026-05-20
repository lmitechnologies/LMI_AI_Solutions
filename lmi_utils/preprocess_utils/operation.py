from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch


class Operation(ABC):
    """
    Bundles forward + revert-images + revert-coords + apply-coords for one preprocessing op.

    Pair this with `Preprocessor.register(op)` and `Reconstructor.register(op)` so a single
    object owns every side of the contract — no chance of forgetting to register a handler
    under a matching name.

    Subclasses must set `name` (the key used in processing_steps and history) and implement
    `forward`. The three coord/image handlers default to no-ops, which is the right behavior
    for image-space-only ops (normalize, clahe, denoise, ...).

    Schema:
        Each forward call emits one history entry of shape::

            {"type": <name>, "metadata": [<per_image_dict>, ...], "id"?: str}

        ``metadata`` is always a per-image list of length B (batch size). Per-image dicts
        use descriptive keys (e.g. ``{"src_size": [w, h], "dst_size": [w, h]}``) rather
        than positional lists.

    Device contract:
        Implementations MUST preserve the input device — output tensors live on the same
        device as inputs. When allocating constants or scratch tensors, always pass
        ``device=<input>.device`` (never default to CPU and ``.to(...)`` later).
    """

    name: str = ""

    @classmethod
    def _finalize_step(cls, configuration: Dict[str, Any], id: Optional[str] = None) -> Dict[str, Any]:
        """Assemble a step dict from a configuration; shared by every subclass's ``build_step``."""
        step: Dict[str, Any] = {"type": cls.name, "configuration": configuration}
        if id is not None:
            step["id"] = id
        return step

    def bind(self, step: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve a caller-supplied runtime patch into a concrete step before forward.

        Args:
            step: The manifest step ``{type, configuration, id?}``.
            runtime: The value for this step keyed from the caller's ``runtime={id: value}``
                dict. Empty dict when no value targeted this step.

        Returns:
            A resolved step. May rewrite ``type`` (macro ops like crop-to-label → crop)
            and/or extend ``configuration`` with runtime-derived fields. Default
            implementation returns the step unchanged — appropriate for any op that does
            not consume caller runtime data.
        """
        return step

    @abstractmethod
    def forward(self, images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[Any]]:
        """
        Run the op.

        Returns:
            (images, metadata_list): processed images and a per-source-image metadata
            list of length B. The metadata is fed back unchanged to ``revert_images``,
            ``revert_coords``, and ``apply_coords``.
        """

    def revert_images(self, images: List[torch.Tensor], metadata: List[Any]) -> List[torch.Tensor]:
        """Default: identity. Override for ops that change image geometry."""
        return images

    def revert_coords(self, results: List[Dict[str, Any]], metadata: List[Any]) -> List[Dict[str, Any]]:
        """Default: identity. Override for ops that change coordinate space."""
        return results

    def apply_coords(self, results: List[Dict[str, Any]], metadata: List[Any]) -> List[Dict[str, Any]]:
        """
        Forward complement of ``revert_coords`` — apply the op's coordinate transform.

        Default: identity. Override on geometric ops so callers can map original-space
        coordinates into preprocessed space (the inverse of ``revert_coords``).
        """
        return results
