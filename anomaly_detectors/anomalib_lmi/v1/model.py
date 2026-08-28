import logging
from collections.abc import Sequence
from typing import Any, Iterable

import torch

from lmi_common.model_factory import ModelFactory

from ..base import Anomalib_Base, AnomalibPT, register_backends


class AnomalibPTv1(AnomalibPT):
    """PT/TorchScript backend with Anomalib v1-specific transform and output extraction."""

    logger = logging.getLogger("AnomalyModel v1 PT")

    def _get_pt_transforms(self) -> Iterable:
        return self.pt_model.transform.transforms

    def _extract_pt_output(self, preds: Any) -> torch.Tensor:
        if isinstance(preds, torch.Tensor):
            return preds
        if isinstance(preds, dict):
            return preds["anomaly_map"]
        if isinstance(preds, Sequence):
            return preds[1]
        raise TypeError(f"Unknown prediction type: {type(preds)}")

    def _forward_with_scores(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        preds = self.pt_model(input_batch)
        if isinstance(preds, torch.Tensor):
            return preds, None
        if isinstance(preds, dict):
            score = preds.get("pred_score", preds.get("pred_scores"))
            return preds["anomaly_map"], score
        if isinstance(preds, Sequence):
            score = preds[0] if len(preds) > 2 and isinstance(preds[0], torch.Tensor) else None
            return preds[1], score
        raise TypeError(f"Unknown prediction type: {type(preds)}")


class AnomalyModel(ModelFactory, Anomalib_Base):
    """AD model factory for Anomalib v1. Dispatches on file extension.

    Supported extensions:
        .engine             -> AnomalibTRT  (TensorRT)
        .onnx               -> AnomalibONNX (ONNX Runtime)
        .pt / .ts / .torchscript -> AnomalibPTv1 (PyTorch / TorchScript)
    """

    logger = logging.getLogger("AnomalyModel v1")
    _registry = {}


register_backends(AnomalyModel, AnomalibPTv1)


if __name__ == "__main__":
    from .._cli import run_cli

    run_cli(AnomalyModel)
