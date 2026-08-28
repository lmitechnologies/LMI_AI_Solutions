import logging
from typing import Any, Iterable

import torch

from lmi_common.model_factory import ModelFactory

from ..base import Anomalib_Base, AnomalibPT, register_backends


class AnomalibPTv2(AnomalibPT):
    """PT/TorchScript backend with Anomalib v2-specific transform and output extraction."""

    logger = logging.getLogger("AnomalyModel v2 PT")

    def _get_pt_transforms(self) -> Iterable:
        return self.pt_model.pre_processor.transform.transforms

    def _extract_pt_output(self, preds: Any) -> torch.Tensor:
        # TolerantAnomalyDINO with emit_patch_class=True returns (InferenceBatch, patch_class_map)
        if isinstance(preds, tuple) and hasattr(preds[0], "anomaly_map"):
            return preds[0].anomaly_map
        # legacy anomalib v1-style tuple: (pred_score, ?, anomaly_map)
        if isinstance(preds, tuple):
            return preds[2]
        return preds.anomaly_map

    def _forward_with_scores(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        preds = self.pt_model(input_batch)
        if isinstance(preds, tuple) and hasattr(preds[0], "anomaly_map"):
            return preds[0].anomaly_map, getattr(preds[0], "pred_score", None)
        if isinstance(preds, tuple):
            return preds[2], preds[0]
        return preds.anomaly_map, getattr(preds, "pred_score", None)


class AnomalyModel(ModelFactory, Anomalib_Base):
    """AD model factory for Anomalib v2. Dispatches on file extension.

    Supported extensions:
        .engine             -> AnomalibTRT  (TensorRT)
        .onnx               -> AnomalibONNX (ONNX Runtime)
        .pt / .ts / .torchscript -> AnomalibPTv2 (PyTorch / TorchScript)
    """

    logger = logging.getLogger("AnomalyModel v2")
    _registry = {}


register_backends(AnomalyModel, AnomalibPTv2)


if __name__ == "__main__":
    from .._cli import run_cli

    run_cli(AnomalyModel)
