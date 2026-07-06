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
