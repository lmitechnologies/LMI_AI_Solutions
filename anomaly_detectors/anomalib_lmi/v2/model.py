import logging
from typing import Any, Iterable

import torch

from anomaly_detectors.ad_core.anomaly_detector_registry import AnomalyDetectorRegistry
from lmi_common.model_factory import ModelFactory

from ..base import Anomalib_Base, AnomalibPT, register_backends


class AnomalibPTv2(AnomalibPT):
    """PT/TorchScript backend with Anomalib v2-specific transform and output extraction."""

    logger = logging.getLogger("AnomalyModel v2 PT")

    def _get_pt_transforms(self) -> Iterable:
        return self.pt_model.pre_processor.transform.transforms

    def _extract_pt_output(self, preds: Any) -> torch.Tensor:
        if isinstance(preds, tuple):
            return preds[2]
        return preds.anomaly_map


@AnomalyDetectorRegistry.register(
    metadata=dict(
        frameworks=["anomalib2"],
        model_names=["patchcore", "padim", "efficientad"],
        tasks=["anomalydetection", "seg"],
        versions=["v2"],
    )
)
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
