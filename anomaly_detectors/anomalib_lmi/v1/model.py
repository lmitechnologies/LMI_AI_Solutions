import logging
from collections.abc import Sequence
from typing import Any, Iterable

import torch

from anomaly_detectors.ad_core.anomaly_detector_registry import AnomalyDetectorRegistry

from ..base import Anomalib_Base


@AnomalyDetectorRegistry.register(
    metadata=dict(
        frameworks=["anomalib1"],
        model_names=["patchcore", "padim", "efficientad"],
        tasks=["anomalydetection", "seg"],
        versions=["v1"],
    )
)
class AnomalyModel(Anomalib_Base):
    """AD model inference for Anomalib v1."""

    logger = logging.getLogger("AnomalyModel v1")

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


if __name__ == "__main__":
    from .._cli import run_cli

    run_cli(AnomalyModel)
