import logging
from typing import Any, Iterable

import torch

from anomaly_detectors.ad_core.anomaly_detector_registry import AnomalyDetectorRegistry

from ..base import Anomalib_Base


@AnomalyDetectorRegistry.register(
    metadata=dict(
        frameworks=["anomalib2"],
        model_names=["patchcore", "padim", "efficientad"],
        tasks=["anomalydetection", "seg"],
        versions=["v2"],
    )
)
class AnomalyModel(Anomalib_Base):
    """AD model inference for Anomalib v2."""

    logger = logging.getLogger("AnomalyModel v2")

    def _get_pt_transforms(self) -> Iterable:
        return self.pt_model.pre_processor.transform.transforms

    def _extract_pt_output(self, preds: Any) -> torch.Tensor:
        if isinstance(preds, tuple):
            return preds[2]
        return preds.anomaly_map


if __name__ == "__main__":
    from .._cli import run_cli

    run_cli(AnomalyModel)
