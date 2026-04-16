from typing import Any, Dict, Optional

from lmi_common.model_registry import ModelRegistry


class AnomalyDetectorRegistry(ModelRegistry):
    PACKAGES = ["anomaly_detectors.anomalib_lmi", "anomaly_detectors.ad_core"]
    TARGET_MODULE_SUFFIXES = [".anomaly_model", ".anomaly_model2", ".anomaly_model_v2"]
    _registry = {}

    @classmethod
    def _get_version(cls, metadata: Dict[str, Any], framework: str) -> str:
        return metadata.get("version", "v0" if "0" in framework else "v1")

    @classmethod
    def _get_task(cls, metadata: Dict[str, Any]) -> Optional[str]:
        return metadata.get("task", "seg") or metadata.get("model_type", "seg")
