import logging
import re
from typing import Any, Dict, Optional

from lmi_common.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class AnomalyDetectorRegistry(ModelRegistry):
    BACKENDS = [
        {
            "frameworks": ["anomalib", "anomalib0"],
            "model_names": ["patchcore", "padim"],
            "tasks": ["anomalydetection", "seg"],
            "versions": ["v0", "v1"],
            "class_path": "anomaly_detectors.anomalib_lmi.v0.model:AnomalyModel",
        },
        {
            "frameworks": ["anomalib1"],
            "model_names": ["patchcore", "padim", "efficientad"],
            "tasks": ["anomalydetection", "seg"],
            "versions": ["v1"],
            "class_path": "anomaly_detectors.anomalib_lmi.v1.model:AnomalyModel",
        },
        {
            "frameworks": ["anomalib2"],
            "model_names": ["patchcore", "padim", "efficientad"],
            "tasks": ["anomalydetection", "seg"],
            "versions": ["v2"],
            "class_path": "anomaly_detectors.anomalib_lmi.v2.model:AnomalyModel",
        },
    ]

    @classmethod
    def _get_version(cls, metadata: Dict[str, Any], framework: str) -> str:
        version = metadata.get("version")
        if version:
            return version
        # infer from trailing digits in the framework name, e.g. "anomalib2" -> "v2"
        match = re.search(r"(\d+)$", framework)
        if match:
            return f"v{match.group(1)}"
        if framework.lower() != "anomalib":  # bare "anomalib" is a known legacy name
            logger.warning(f"No 'version' in metadata and no trailing digits in framework '{framework}'; defaulting to 'v1'.")
        return "v1"

    @classmethod
    def _get_task(cls, metadata: Dict[str, Any]) -> Optional[str]:
        return metadata.get("task") or metadata.get("model_type") or "seg"
