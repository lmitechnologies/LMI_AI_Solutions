import logging

from anomaly_detectors.ad_core.anomaly_detector_registry import AnomalyDetectorRegistry

logger = logging.getLogger(__name__)


def test_auto_register_models():
    """
    Test that all anomaly detectors are automatically registered.
    """
    AnomalyDetectorRegistry.auto_register_models()

    # Check if the registry is populated
    assert len(AnomalyDetectorRegistry._registry) > 0, "AnomalyDetectorRegistry should have registered classes."

    # Check if specific known detectors are registered
    known_keys = [
        ("anomalib2", "patchcore", "anomalydetection", "v2"),
        ("anomalib1", "patchcore", "anomalydetection", "v1"),
        ("anomalib", "patchcore", "anomalydetection", "v0"),
        ("anomalib0", "padim", "anomalydetection", "v0"),
    ]
    for key in known_keys:
        key2 = AnomalyDetectorRegistry._generate_key(*key, info={})
        assert key2 in AnomalyDetectorRegistry._registry.keys(), f"Expected {key2} to be registered in AnomalyDetectorRegistry."
