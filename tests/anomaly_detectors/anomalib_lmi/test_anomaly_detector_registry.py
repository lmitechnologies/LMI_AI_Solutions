import logging

from anomaly_detectors.ad_core.anomaly_detector_registry import AnomalyDetectorRegistry

logger = logging.getLogger(__name__)


def test_backends_table_covers_known_detectors():
    """The BACKENDS table expands to the known lookup keys — no backend imports needed."""
    assert len(AnomalyDetectorRegistry._key_map) > 0, "AnomalyDetectorRegistry should have registered keys."

    known_keys = [
        ("anomalib2", "patchcore", "anomalydetection", "v2"),
        ("anomalib1", "patchcore", "anomalydetection", "v1"),
        ("anomalib0", "patchcore", "anomalydetection", "v0"),
        ("anomalib0", "padim", "anomalydetection", "v0"),
    ]
    for key in known_keys:
        key2 = AnomalyDetectorRegistry._generate_key(*key)
        assert key2 in AnomalyDetectorRegistry._key_map, f"Expected {key2} to be registered in AnomalyDetectorRegistry."

    # frameworks are numbered; the bare legacy name is not registered under any version
    for version in ("v0", "v1", "v2"):
        bare = AnomalyDetectorRegistry._generate_key("anomalib", "patchcore", "anomalydetection", version)
        assert bare not in AnomalyDetectorRegistry._key_map


def test_get_version_inference():
    assert AnomalyDetectorRegistry._get_version({"version": "v2"}, "anomalib") == "v2"  # explicit wins
    assert AnomalyDetectorRegistry._get_version({}, "anomalib0") == "v0"
    assert AnomalyDetectorRegistry._get_version({}, "anomalib2") == "v2"
    assert AnomalyDetectorRegistry._get_version({}, "anomalib12") == "v12"


def test_get_version_warns_on_non_standard_framework(caplog):
    """A framework name without trailing digits carries no version, so it warns and falls back to v1."""
    for framework in ("anomalib2beta", "anomalib"):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert AnomalyDetectorRegistry._get_version({}, framework) == "v1"
        assert any("defaulting to 'v1'" in r.message for r in caplog.records), framework


def test_get_task_prefers_task_then_model_type():
    assert AnomalyDetectorRegistry._get_task({"task": "anomalydetection", "model_type": "x"}) == "anomalydetection"
    assert AnomalyDetectorRegistry._get_task({"model_type": "anomalydetection"}) == "anomalydetection"
    assert AnomalyDetectorRegistry._get_task({}) == "seg"
