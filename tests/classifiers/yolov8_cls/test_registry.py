from cls_core.classifier_registry import ClassifierRegistry
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_auto_registration():
    """
    Test that the auto-registration of classifiers works correctly.
    This will check if the classifiers are registered in the ClassifierRegistry.
    """
    ClassifierRegistry.auto_register_models()
    assert len(ClassifierRegistry._registry) > 0, "ClassifierRegistry should have registered classifiers."

    to_be_tested_keys = [
        ("ultralytics", "yolov8", "classification", "v0"),
    ]

    for key in to_be_tested_keys:
        key2 = ClassifierRegistry._generate_key(*key, info={})
        assert key2 in ClassifierRegistry._registry, f"Model {key} should be registered."
