import logging

from classifiers.cls_core.classifier_registry import ClassifierRegistry

logger = logging.getLogger(__name__)


def test_backends_table_covers_known_classifiers():
    """The BACKENDS table expands to the known lookup keys — no backend imports needed."""
    assert len(ClassifierRegistry._key_map) > 0, "ClassifierRegistry should have registered keys."

    to_be_tested_keys = [
        ("ultralytics", "yolo", "classification", "v1"),
    ]

    for key in to_be_tested_keys:
        key2 = ClassifierRegistry._generate_key(*key, info={})
        assert key2 in ClassifierRegistry._key_map, f"Model {key} should be registered."
