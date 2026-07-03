import logging

import pytest

from lmi_common.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concrete registry subclass used across all tests
# ---------------------------------------------------------------------------


class FakeRegistry(ModelRegistry):
    PACKAGES = []
    TARGET_MODULE_SUFFIXES = [".model"]
    _registry = {}


# ---------------------------------------------------------------------------
# Helper: fresh registry state per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_registry():
    """Reset FakeRegistry between tests so registrations don't bleed over."""
    FakeRegistry._registry.clear()
    FakeRegistry._failed_imports.clear()
    yield
    FakeRegistry._registry.clear()
    FakeRegistry._failed_imports.clear()


# ---------------------------------------------------------------------------
# __init_subclass__ enforcement
# ---------------------------------------------------------------------------


def test_subclass_missing_all_required_attributes_raises():
    with pytest.raises(TypeError, match="must define"):

        class BadRegistry(ModelRegistry):
            pass  # missing PACKAGES, TARGET_MODULE_SUFFIXES, _registry


def test_subclass_missing_one_required_attribute_raises():
    with pytest.raises(TypeError, match="must define"):

        class PartialRegistry(ModelRegistry):
            PACKAGES = []
            TARGET_MODULE_SUFFIXES = [".model"]
            # missing _registry


def test_subclass_with_all_required_attributes_is_fine():
    class GoodRegistry(ModelRegistry):
        PACKAGES = []
        TARGET_MODULE_SUFFIXES = []
        _registry = {}

    assert hasattr(GoodRegistry, "_registry")


# ---------------------------------------------------------------------------
# register() — valid metadata
# ---------------------------------------------------------------------------


def test_register_single_combination():
    @FakeRegistry.register(
        metadata=dict(
            frameworks=["fw_a"],
            model_names=["model_x"],
            tasks=["detect"],
            versions=["v1"],
        )
    )
    class DummyModel:
        pass

    key = FakeRegistry._generate_key("fw_a", "model_x", "detect", "v1", {})
    assert key in FakeRegistry._registry
    assert FakeRegistry._registry[key] is DummyModel


def test_register_multiple_frameworks_and_tasks():
    @FakeRegistry.register(
        metadata=dict(
            frameworks=["fw_a", "fw_b"],
            model_names=["model_x"],
            tasks=["detect", "segment"],
            versions=["v1"],
        )
    )
    class DummyModel:
        pass

    for fw in ("fw_a", "fw_b"):
        for task in ("detect", "segment"):
            key = FakeRegistry._generate_key(fw, "model_x", task, "v1", {})
            assert key in FakeRegistry._registry


def test_register_with_info_field():
    @FakeRegistry.register(
        metadata=dict(
            frameworks=["fw_a"],
            model_names=["model_x"],
            tasks=["detect"],
            versions=["v1"],
            info={"extra": "value"},
        )
    )
    class DummyModel:
        pass

    key = FakeRegistry._generate_key("fw_a", "model_x", "detect", "v1", {"extra": "value"})
    assert key in FakeRegistry._registry


# ---------------------------------------------------------------------------
# register() — invalid metadata
# ---------------------------------------------------------------------------


def test_register_missing_frameworks_raises():
    with pytest.raises((TypeError, ValueError)):

        @FakeRegistry.register(
            metadata=dict(
                model_names=["model_x"],
                tasks=["detect"],
                versions=["v1"],
            )
        )
        class DummyModel:
            pass


def test_register_non_list_field_raises():
    with pytest.raises(TypeError):

        @FakeRegistry.register(
            metadata=dict(
                frameworks="fw_a",  # should be a list
                model_names=["model_x"],
                tasks=["detect"],
                versions=["v1"],
            )
        )
        class DummyModel:
            pass


def test_register_empty_list_field_raises():
    with pytest.raises(ValueError):

        @FakeRegistry.register(
            metadata=dict(
                frameworks=[],  # empty list not allowed
                model_names=["model_x"],
                tasks=["detect"],
                versions=["v1"],
            )
        )
        class DummyModel:
            pass


# ---------------------------------------------------------------------------
# register() — duplicate key (should raise)
# ---------------------------------------------------------------------------


def test_register_duplicate_key_raises():
    @FakeRegistry.register(
        metadata=dict(
            frameworks=["fw_a"],
            model_names=["model_x"],
            tasks=["detect"],
            versions=["v1"],
        )
    )
    class FirstModel:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @FakeRegistry.register(
            metadata=dict(
                frameworks=["fw_a"],
                model_names=["model_x"],
                tasks=["detect"],
                versions=["v1"],
            )
        )
        class SecondModel:
            pass

    # First registration stays in place
    key = FakeRegistry._generate_key("fw_a", "model_x", "detect", "v1", {})
    assert FakeRegistry._registry[key] is FirstModel


# ---------------------------------------------------------------------------
# _generate_key() — case normalisation
# ---------------------------------------------------------------------------


def test_generate_key_is_case_insensitive():
    key1 = FakeRegistry._generate_key("FW_A", "MODEL_X", "DETECT", "v1", {})
    key2 = FakeRegistry._generate_key("fw_a", "model_x", "detect", "v1", {})
    assert key1 == key2


def test_generate_key_differs_by_version():
    key1 = FakeRegistry._generate_key("fw_a", "model_x", "detect", "v1", {})
    key2 = FakeRegistry._generate_key("fw_a", "model_x", "detect", "v2", {})
    assert key1 != key2


def test_generate_key_differs_by_info():
    key1 = FakeRegistry._generate_key("fw_a", "model_x", "detect", "v1", {})
    key2 = FakeRegistry._generate_key("fw_a", "model_x", "detect", "v1", {"extra": "value"})
    assert key1 != key2


# ---------------------------------------------------------------------------
# get_class() — happy path
# ---------------------------------------------------------------------------


def test_get_class_returns_registered_class():
    @FakeRegistry.register(
        metadata=dict(
            frameworks=["fw_a"],
            model_names=["model_x"],
            tasks=["detect"],
            versions=["v1"],
        )
    )
    class DummyModel:
        pass

    cls = FakeRegistry.get_class({"framework": "fw_a", "model_name": "model_x", "task": "detect", "version": "v1"})
    assert cls is DummyModel


def test_get_class_is_case_insensitive():
    @FakeRegistry.register(
        metadata=dict(
            frameworks=["fw_a"],
            model_names=["model_x"],
            tasks=["detect"],
            versions=["v1"],
        )
    )
    class DummyModel:
        pass

    cls = FakeRegistry.get_class({"framework": "FW_A", "model_name": "MODEL_X", "task": "DETECT", "version": "v1"})
    assert cls is DummyModel


def test_get_class_uses_package_and_algorithm_aliases():
    """'package' and 'algorithm' are accepted as aliases for 'framework' and 'model_name'."""

    @FakeRegistry.register(
        metadata=dict(
            frameworks=["fw_b"],
            model_names=["model_y"],
            tasks=["classify"],
            versions=["v1"],
        )
    )
    class AliasModel:
        pass

    cls = FakeRegistry.get_class({"package": "fw_b", "algorithm": "model_y", "task": "classify", "version": "v1"})
    assert cls is AliasModel


def test_get_class_uses_model_type_alias_for_task():
    @FakeRegistry.register(
        metadata=dict(
            frameworks=["fw_a"],
            model_names=["model_x"],
            tasks=["seg"],
            versions=["v1"],
        )
    )
    class DummyModel:
        pass

    cls = FakeRegistry.get_class({"framework": "fw_a", "model_name": "model_x", "model_type": "seg", "version": "v1"})
    assert cls is DummyModel


def test_get_class_defaults_version_to_v1():
    @FakeRegistry.register(
        metadata=dict(
            frameworks=["fw_a"],
            model_names=["model_x"],
            tasks=["detect"],
            versions=["v1"],
        )
    )
    class DummyModel:
        pass

    # no "version" key — should default to "v1"
    cls = FakeRegistry.get_class({"framework": "fw_a", "model_name": "model_x", "task": "detect"})
    assert cls is DummyModel


# ---------------------------------------------------------------------------
# get_class() — error cases
# ---------------------------------------------------------------------------


def test_get_class_missing_framework_raises():
    with pytest.raises(ValueError, match="framework"):
        FakeRegistry.get_class({"model_name": "model_x", "task": "detect"})


def test_get_class_missing_task_raises():
    with pytest.raises(ValueError):
        FakeRegistry.get_class({"framework": "fw_a", "model_name": "model_x"})


def test_get_class_unknown_combination_raises():
    with pytest.raises(ValueError, match="No class found"):
        FakeRegistry.get_class({"framework": "nonexistent", "model_name": "ghost", "task": "detect", "version": "v1"})


# ---------------------------------------------------------------------------
# auto_register_models() — empty / invalid packages
# ---------------------------------------------------------------------------


def test_auto_register_models_empty_packages_is_noop():
    FakeRegistry.auto_register_models()
    assert len(FakeRegistry._registry) == 0


def test_auto_register_models_bad_package_logs_warning(caplog):
    class RegistryWithBadPackage(ModelRegistry):
        PACKAGES = ["this.package.does.not.exist"]
        TARGET_MODULE_SUFFIXES = [".model"]
        _registry = {}

    with caplog.at_level(logging.WARNING):
        RegistryWithBadPackage.auto_register_models()

    assert any("this.package.does.not.exist" in record.message for record in caplog.records)
    assert len(RegistryWithBadPackage._registry) == 0


# ---------------------------------------------------------------------------
# Lazy discovery and failed-import reporting
# ---------------------------------------------------------------------------


def test_get_class_triggers_discovery_once(monkeypatch):
    class LazyRegistry(ModelRegistry):
        PACKAGES = []
        TARGET_MODULE_SUFFIXES = [".model"]
        _registry = {}

    calls = []
    original = LazyRegistry.auto_register_models

    def spy():
        calls.append(1)
        original()

    monkeypatch.setattr(LazyRegistry, "auto_register_models", spy)

    @LazyRegistry.register(metadata=dict(frameworks=["fw"], model_names=["m"], tasks=["t"], versions=["v1"]))
    class Dummy:
        pass

    LazyRegistry.get_class({"framework": "fw", "model_name": "m", "task": "t"})
    LazyRegistry.get_class({"framework": "fw", "model_name": "m", "task": "t"})
    assert len(calls) == 1, "Discovery should run once, on the first lookup"


def test_failed_import_is_recorded_and_surfaced_in_lookup_error():
    class RegistryWithBadPackage(ModelRegistry):
        PACKAGES = ["this.package.does.not.exist"]
        TARGET_MODULE_SUFFIXES = [".model"]
        _registry = {}

    with pytest.raises(ValueError, match="failed to import"):
        RegistryWithBadPackage.get_class({"framework": "fw", "model_name": "m", "task": "t"})
    assert "this.package.does.not.exist" in RegistryWithBadPackage._failed_imports
