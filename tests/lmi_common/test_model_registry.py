import logging

import pytest

import lmi_common.model_registry as model_registry_module
from lmi_common.model_registry import DuplicateRegistrationError, ModelRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concrete registry subclass used across all tests
# ---------------------------------------------------------------------------


class FakeRegistry(ModelRegistry):
    PACKAGES = {}
    _registry = {}


# ---------------------------------------------------------------------------
# Helper: fresh registry state per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_registry():
    """Reset FakeRegistry between tests so registrations don't bleed over."""

    def reset():
        FakeRegistry._registry.clear()
        FakeRegistry._failed_imports.clear()
        FakeRegistry._attempted_imports.clear()
        FakeRegistry._discovered = False

    reset()
    yield
    reset()


# ---------------------------------------------------------------------------
# __init_subclass__ enforcement
# ---------------------------------------------------------------------------


def test_subclass_missing_all_required_attributes_raises():
    with pytest.raises(TypeError, match="must define"):

        class BadRegistry(ModelRegistry):
            pass  # missing PACKAGES, _registry


def test_subclass_missing_one_required_attribute_raises():
    with pytest.raises(TypeError, match="must define"):

        class PartialRegistry(ModelRegistry):
            PACKAGES = {}
            # missing _registry


def test_subclass_with_all_required_attributes_is_fine():
    class GoodRegistry(ModelRegistry):
        PACKAGES = {}
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

    with pytest.raises(DuplicateRegistrationError, match="already registered"):

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


def test_auto_register_models_bad_module_logs_warning(caplog):
    class RegistryWithBadModule(ModelRegistry):
        PACKAGES = {"fw": ["this.module.does.not.exist"]}
        _registry = {}

    with caplog.at_level(logging.WARNING):
        RegistryWithBadModule.auto_register_models()

    assert any("this.module.does.not.exist" in record.message for record in caplog.records)
    assert len(RegistryWithBadModule._registry) == 0


# ---------------------------------------------------------------------------
# Lazy discovery and failed-import reporting
# ---------------------------------------------------------------------------


def test_get_class_unknown_framework_triggers_full_discovery_once(monkeypatch):
    class LazyRegistry(ModelRegistry):
        PACKAGES = {}
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
    assert len(calls) == 1, "Full discovery should run once, on the first lookup"


def test_get_class_imports_only_requested_framework(monkeypatch):
    class TargetedRegistry(ModelRegistry):
        PACKAGES = {"fw_a": ["fake.module_a"], "fw_b": ["fake.module_b"]}
        _registry = {}

    imported = []

    def fake_import(name):
        imported.append(name)
        if name == "fake.module_a":

            @TargetedRegistry.register(metadata=dict(frameworks=["fw_a"], model_names=["m"], tasks=["t"], versions=["v1"]))
            class BackendA:
                pass

    monkeypatch.setattr(model_registry_module.importlib, "import_module", fake_import)

    cls = TargetedRegistry.get_class({"framework": "fw_a", "model_name": "m", "task": "t"})
    assert cls.__name__ == "BackendA"
    assert imported == ["fake.module_a"], "Only the requested framework's module should be imported"


def test_get_class_miss_falls_back_to_full_discovery(monkeypatch):
    """An incomplete PACKAGES entry self-heals: a miss after the targeted import triggers full discovery."""

    class IncompleteRegistry(ModelRegistry):
        PACKAGES = {"fw_a": ["fake.module_a"], "fw_b": ["fake.module_b"]}
        _registry = {}

    imported = []

    def fake_import(name):
        imported.append(name)
        if name == "fake.module_b":
            # fw_a's backend actually lives in fw_b's module — missing from the fw_a map entry
            @IncompleteRegistry.register(metadata=dict(frameworks=["fw_a"], model_names=["m"], tasks=["t"], versions=["v1"]))
            class BackendA:
                pass

    monkeypatch.setattr(model_registry_module.importlib, "import_module", fake_import)

    cls = IncompleteRegistry.get_class({"framework": "fw_a", "model_name": "m", "task": "t"})
    assert cls.__name__ == "BackendA"
    assert imported == ["fake.module_a", "fake.module_b"]


def test_duplicate_registration_during_import_propagates(monkeypatch):
    """A duplicate key is a code bug: it must not be downgraded to a recorded import failure."""

    class CollidingRegistry(ModelRegistry):
        PACKAGES = {"fw": ["fake.colliding_module"]}
        _registry = {}

    def fake_import(name):
        @CollidingRegistry.register(metadata=dict(frameworks=["fw"], model_names=["m"], tasks=["t"], versions=["v1"]))
        class First:
            pass

        @CollidingRegistry.register(metadata=dict(frameworks=["fw"], model_names=["m"], tasks=["t"], versions=["v1"]))
        class Second:
            pass

    monkeypatch.setattr(model_registry_module.importlib, "import_module", fake_import)

    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        CollidingRegistry.get_class({"framework": "fw", "model_name": "m", "task": "t"})
    assert "fake.colliding_module" not in CollidingRegistry._failed_imports


def test_failed_import_is_recorded_and_surfaced_in_lookup_error():
    class RegistryWithBadModule(ModelRegistry):
        PACKAGES = {"fw": ["this.module.does.not.exist"]}
        _registry = {}

    with pytest.raises(ValueError, match="failed to import"):
        RegistryWithBadModule.get_class({"framework": "fw", "model_name": "m", "task": "t"})
    assert "this.module.does.not.exist" in RegistryWithBadModule._failed_imports
