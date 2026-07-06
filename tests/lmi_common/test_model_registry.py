import sys
import types

import pytest

import lmi_common.model_registry as model_registry_module
from lmi_common.model_registry import DuplicateRegistrationError, ModelRegistry

# ---------------------------------------------------------------------------
# Fixtures: an importable fake backend module and a registry factory
# ---------------------------------------------------------------------------

FAKE_MODULE = "fake_backend_module_for_registry_tests"


class DummyModel:
    pass


class OtherModel:
    pass


@pytest.fixture(autouse=True)
def fake_backend_module():
    """Install an importable fake module exposing DummyModel / OtherModel."""
    module = types.ModuleType(FAKE_MODULE)
    module.DummyModel = DummyModel
    module.OtherModel = OtherModel
    sys.modules[FAKE_MODULE] = module
    yield
    sys.modules.pop(FAKE_MODULE, None)


def make_registry(*entries):
    """Define a fresh ModelRegistry subclass with the given BACKENDS entries."""
    return type("TestRegistry", (ModelRegistry,), {"BACKENDS": list(entries)})


def entry(**overrides):
    """A valid BACKENDS entry, with fields overridable per test."""
    base = {
        "frameworks": ["fw_a"],
        "model_names": ["model_x"],
        "tasks": ["detect"],
        "versions": ["v1"],
        "class_path": f"{FAKE_MODULE}:DummyModel",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# __init_subclass__ enforcement
# ---------------------------------------------------------------------------


def test_subclass_missing_backends_raises():
    with pytest.raises(TypeError, match="must define"):

        class BadRegistry(ModelRegistry):
            pass  # missing BACKENDS


def test_subclass_with_backends_is_fine():
    registry = make_registry()
    assert registry._key_map == {}


# ---------------------------------------------------------------------------
# BACKENDS expansion — valid entries
# ---------------------------------------------------------------------------


def test_expand_single_combination():
    registry = make_registry(entry())
    key = registry._generate_key("fw_a", "model_x", "detect", "v1")
    assert registry._key_map == {key: f"{FAKE_MODULE}:DummyModel"}


def test_expand_multiple_frameworks_and_tasks():
    registry = make_registry(entry(frameworks=["fw_a", "fw_b"], tasks=["detect", "segment"]))
    for fw in ("fw_a", "fw_b"):
        for task in ("detect", "segment"):
            assert registry._generate_key(fw, "model_x", task, "v1") in registry._key_map


# ---------------------------------------------------------------------------
# BACKENDS expansion — invalid entries
# ---------------------------------------------------------------------------


def test_entry_missing_frameworks_raises():
    bad = entry()
    del bad["frameworks"]
    with pytest.raises((TypeError, ValueError)):
        make_registry(bad)


def test_entry_non_list_field_raises():
    with pytest.raises(TypeError):
        make_registry(entry(frameworks="fw_a"))  # should be a list


def test_entry_empty_list_field_raises():
    with pytest.raises(ValueError):
        make_registry(entry(frameworks=[]))


def test_entry_missing_class_path_raises():
    bad = entry()
    del bad["class_path"]
    with pytest.raises(ValueError, match="class_path"):
        make_registry(bad)


def test_entry_class_path_without_class_name_raises():
    with pytest.raises(ValueError, match="class_path"):
        make_registry(entry(class_path="just.a.module"))


# ---------------------------------------------------------------------------
# BACKENDS expansion — duplicate keys (should raise at class definition)
# ---------------------------------------------------------------------------


def test_duplicate_key_across_entries_raises():
    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        make_registry(entry(), entry(class_path=f"{FAKE_MODULE}:OtherModel"))


def test_duplicate_key_with_same_class_path_still_raises():
    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        make_registry(entry(), entry())


def test_overlapping_entries_collide_on_shared_combination():
    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        make_registry(
            entry(tasks=["detect", "segment"]),
            entry(tasks=["segment"], class_path=f"{FAKE_MODULE}:OtherModel"),
        )


# ---------------------------------------------------------------------------
# _generate_key() — case normalisation
# ---------------------------------------------------------------------------


def test_generate_key_is_case_insensitive():
    key1 = ModelRegistry._generate_key("FW_A", "MODEL_X", "DETECT", "v1")
    key2 = ModelRegistry._generate_key("fw_a", "model_x", "detect", "v1")
    assert key1 == key2


def test_generate_key_differs_by_version():
    key1 = ModelRegistry._generate_key("fw_a", "model_x", "detect", "v1")
    key2 = ModelRegistry._generate_key("fw_a", "model_x", "detect", "v2")
    assert key1 != key2


# ---------------------------------------------------------------------------
# get_class() — happy path
# ---------------------------------------------------------------------------


def test_get_class_returns_registered_class():
    registry = make_registry(entry())
    cls = registry.get_class({"framework": "fw_a", "model_name": "model_x", "task": "detect", "version": "v1"})
    assert cls is DummyModel


def test_get_class_is_case_insensitive():
    registry = make_registry(entry())
    cls = registry.get_class({"framework": "FW_A", "model_name": "MODEL_X", "task": "DETECT", "version": "v1"})
    assert cls is DummyModel


def test_get_class_uses_package_and_algorithm_aliases():
    """'package' and 'algorithm' are accepted as aliases for 'framework' and 'model_name'."""
    registry = make_registry(entry(frameworks=["fw_b"], model_names=["model_y"], tasks=["classify"]))
    cls = registry.get_class({"package": "fw_b", "algorithm": "model_y", "task": "classify", "version": "v1"})
    assert cls is DummyModel


def test_get_class_uses_model_type_alias_for_task():
    registry = make_registry(entry(tasks=["seg"]))
    cls = registry.get_class({"framework": "fw_a", "model_name": "model_x", "model_type": "seg", "version": "v1"})
    assert cls is DummyModel


def test_get_class_defaults_version_to_v1():
    registry = make_registry(entry())
    # no "version" key — should default to "v1"
    cls = registry.get_class({"framework": "fw_a", "model_name": "model_x", "task": "detect"})
    assert cls is DummyModel


def test_get_class_imports_backend_module_once(monkeypatch):
    registry = make_registry(entry())
    imported = []
    original = model_registry_module.importlib.import_module

    def counting_import(name):
        imported.append(name)
        return original(name)

    monkeypatch.setattr(model_registry_module.importlib, "import_module", counting_import)

    lookup = {"framework": "fw_a", "model_name": "model_x", "task": "detect"}
    assert registry.get_class(lookup) is DummyModel
    assert registry.get_class(lookup) is DummyModel
    assert imported == [FAKE_MODULE], "The backend module should be imported once and the class cached"


# ---------------------------------------------------------------------------
# get_class() — error cases
# ---------------------------------------------------------------------------


def test_get_class_missing_framework_raises():
    registry = make_registry(entry())
    with pytest.raises(ValueError, match="framework"):
        registry.get_class({"model_name": "model_x", "task": "detect"})


def test_get_class_missing_task_raises():
    registry = make_registry(entry())
    with pytest.raises(ValueError):
        registry.get_class({"framework": "fw_a", "model_name": "model_x"})


def test_get_class_unknown_combination_raises_and_lists_available_keys():
    registry = make_registry(entry())
    with pytest.raises(ValueError, match="No backend registered") as exc_info:
        registry.get_class({"framework": "nonexistent", "model_name": "ghost", "task": "detect", "version": "v1"})
    assert "fw_a" in str(exc_info.value), "Available keys should be listed in the error"


def test_get_class_import_failure_raises_with_module_name():
    registry = make_registry(entry(class_path="this.module.does.not.exist:Ghost"))
    with pytest.raises(ImportError, match="this.module.does.not.exist"):
        registry.get_class({"framework": "fw_a", "model_name": "model_x", "task": "detect"})


def test_get_class_bad_class_name_raises():
    registry = make_registry(entry(class_path=f"{FAKE_MODULE}:DoesNotExist"))
    with pytest.raises(ImportError, match="no attribute"):
        registry.get_class({"framework": "fw_a", "model_name": "model_x", "task": "detect"})
