import logging

import pytest

from lmi_common.model_factory import ModelFactory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal concrete base so ModelFactory.__new__ can call the real constructor
# ---------------------------------------------------------------------------


class FakeBase:
    def __init__(self, model_path, *args, **kwargs):
        self.model_path = model_path
        self.args = args
        self.kwargs = kwargs


# ---------------------------------------------------------------------------
# __init_subclass__ enforcement
# ---------------------------------------------------------------------------


def test_subclass_without_registry_raises():
    with pytest.raises(TypeError, match="_registry"):

        class BadFactory(ModelFactory):
            pass  # missing _registry = {}


def test_subclass_with_registry_is_fine():
    class GoodFactory(ModelFactory):
        _registry = {}

    assert hasattr(GoodFactory, "_registry")


# ---------------------------------------------------------------------------
# register() decorator
# ---------------------------------------------------------------------------


class ConcreteFactory(ModelFactory):
    _registry = {}


@ConcreteFactory.register("pt")
class PTBackend(FakeBase):
    pass


@ConcreteFactory.register("engine")
class EngineBackend(FakeBase):
    pass


def test_register_populates_registry():
    assert "pt" in ConcreteFactory._registry
    assert "engine" in ConcreteFactory._registry
    assert ConcreteFactory._registry["pt"] is PTBackend
    assert ConcreteFactory._registry["engine"] is EngineBackend


def test_register_returns_class_unchanged():
    class AnotherFactory(ModelFactory):
        _registry = {}

    @AnotherFactory.register("onnx")
    class OnnxBackend(FakeBase):
        pass

    assert OnnxBackend.__name__ == "OnnxBackend"


# ---------------------------------------------------------------------------
# __new__() — dispatches by file extension
# ---------------------------------------------------------------------------


def test_new_dispatches_to_pt_backend():
    instance = ConcreteFactory("mymodel.pt")
    assert type(instance) is PTBackend
    assert instance.model_path == "mymodel.pt"


def test_new_dispatches_to_engine_backend():
    instance = ConcreteFactory("mymodel.engine")
    assert type(instance) is EngineBackend
    assert instance.model_path == "mymodel.engine"


def test_new_forwards_args_and_kwargs():
    instance = ConcreteFactory("mymodel.pt", "extra_arg", key="value")
    assert instance.args == ("extra_arg",)
    assert instance.kwargs == {"key": "value"}


def test_new_uses_last_dot_segment_as_extension():
    """Path with multiple dots: only the last segment is the extension."""
    instance = ConcreteFactory("/some/path/my.model.weights.pt")
    assert type(instance) is PTBackend


# ---------------------------------------------------------------------------
# __new__() — unsupported extension
# ---------------------------------------------------------------------------


def test_new_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported model file extension"):
        ConcreteFactory("mymodel.xyz")


def test_new_no_extension_raises():
    with pytest.raises(ValueError, match="Unsupported model file extension"):
        ConcreteFactory("mymodel")


# ---------------------------------------------------------------------------
# Registry isolation: each factory subclass has its own registry
# ---------------------------------------------------------------------------


def test_registries_are_isolated():
    class FactoryA(ModelFactory):
        _registry = {}

    class FactoryB(ModelFactory):
        _registry = {}

    @FactoryA.register("pt")
    class ModelA(FakeBase):
        pass

    @FactoryB.register("engine")
    class ModelB(FakeBase):
        pass

    assert "pt" in FactoryA._registry
    assert "pt" not in FactoryB._registry
    assert "engine" in FactoryB._registry
    assert "engine" not in FactoryA._registry


# ---------------------------------------------------------------------------
# Overwriting an extension registration
# ---------------------------------------------------------------------------


def test_register_overwrites_existing_extension():
    class OverwriteFactory(ModelFactory):
        _registry = {}

    @OverwriteFactory.register("pt")
    class FirstModel(FakeBase):
        pass

    @OverwriteFactory.register("pt")
    class SecondModel(FakeBase):
        pass

    # Last registration wins
    assert OverwriteFactory._registry["pt"] is SecondModel
