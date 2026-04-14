class ModelFactory:
    """Mixin that provides extension-based dispatch to registered backend classes.

    Subclasses must define ``_registry = {}`` as a class attribute so each factory
    maintains its own independent mapping of file extensions to backend classes.

    Usage::

        class MyModel(ModelFactory, ODBase):
            _registry = {}

        @MyModel.register("engine")
        class MyModelTRT(ODBase): ...

        @MyModel.register("pt")
        class MyModelPT(ODBase): ...
    """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_registry" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define '_registry = {{}}'")

    @classmethod
    def register(cls, format):
        def decorator(wrapper_cls):
            cls._registry[format] = wrapper_cls
            return wrapper_cls

        return decorator

    def __new__(cls, model_path, *args, **kwargs):
        ext = model_path.split(".")[-1]
        wrapper_cls = cls._registry.get(ext)
        if wrapper_cls is None:
            raise ValueError(f"Unsupported model file extension: .{ext}")
        return wrapper_cls(model_path, *args, **kwargs)
