import importlib
import json
import logging
import pkgutil
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Base class for metadata-driven model registries.

    Subclasses must define ``PACKAGES``, ``TARGET_MODULE_SUFFIXES``, and
    ``_registry`` as class attributes — enforced at class definition time.

    Usage::

        class MyRegistry(ModelRegistry):
            PACKAGES = ["my_package.backend_a", "my_package.backend_b"]
            TARGET_MODULE_SUFFIXES = [".model"]
            _registry = {}

        @MyRegistry.register({
            "frameworks": ["my_fw"],
            "model_names": ["my_model"],
            "tasks": ["detect"],
            "versions": ["v1"],
        })
        class MyBackend: ...
    """

    PACKAGES: List[str] = []
    TARGET_MODULE_SUFFIXES: List[str] = []
    _registry: Dict[Tuple[str, str, str, str, str], Type] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        required = ("PACKAGES", "TARGET_MODULE_SUFFIXES", "_registry")
        missing = [attr for attr in required if attr not in cls.__dict__]
        if missing:
            raise TypeError(f"{cls.__name__} must define: {', '.join(missing)}")

    @classmethod
    def _generate_key(
        cls,
        framework: str,
        model_name: str,
        task: str,
        version: str,
        info: Dict[str, Any],
    ) -> Tuple[str, str, str, str, str]:
        return (
            framework.lower(),
            model_name.lower(),
            task.lower(),
            version,
            json.dumps(info, sort_keys=True),
        )

    @classmethod
    def register(cls, metadata: Dict[str, Any]):
        logger.debug(f"Registering class with metadata: {metadata}")
        frameworks: Optional[List[str]] = metadata.get("frameworks")
        model_names: Optional[List[str]] = metadata.get("model_names")
        tasks: Optional[List[str]] = metadata.get("tasks")
        versions: Optional[List[str]] = metadata.get("versions")
        info: Dict[str, Any] = metadata.get("info", {})

        if (
            not isinstance(frameworks, list)
            or not isinstance(model_names, list)
            or not isinstance(tasks, list)
            or not isinstance(versions, list)
        ):
            raise TypeError("'frameworks', 'model_names', 'tasks', and 'versions' must be lists.")

        if not all([frameworks, model_names, tasks, versions]):
            raise ValueError("Metadata must include 'frameworks', 'model_names', 'tasks', and 'versions' (all non-empty lists).")

        def decorator(wrapper_cls: Type) -> Type:
            for framework in frameworks:
                for model_name in model_names:
                    for task in tasks:
                        for version in versions:
                            key = cls._generate_key(framework, model_name, task, version, info)
                            if key in cls._registry:
                                existing_cls = cls._registry[key]
                                logger.warning(
                                    f"Combination already registered: "
                                    f"framework='{framework}', model_name='{model_name}', "
                                    f"task='{task}', version='{version}', info='{json.dumps(info, sort_keys=True)}' "
                                    f"points to {existing_cls.__module__}. Cannot re-register with {wrapper_cls.__module__}."
                                )
                                continue
                            cls._registry[key] = wrapper_cls
            return wrapper_cls

        return decorator

    @classmethod
    def _get_version(cls, metadata: Dict[str, Any], framework: str) -> str:
        """Return the version string for a registry lookup. Override to customize."""
        return metadata.get("version", "v1")

    @classmethod
    def _get_task(cls, metadata: Dict[str, Any]) -> Optional[str]:
        """Return the task string for a registry lookup. Override to customize."""
        return metadata.get("task") or metadata.get("model_type")

    @classmethod
    def get_class(cls, metadata: Dict[str, Any]) -> Type:
        framework: Optional[str] = metadata.get("framework") or metadata.get("package")
        model_name: Optional[str] = metadata.get("model_name") or metadata.get("algorithm")
        task: Optional[str] = cls._get_task(metadata)
        info: Dict[str, Any] = metadata.get("info", {})

        if not all([framework, model_name, task]):
            raise ValueError(
                "Lookup metadata must include 'framework' (or 'package'), 'model_name' (or 'algorithm'), and 'task' (or 'model_type')."
            )

        version: str = cls._get_version(metadata, framework)
        key = cls._generate_key(framework, model_name, task, version, info)
        wrapper_cls = cls._registry.get(key)

        if wrapper_cls is None:
            available_keys = "\n".join(map(str, cls._registry.keys()))
            raise ValueError(
                f"No class found registered for combination: "
                f"framework='{framework.lower()}', model_name='{model_name.lower()}', "
                f"task='{task.lower()}', version='{version}', info='{json.dumps(info, sort_keys=True)}'.\n"
                f"Lookup key: {key}\n"
                f"Available keys:\n{available_keys}"
            )

        return wrapper_cls

    @classmethod
    def auto_register_models(cls):
        """Dynamically discovers and imports modules to trigger registration."""
        logger.info(f"Starting auto-discovery of models in: {cls.PACKAGES}")
        for package_name in cls.PACKAGES:
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                logger.warning(f"Failed to import package '{package_name}': {e}. Skipping.")
                continue
            for _, module_name, _ in pkgutil.walk_packages(package.__path__, package_name + "."):
                if any(module_name.endswith(suffix) for suffix in cls.TARGET_MODULE_SUFFIXES):
                    try:
                        importlib.import_module(module_name)
                        logger.info(f"Successfully imported {module_name}")
                    except ImportError as e:
                        logger.warning(f"Failed to import {module_name}: {e}")
