import importlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class DuplicateRegistrationError(ValueError):
    """Two ``BACKENDS`` entries expand to the same lookup key — a code bug, raised when the registry class is defined."""


class ModelRegistry:
    """Base class for metadata-driven model registries.

    Subclasses declare every backend in a ``BACKENDS`` table. The table is
    expanded and validated at class definition time, so duplicate lookup keys
    raise ``DuplicateRegistrationError`` as soon as the registry module is
    imported — without importing any backend.

    Usage::

        class MyRegistry(ModelRegistry):
            BACKENDS = [
                {
                    "frameworks": ["my_fw"],
                    "model_names": ["my_model"],
                    "tasks": ["detect"],
                    "versions": ["v1"],
                    "class_path": "my_package.backend_a.model:MyBackend",
                },
            ]

    ``class_path`` is a ``"module:ClassName"`` import string; registry modules
    must not import backends directly so the table stays checkable in any
    environment. ``get_class`` imports only the single module that provides the
    requested backend and caches the resolved class.
    """

    BACKENDS: List[Dict[str, Any]] = []
    _key_map: Dict[Tuple[str, str, str, str, str], str] = {}
    _class_cache: Dict[Tuple[str, str, str, str, str], Type] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "BACKENDS" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define: BACKENDS")
        cls._class_cache = {}
        cls._key_map = cls._expand_backends()

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
    def _expand_backends(cls) -> Dict[Tuple[str, str, str, str, str], str]:
        """Expand ``BACKENDS`` into a lookup-key → class_path map, validating entries and rejecting duplicate keys."""
        key_map: Dict[Tuple[str, str, str, str, str], str] = {}
        for entry in cls.BACKENDS:
            frameworks = entry.get("frameworks")
            model_names = entry.get("model_names")
            tasks = entry.get("tasks")
            versions = entry.get("versions")
            info: Dict[str, Any] = entry.get("info", {})
            class_path = entry.get("class_path")

            if not all(isinstance(field, list) for field in (frameworks, model_names, tasks, versions)):
                raise TypeError(f"{cls.__name__}: 'frameworks', 'model_names', 'tasks', and 'versions' must be lists in entry: {entry}")
            if not all([frameworks, model_names, tasks, versions]):
                raise ValueError(
                    f"{cls.__name__}: 'frameworks', 'model_names', 'tasks', and 'versions' must be non-empty in entry: {entry}"
                )
            if not isinstance(class_path, str) or ":" not in class_path:
                raise ValueError(f"{cls.__name__}: 'class_path' must be a 'module:ClassName' string in entry: {entry}")

            for framework in frameworks:
                for model_name in model_names:
                    for task in tasks:
                        for version in versions:
                            key = cls._generate_key(framework, model_name, task, version, info)
                            if key in key_map:
                                raise DuplicateRegistrationError(
                                    f"{cls.__name__}: combination already registered: "
                                    f"framework='{framework}', model_name='{model_name}', "
                                    f"task='{task}', version='{version}', info='{json.dumps(info, sort_keys=True)}' "
                                    f"points to '{key_map[key]}'. Cannot re-register with '{class_path}'."
                                )
                            key_map[key] = class_path
        return key_map

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

        wrapper_cls = cls._class_cache.get(key)
        if wrapper_cls is not None:
            return wrapper_cls

        class_path = cls._key_map.get(key)
        if class_path is None:
            available_keys = "\n".join(map(str, sorted(cls._key_map)))
            raise ValueError(
                f"No backend registered for combination: "
                f"framework='{framework.lower()}', model_name='{model_name.lower()}', "
                f"task='{task.lower()}', version='{version}', info='{json.dumps(info, sort_keys=True)}'.\n"
                f"Lookup key: {key}\n"
                f"Available keys:\n{available_keys}"
            )

        wrapper_cls = cls._resolve_class_path(class_path)
        cls._class_cache[key] = wrapper_cls
        return wrapper_cls

    @classmethod
    def _resolve_class_path(cls, class_path: str) -> Type:
        """Import the backend module named by ``class_path`` and return its class."""
        module_name, _, class_name = class_path.partition(":")
        try:
            module = importlib.import_module(module_name)
            logger.info(f"Successfully imported {module_name}")
        except Exception as e:
            raise ImportError(f"Backend module '{module_name}' failed to import: {type(e).__name__}: {e}") from e
        try:
            return getattr(module, class_name)
        except AttributeError as e:
            raise ImportError(
                f"Module '{module_name}' has no attribute '{class_name}'; check 'class_path' in {cls.__name__}.BACKENDS."
            ) from e
