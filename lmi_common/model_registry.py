import importlib
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger(__name__)


class DuplicateRegistrationError(ValueError):
    """Two classes registered the same lookup key — a code bug, so discovery re-raises it instead of skipping."""


class ModelRegistry:
    """Base class for metadata-driven model registries.

    Subclasses must define ``PACKAGES`` and ``_registry`` as class attributes —
    enforced at class definition time.

    ``PACKAGES`` maps a lowercase framework name to the modules whose import
    triggers registration of its backends. ``get_class`` imports only the
    modules for the requested framework; on an unknown framework or a lookup
    miss it falls back to importing everything (``auto_register_models``).
    Modules that fail to import are skipped; their errors are recorded and
    included in the ``get_class`` error message on a failed lookup. The one
    exception is ``DuplicateRegistrationError``, which propagates immediately.

    Usage::

        class MyRegistry(ModelRegistry):
            PACKAGES = {"my_fw": ["my_package.backend_a.model"]}
            _registry = {}

        @MyRegistry.register({
            "frameworks": ["my_fw"],
            "model_names": ["my_model"],
            "tasks": ["detect"],
            "versions": ["v1"],
        })
        class MyBackend: ...
    """

    PACKAGES: Dict[str, List[str]] = {}
    _registry: Dict[Tuple[str, str, str, str, str], Type] = {}
    _discovered: bool = False
    _attempted_imports: Set[str] = set()
    _failed_imports: Dict[str, str] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        required = ("PACKAGES", "_registry")
        missing = [attr for attr in required if attr not in cls.__dict__]
        if missing:
            raise TypeError(f"{cls.__name__} must define: {', '.join(missing)}")
        cls._discovered = False
        cls._attempted_imports = set()
        cls._failed_imports = {}

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
                                raise DuplicateRegistrationError(
                                    f"Combination already registered: "
                                    f"framework='{framework}', model_name='{model_name}', "
                                    f"task='{task}', version='{version}', info='{json.dumps(info, sort_keys=True)}' "
                                    f"points to {existing_cls.__module__}.{existing_cls.__name__}. "
                                    f"Cannot re-register with {wrapper_cls.__module__}.{wrapper_cls.__name__}."
                                )
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

        modules = cls.PACKAGES.get(framework.lower())
        if modules is not None:
            cls._import_modules(modules)
        elif not cls._discovered:
            cls.auto_register_models()

        version: str = cls._get_version(metadata, framework)
        key = cls._generate_key(framework, model_name, task, version, info)
        wrapper_cls = cls._registry.get(key)

        if wrapper_cls is None and not cls._discovered:
            # Self-heal an incomplete PACKAGES entry, and list every known key in the error below.
            cls.auto_register_models()
            wrapper_cls = cls._registry.get(key)

        if wrapper_cls is None:
            available_keys = "\n".join(map(str, cls._registry.keys()))
            failed_note = ""
            if cls._failed_imports:
                failed_lines = "\n".join(f"  {module}: {error}" for module, error in cls._failed_imports.items())
                failed_note = f"\nNote: these backend modules failed to import and could not register:\n{failed_lines}"
            raise ValueError(
                f"No class found registered for combination: "
                f"framework='{framework.lower()}', model_name='{model_name.lower()}', "
                f"task='{task.lower()}', version='{version}', info='{json.dumps(info, sort_keys=True)}'.\n"
                f"Lookup key: {key}\n"
                f"Available keys:\n{available_keys}"
                f"{failed_note}"
            )

        return wrapper_cls

    @classmethod
    def _import_modules(cls, module_names: List[str]):
        """Import each module once to trigger registration; record failures in ``_failed_imports``.

        ``DuplicateRegistrationError`` propagates — it signals a code bug, not a missing dependency.
        """
        for module_name in module_names:
            if module_name in cls._attempted_imports:
                continue
            cls._attempted_imports.add(module_name)
            try:
                importlib.import_module(module_name)
                logger.info(f"Successfully imported {module_name}")
            except DuplicateRegistrationError:
                raise
            except Exception as e:
                logger.warning(f"Failed to import {module_name}: {e}")
                cls._failed_imports[module_name] = f"{type(e).__name__}: {e}"

    @classmethod
    def auto_register_models(cls):
        """Eagerly import every backend module listed in ``PACKAGES``."""
        all_modules = sorted({module for modules in cls.PACKAGES.values() for module in modules})
        logger.info(f"Starting auto-discovery of models in: {all_modules}")
        cls._discovered = True
        cls._import_modules(all_modules)
