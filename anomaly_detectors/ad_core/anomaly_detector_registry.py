import json
from typing import Type, Dict, Tuple, Any, Optional, List
import logging
import importlib
import pkgutil


PACKAGES = ["anomalib_lmi", "ad_core"]
TARGET_MODULE_SUFFIXES = [
    ".anomaly_model",
    ".anomaly_model2",
]  # Target suffixes to look for in the packages


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnomalyDetectorRegistry:
    _registry: Dict[Tuple[str, str, str, str, str], Type] = {}

    @classmethod
    def _generate_key(
        cls,
        framework: str,
        model_name: str,
        task: str,
        version: str,
        info: Dict[str, Any],
    ) -> Tuple[str, str, str, str, str]:
        sorted_info_str = json.dumps(info, sort_keys=True)
        return (
            framework.lower(),
            model_name.lower(),
            task.lower(),
            version,
            sorted_info_str,
        )

    @classmethod
    def register(cls, metadata: Dict[str, Any]):
        logger.debug(f"Registering class with metadata: {metadata}")
        frameworks: Optional[List[str]] = metadata.get("frameworks")
        model_names: Optional[List[str]] = metadata.get("model_names")
        tasks: Optional[List[str]] = metadata.get("tasks")
        versions: Optional[List[str]] = metadata.get("versions")
        info: Dict[str, Any] = metadata.get("info", {})

        if not all([frameworks, model_names, tasks, versions]):
            raise ValueError("Metadata must include 'frameworks', 'model_names', 'tasks', and 'versions' (all non-empty lists).")

        def decorator(wrapper_cls: Type) -> Type:
            """The actual decorator that registers the class."""
            if (
                not isinstance(frameworks, list)
                or not isinstance(model_names, list)
                or not isinstance(tasks, list)
                or not isinstance(versions, list)
            ):
                raise TypeError("'frameworks', 'model_names', 'tasks', and 'versions' must be lists.")

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
                            cls._registry[key] = wrapper_cls
            return wrapper_cls

        return decorator

    @classmethod
    def get_class(cls, metadata: Dict[str, Any]) -> Type:
        framework: Optional[str] = metadata.get("framework") or metadata.get("package")
        model_name: Optional[str] = metadata.get("model_name") or metadata.get("algorithm")
        task: Optional[str] = metadata.get("task", "seg") or metadata.get("model_type", "seg")

        info: Dict[str, Any] = metadata.get("info", {})

        if not all([framework, model_name, task]):
            raise ValueError(
                "Lookup metadata must include 'framework' (or 'package'), 'model_name' (or 'algorithm'), and 'task' (or 'model_type')."
            )
        version: str = metadata.get("version", "v0" if "0" in framework else "v1")
        key = cls._generate_key(framework, model_name, task, version, info)

        wrapper_cls = cls._registry.get(key)

        if wrapper_cls is None:
            available_keys = "\n".join(map(str, cls._registry.keys()))  # For debugging
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
        """
        Dynamically discovers and imports models to trigger registration.
        """

        logger.info("Starting auto-discovery of anomaly detector models...")
        for package_name in PACKAGES:
            package = importlib.import_module(package_name)
            package_path = package.__path__

            # search for target module in each package
            for _, module_name, _ in pkgutil.walk_packages(package_path, package_name + "."):
                if any(module_name.endswith(target) for target in TARGET_MODULE_SUFFIXES):
                    try:
                        importlib.import_module(f"{module_name}")
                        logger.info(f"Successfully imported {module_name}")
                    except ImportError as e:
                        logger.warning(f"Failed to import {module_name}: {e}")
