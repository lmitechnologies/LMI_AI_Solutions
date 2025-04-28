import json
from typing import Type, Dict, Tuple, Any, Optional, List

class AnomalyDetectorRegistry:
    _registry: Dict[Tuple[str, str, str, str, str], Type] = {}

    @classmethod
    def _generate_key(cls, framework: str, model_name: str, task: str, version: str, info: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        sorted_info_str = json.dumps(info, sort_keys=True)
        return (
            framework.lower(),
            model_name.lower(),
            task.lower(),
            version,
            sorted_info_str
        )

    @classmethod
    def register(cls, metadata: Dict[str, Any]):
        frameworks: Optional[List[str]] = metadata.get('frameworks')
        model_names: Optional[List[str]] = metadata.get('model_names')
        tasks: Optional[List[str]] = metadata.get('tasks')
        versions: Optional[List[str]] = metadata.get('versions')
        info: Dict[str, Any] = metadata.get('info', {})

        if not all([frameworks, model_names, tasks, versions]):
            raise ValueError("Metadata must include 'frameworks', 'model_names', 'tasks', and 'versions' (all non-empty lists).")

        def decorator(wrapper_cls: Type) -> Type:
            """The actual decorator that registers the class."""
            if not isinstance(frameworks, list) or not isinstance(model_names, list) or \
               not isinstance(tasks, list) or not isinstance(versions, list):
                 raise TypeError("'frameworks', 'model_names', 'tasks', and 'versions' must be lists.")

            for framework in frameworks:
                for model_name in model_names:
                    for task in tasks:
                        for version in versions:
                            key = cls._generate_key(framework, model_name, task, version, info)
                            if key in cls._registry:
                                existing_cls = cls._registry[key]
                                raise ValueError(
                                    f"Combination already registered: "
                                    f"framework='{framework}', model_name='{model_name}', "
                                    f"task='{task}', version='{version}', info='{json.dumps(info, sort_keys=True)}' "
                                    f"points to {existing_cls.__name__}. Cannot re-register with {wrapper_cls.__name__}."
                                )
                            cls._registry[key] = wrapper_cls
            return wrapper_cls
        return decorator

    @classmethod
    def get_class(cls, metadata: Dict[str, Any]) -> Type:
        framework: Optional[str] = metadata.get('framework') or metadata.get('package')
        model_name: Optional[str] = metadata.get('model_name') or metadata.get('algorithm')
        task: Optional[str] = metadata.get('task', 'seg') or metadata.get('model_type', 'seg')
        version: str = metadata.get('version', 'v1') 
        info: Dict[str, Any] = metadata.get('info', {})

        if not all([framework, model_name, task]):
            raise ValueError("Lookup metadata must include 'framework' (or 'package'), 'model_name' (or 'algorithm'), and 'task' (or 'model_type').")

        key = cls._generate_key(framework, model_name, task, version, info)

        wrapper_cls = cls._registry.get(key)

        if wrapper_cls is None:
            available_keys = "\n".join(map(str, cls._registry.keys())) # For debugging
            raise ValueError(
                f"No class found registered for combination: "
                f"framework='{framework.lower()}', model_name='{model_name.lower()}', "
                f"task='{task.lower()}', version='{version}', info='{json.dumps(info, sort_keys=True)}'.\n"
                f"Lookup key: {key}\n"
                f"Available keys:\n{available_keys}"
            )

        return wrapper_cls
