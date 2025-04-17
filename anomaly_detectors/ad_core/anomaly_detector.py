import json

class AnomalyDetector:
    _registry = {}

    @classmethod
    def register(cls,metadata):
        """
        Decorator to register a wrapper class for multiple versions, model names, and tasks.
        """
        frameworks = metadata.get('frameworks', None)
        model_names = metadata.get('model_names', None)
        tasks = metadata.get('tasks', ['seg'])
        versions = metadata.get('versions', None)
        info = metadata.get('info', {})
        
        def decorator(wrapper_cls):
            assert all([frameworks, model_names, tasks, versions]), "frameworks, model_names, tasks, and versions must be specified."
            for framework in frameworks:
                for model_name in model_names:
                    for task in tasks:
                        for version in versions:
                            key = (framework, model_name, task, version,json.dumps(info))
                            if key in cls._registry:
                                raise ValueError(f"Wrapper already registered for version={version}, model_name='{model_name}', task='{task}', framework='{framework}', info='{json.dumps(info)}'.")
                            cls._registry[key] = wrapper_cls
            return wrapper_cls
        return decorator
    
    def __new__(cls, metadata,*args, **kwargs):
        """
        Override __new__ to return the appropriate wrapper instance based on version, model_name, and task.
        """
        framework = metadata.get('framework') or metadata.get('package')
        model_name = metadata.get('model_name') or metadata.get('algorithm')
        task = metadata.get('task', 'seg') or metadata.get('model_type', 'seg')
        version = metadata.get('version', 'v1') # Default version
        info = metadata.get('info', {})
        model_path = metadata.get('model_path', None)
        
        if not all([framework.lower(), model_name.lower(), task.lower()]):
            raise ValueError("framework/package, model_name/algorithm, task/model_type must be specified.")
        
        key = (framework.lower(), model_name.lower(), task.lower(), version, json.dumps(info))
        
        if key not in cls._registry:
            raise ValueError(f"No class found for version={version}, model_name='{model_name}', task='{task}', framework='{framework}', metadata='{json.dumps(info)}'.")
        
        wrapper_cls = cls._registry[key]
        
        if model_path is not None:
            return wrapper_cls(model_path, *args, **kwargs)
        else:
            return wrapper_cls(*args, **kwargs)

