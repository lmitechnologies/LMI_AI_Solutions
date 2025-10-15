from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class Artifact:
    """Represents a single model artifact type (e.g., pt, onnx, trt)."""
    model_path: str
    image_size: List[int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Artifact':
        """Creates an Artifact instance from a dictionary."""
        return cls(
            model_path=data.get("model_path", ""),
            image_size=data.get("image_size", [])
        )

@dataclass
class Details:
    """Contains model information that cannot be updated at runtime."""
    global_preprocessing: List[str]
    training_package: str
    training_algorithm: str
    base_model: str
    defect_class_list: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Details':
        """Creates a Details instance from a dictionary."""
        return cls(
            global_preprocessing=data.get("global_preprocessing", []),
            training_package=data.get("training_package", ""),
            training_algorithm=data.get("training_algorithm", ""),
            base_model=data.get("base_model", ""),
            defect_class_list=data.get("defect_class_list")
        )

@dataclass
class Configs:
    """
    Contains model information that can be updated at runtime.
    Uses Optional for fields that may not apply to all model types.
    The `**kwargs` allows for flexible handling of dynamic defect confidence keys.
    """
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    defect_confidence: Dict[str, float] = field(default_factory=dict)


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Configs':
        """
        Creates a Configs instance from a dictionary, separating known fields
        from dynamic defect confidence scores.
        """
        known_fields = {"threshold_min", "threshold_max"}
        
        # Initialize with known fields
        instance = cls(
            threshold_min=data.get("threshold_min"),
            threshold_max=data.get("threshold_max")
        )

        for key, value in data.items():
            if key not in known_fields:
                instance.defect_confidence[key] = value
        
        return instance


@dataclass
class Model:
    """Represents a complete model configuration."""
    model_role: str
    model_type: str
    model_name: str
    model_version: str
    artifacts: Dict[str, Artifact]
    details: Details
    configs: Configs
    format: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Model':
        """Creates a Model instance from a dictionary."""
        artifacts_data = data.get("artifacts", {})
        artifacts = {k: Artifact.from_dict(v) for k, v in artifacts_data.items()}

        return cls(
            model_role=data.get("model_role", ""),
            model_type=data.get("model_type", ""),
            model_name=data.get("model_name", ""),
            model_version=data.get("model_version", ""),
            artifacts=artifacts,
            details=Details.from_dict(data.get("details", {})),
            configs=Configs.from_dict(data.get("configs", {})),
            format=data.get("format", "")
        )
    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the model as a dictionary."""
        metadata = {
            "model_path": self.artifacts.get(self.format, {}).model_path if self.format in self.artifacts else "",
            "image_size": self.artifacts.get(self.format, {}).image_size if self.format in self.artifacts else [],
            "model_type": self.model_type.lower(),
            "algorithm": self.details.training_algorithm.lower(),
            "package": self.details.training_package.lower(),
        }
        return metadata

@dataclass
class ModelCollection:
    """Represents the top-level object containing all models."""
    models: Dict[str, Model]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCollection':
        """Creates a ModelCollection from the root dictionary."""
        # The root dictionary has a single key "model"
        models = {role: Model.from_dict(model_info) for role, model_info in data.items() if model_info is not None}
        return cls(models=models)
    
    def get_metadata(self) -> Dict[str, Any]:
        configs = {}
        for model in self.models.values():
            configs[model.model_role] = model.get_metadata()
        return configs

class ModelSchemaV_2:
    """Schema for model version 2."""
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ModelCollection:
        """Creates a ModelCollection instance from a dictionary."""
        return ModelCollection.from_dict(data)
    
    @staticmethod
    def get_metadata(model_collection: ModelCollection) -> Dict[str, Any]:
        """Returns the metadata of the model collection."""
        return model_collection.get_metadata()