from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


class Artifact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_path: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)


class PreprocessStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    configuration: Dict[str, Any]
    id: Optional[str] = None


class Details(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_size: List[int] = Field(default_factory=list)
    preprocessing: List[PreprocessStep] = Field(default_factory=list)
    training_package: str = ""
    training_algorithm: str = ""
    confidence_threshold: Optional[float] = None
    classes: Optional[List[str]] = None


class ODConfigs(BaseModel):
    """Configs for ObjectDetection / InstanceSegmentation."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    to_fail: Dict[str, bool] = Field(default_factory=dict, alias="to-fail")
    confidence: Dict[str, float] = Field(default_factory=dict)


class ADConfigs(BaseModel):
    """Configs for AnomalyDetection."""

    model_config = ConfigDict(extra="ignore")

    min_threshold: float
    max_threshold: float


class _ModelBase(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model_role: str
    model_name: str
    model_version: str
    format: str
    artifacts: Dict[str, Artifact] = Field(default_factory=dict)
    details: Details

    def get_metadata(self) -> Dict[str, Any]:
        artifact = self.artifacts.get(self.format)
        return {
            "model_path": artifact.model_path if artifact else "",
            "image_size": self.details.image_size,
            "model_type": self.model_type.lower(),
            "algorithm": self.details.training_algorithm.lower(),
            "package": self.details.training_package.lower(),
        }


class ODModel(_ModelBase):
    model_type: Literal["ObjectDetection", "InstanceSegmentation"]
    configs: ODConfigs


class ADModel(_ModelBase):
    model_type: Literal["AnomalyDetection"]
    configs: ADConfigs


Model = Annotated[
    Union[ODModel, ADModel],
    Field(discriminator="model_type"),
]


class ModelCollection(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    models: Dict[str, Model]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCollection":
        return cls.model_validate({"models": {k: v for k, v in data.items() if v is not None}})

    def get_metadata(self) -> Dict[str, Any]:
        return {role: model.get_metadata() for role, model in self.models.items()}

    def get_preprocessing(self) -> Dict[str, List[Dict[str, Any]]]:
        supported = {"resize", "tile"}
        tiling_keys = {"height", "width", "x_stride", "y_stride"}

        out: Dict[str, List[Dict[str, Any]]] = {}
        for role, model in self.models.items():
            ops: List[Dict[str, Any]] = []
            for step in model.details.preprocessing:
                if step.type not in supported:
                    raise ValueError(f"Unsupported type '{step.type}'.")
                if step.type == "tile":
                    if not tiling_keys.issubset(step.configuration.keys()):
                        raise ValueError(f"Tiling configuration must contain keys: {tiling_keys}.")
                    cfg = step.configuration
                    ops.append(
                        {
                            "type": "tile",
                            "configuration": {
                                "tile_size": [cfg["height"], cfg["width"]],
                                "stride": [cfg["y_stride"], cfg["x_stride"]],
                            },
                        }
                    )
                else:
                    ops.append(step.model_dump(exclude_none=True))
            out[role] = ops
        return out


class ModelSchemaV_3:
    """Schema for model version 3."""

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ModelCollection:
        return ModelCollection.from_dict(data)

    @staticmethod
    def get_metadata(model_collection: ModelCollection) -> Dict[str, Any]:
        return model_collection.get_metadata()
