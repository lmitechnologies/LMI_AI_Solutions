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
    # Optional informational tag. GoFactory supplies it; local/static manifests may omit it.
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


class ClassificationConfigs(BaseModel):
    """Configs for Classification."""

    model_config = ConfigDict(extra="ignore")


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


class ClassificationModel(_ModelBase):
    # only used for static models
    model_type: Literal["Classification"]
    configs: ClassificationConfigs


Model = Annotated[
    Union[ODModel, ADModel, ClassificationModel],
    Field(discriminator="model_type"),
]


class ModelCollectionV3(BaseModel):
    """Schema for model version 3."""

    model_config = ConfigDict(protected_namespaces=())

    models: Dict[str, Model]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCollectionV3":
        return cls.model_validate({"models": {k: v for k, v in data.items() if v is not None}})

    def get_metadata(self) -> Dict[str, Any]:
        return {role: model.get_metadata() for role, model in self.models.items()}

    def get_global_preprocessing(self) -> Dict[str, List[Dict[str, Any]]]:
        supported = {"resize", "tile"}
        ignored = {"crop-to-label"}
        tiling_keys = {"height", "width", "x_stride", "y_stride"}

        out: Dict[str, List[Dict[str, Any]]] = {}
        for role, model in self.models.items():
            ops: List[Dict[str, Any]] = []
            for step in model.details.preprocessing:
                if step.type in ignored:
                    continue
                if step.type not in supported:
                    raise ValueError(f"Unsupported type '{step.type}'.")
                if step.type == "tile":
                    if not tiling_keys.issubset(step.configuration.keys()):
                        raise ValueError(f"Tiling configuration must contain keys: {tiling_keys}.")
                    cfg = step.configuration
                    entry: Dict[str, Any] = {
                        "type": "tile",
                        "configuration": {
                            "tile_size": [cfg["height"], cfg["width"]],
                            "stride": [cfg["y_stride"], cfg["x_stride"]],
                        },
                    }
                else:
                    entry = step.model_dump(exclude={"id"}, exclude_none=True)

                if step.id is not None:
                    entry["id"] = step.id
                ops.append(entry)

            out[role] = ops
        return out
