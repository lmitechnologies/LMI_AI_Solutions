import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated


class Artifact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_path: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)


class PreprocessStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    configuration: Dict[str, Any]
    # GoFactory always supplies it; static manifests auto-fills a unique random id.
    id: Optional[str] = None


class Details(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_size: List[int] = Field(default_factory=list)
    preprocessing: List[PreprocessStep] = Field(default_factory=list)
    training_package: str = ""
    training_algorithm: str = ""
    confidence_threshold: Optional[float] = None
    classes: Optional[List[str]] = None

    @model_validator(mode="after")
    def _fill_preprocessing_ids(self) -> "Details":
        existing = {step.id for step in self.preprocessing if step.id}
        for step in self.preprocessing:
            if step.id:
                continue
            while True:
                new_id = uuid.uuid4().hex[:8]
                if new_id not in existing:
                    break
            existing.add(new_id)
            step.id = new_id
        return self


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
        supported = {"resize", "tile", "crop-to-label", "rotate"}
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
                    entry: Dict[str, Any] = {
                        "type": "tile",
                        "configuration": {
                            "tile_size": [cfg["height"], cfg["width"]],
                            "stride": [cfg["y_stride"], cfg["x_stride"]],
                        },
                    }
                else:
                    entry = step.model_dump(exclude={"id"}, exclude_none=True)

                entry["id"] = step.id
                ops.append(entry)

            labels = [op["configuration"].get("label") for op in ops if op["type"] == "crop-to-label"]
            if len(labels) != len(set(labels)):
                raise ValueError(
                    f"Model '{role}' has duplicate crop-to-label labels: {labels}. "
                    f"Each crop-to-label 'label' must be unique within a model role."
                )
            out[role] = ops
        return out
