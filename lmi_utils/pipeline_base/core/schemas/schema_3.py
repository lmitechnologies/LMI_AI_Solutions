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
    id: Optional[str] = None


class Details(BaseModel):
    """
    Some fields only apply to certain model types:
    - classes / confidence_threshold: OD / InstanceSegmentation
    - threshold_min / threshold_max: AnomalyDetection
    """

    model_config = ConfigDict(extra="ignore")

    image_size: List[int] = Field(default_factory=list)
    global_preprocessing: List[PreprocessStep] = Field(default_factory=list)

    training_package: str = ""
    training_algorithm: str = ""

    classes: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None

    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None


class ODConfigs(BaseModel):
    """Configs for ObjectDetection / InstanceSegmentation."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    to_fail: Dict[str, bool] = Field(default_factory=dict, alias="to-fail")
    confidence: Dict[str, float] = Field(default_factory=dict)


class ADConfigs(BaseModel):
    """Configs for AnomalyDetection."""

    model_config = ConfigDict(extra="ignore")

    threshold_min: float
    threshold_max: float


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

    @model_validator(mode="before")
    @classmethod
    def populate_configs_from_details(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        details = data.get("details", {})
        configs = dict(data.get("configs", {}))

        classes = details.get("classes", [])
        confidence_threshold = details.get("confidence_threshold")

        existing_to_fail = configs.get("to-fail", configs.get("to_fail", {}))
        existing_confidence = configs.get("confidence", {})

        to_fail = dict(existing_to_fail)
        confidence = dict(existing_confidence)

        for class_name in classes:
            # If config already explicitly says True/False, preserve it.
            to_fail.setdefault(class_name, True)

            # If config already has a per-class confidence, preserve it.
            if confidence_threshold is not None:
                confidence.setdefault(class_name, float(confidence_threshold))

        configs["to-fail"] = to_fail
        configs["confidence"] = confidence

        data["configs"] = configs
        return data


class ADModel(_ModelBase):
    model_type: Literal["AnomalyDetection"]
    configs: ADConfigs

    @model_validator(mode="before")
    @classmethod
    def populate_configs_from_details(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        details = data.get("details", {})
        configs = dict(data.get("configs", {}))

        missing = [
            key
            for key in ("threshold_min", "threshold_max")
            if details.get(key) is None
        ]

        if missing:
            raise ValueError(
                f"AnomalyDetection details must contain: {missing}"
            )

        # Details is the source of truth for AD thresholds.
        configs["threshold_min"] = float(details["threshold_min"])
        configs["threshold_max"] = float(details["threshold_max"])

        data["configs"] = configs
        return data


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

    def get_global_preprocessing(self) -> Dict[str, List[Dict[str, Any]]]:
        supported = {"resize", "tile"}
        tiling_keys = {"height", "width", "x_stride", "y_stride"}

        out: Dict[str, List[Dict[str, Any]]] = {}
        for role, model in self.models.items():
            ops: List[Dict[str, Any]] = []
            for step in model.details.global_preprocessing:
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
