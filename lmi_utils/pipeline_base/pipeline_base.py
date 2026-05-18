import collections
import functools
import json
import logging
import traceback
from abc import ABCMeta, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from anomaly_detectors.ad_core.anomaly_detector import AnomalyDetector
from classifiers.cls_core.classifier import Classifier
from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    Mask,
    Point2d,
    Polygon,
)

# LMI AIS repo's modules
from lmi_utils.image_utils.types import ImageBatch, ImageLike
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor
from object_detectors.od_core.object_detector import ObjectDetector

from .core.schemas.schema_2 import ModelCollectionV2
from .core.schemas.schema_3 import ModelCollectionV3


class PipelineBase(metaclass=ABCMeta):
    logger = logging.getLogger(__name__)

    # Maps prediction keys to the specific classes and types needed for annotation.
    # This is used for uploading labels to label studio.
    _PREDICTION_HANDLERS = {
        "boxes": {
            "value_factory": lambda v: Box(
                x_min=v[0],
                y_min=v[1],
                x_max=v[2],
                y_max=v[3],
                angle=v[4] if len(v) > 4 else 0,
            ),
            "type": AnnotationType.BOX.value,
        },
        "polygons": {
            "value_factory": lambda v: Polygon(points=v),
            "type": AnnotationType.POLYGON.value,
        },
        "masks": {
            "value_factory": lambda v: Mask(mask=v),
            "type": AnnotationType.MASK.value,
        },
        "keypoints": {
            "value_factory": lambda v: Point2d(x=v[0], y=v[1]),
            "type": AnnotationType.KEYPOINT.value,
        },
    }

    # Maps gadget version to model_roles handler
    _MODEL_ROLES_HANDLERS = {
        "1": None,  # no longer supported
        "2": ModelCollectionV2.from_dict,
        "3": ModelCollectionV3.from_dict,
    }

    def __init__(self, **kwargs: Any) -> None:
        """
        init the pipeline.
        it has the following attributes:

            models: a dictionary of model instances, e.g., {model_name: model_instance}
            results: a dictionary of the results, e.g.,
                {'outputs':{}, 'automation_keys':[], 'factory_keys':[], 'tags':[], 'should_archive':True, 'decision':None}
            _preprocessing: a dictionary of global preprocessing configs for each model role.
            version: the gadget version. It determines which model_roles handler to be used.
            preprocessor: an instance of Preprocessor class for preprocessing inputs.
            reconstructor: an instance of Reconstructor class for reconstructing outputs.
        """
        self.models = collections.OrderedDict()
        self._preprocessing = collections.OrderedDict()
        self.version = kwargs.get("version", "3")
        self.preprocessor = Preprocessor()
        self.reconstructor = Reconstructor()
        self.init_results()

    def _load_model(self, model_name: str, metadata: dict, **kwargs: Any) -> None:
        """load a model with the given metadata.

        Args:
            model_name (str): the name of the model to be loaded.
            metadata (dict): the metadata of the model to be loaded.
            kwargs (dict): additional arguments to be passed to the model constructor.
        """
        required_keys = ["model_path", "image_size", "model_type", "package"]
        missing_keys = [key for key in required_keys if not metadata.get(key)]
        if missing_keys:
            raise ValueError(f"Missing required metadata keys {missing_keys} for model: {model_name}")

        self.logger.info(f"Loading {model_name} from {metadata['model_path']}")

        meta_copy = metadata.copy()
        model_type = meta_copy["model_type"].lower()

        # Default to v1, but downgrade to v0 for specific cases
        meta_copy.setdefault("version", "v1")
        if meta_copy["package"] == "detectron2":
            meta_copy["version"] = "v0"

        model_classes: Dict[str, Type] = {
            "anomalydetection": AnomalyDetector,
            "classification": Classifier,
            "objectdetection": ObjectDetector,
            "instancesegmentation": ObjectDetector,
            "keypointdetection": ObjectDetector,
            "orientedobjectdetection": ObjectDetector,
        }
        model_class = model_classes.get(model_type)

        if not model_class:
            raise ValueError(f"model_type '{model_type}' is not supported")

        self.models[model_name] = model_class(meta_copy, **kwargs)

    def _parse_model_roles(self, model_roles: dict, **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Parse model_roles by version and convert it to match the required format for initializing AIS repo models.
        Also, it loads the global preprocessing steps.

        Args:
            model_roles (dict): the model roles to parse.
            **kwargs (Any): additional arguments to be passed to the model constructor.

        Raises:
            ValueError: If the version is not supported.

        Returns:
            dict: The parsed model roles.
            dict: The global preprocessing steps.
        """
        version = kwargs.get("version", self.version)

        # Validate version
        if version not in self._MODEL_ROLES_HANDLERS:
            raise ValueError(f"Unsupported version: {version}")

        handler = self._MODEL_ROLES_HANDLERS[version]

        # Version 1: No longer supported
        if handler is None:
            raise ValueError(f"Gadget version {version} is no longer supported. Please upgrade to the latest version.")

        # Version 2+: Supports global preprocessing
        instance = handler(model_roles)
        return instance.get_metadata(), instance.get_global_preprocessing()

    def load_models(self, model_roles: dict, configs: dict, filter: str = "-model", **kwargs: Any) -> None:
        """
        Load multiple models based on the provided model_roles, configs and filter. It also loads the global preprocessing for the models.
        The model_roles are used for loading models from the GoFactory or static models.
        The configs are used for loading pipeline configs from pipeline_def.json or the runtime.
        It also filters out not relevant models based on the provided filter string.

        Args:
            model_roles (dict): a dictionary from gofactory or static_models.
            configs (dict): the configs from pipeline_def.json or the runtime.
            filter (str, optional): filter models by name. Defaults to "-model".
        kwargs:
            verbose (bool, optional): whether to log the original and parsed model roles. Defaults to False.
        """
        parsed_model_roles, global_preprocessing = self._parse_model_roles(model_roles, **kwargs)
        if not global_preprocessing:
            raise ValueError("Global preprocessing is not defined in model roles.")

        if kwargs.get("verbose", False):
            self.logger.info(f"Original Model Roles: {json.dumps(model_roles, indent=2)}\n")
            self.logger.info(f"Parsed Model Roles: {json.dumps(parsed_model_roles, indent=2)}\n")
            self.logger.info(f"Global Preprocessing: {json.dumps(global_preprocessing, indent=2)}\n")

        # filter configs to get target model keys
        target_model_keys = [k for k in model_roles.keys() if filter in k]
        for model_key in target_model_keys:
            config_to_use = parsed_model_roles.get(model_key)
            if config_to_use is None:
                self.logger.warning(f"Not found '{model_key}' configs in parsed model roles. Skipping.")
                continue

            model_source = "Static" if "static" in Path(config_to_use["model_path"]).parts else "GoFactory"
            self._load_model(model_key, config_to_use, **kwargs)

            if model_key in global_preprocessing:
                self._preprocessing[model_key] = global_preprocessing[model_key]
                self.logger.info(f"Loaded global preprocessing for '{model_key}'")
            else:
                raise ValueError(
                    f"Global preprocessing is enabled but no preprocessing config found for '{model_key}'. "
                    f"Add preprocessing config in Gadget or static manifest."
                )

            self.logger.info(f"Successfully loaded {model_source} model: {model_key}\n")
        self.logger.info(f"Final loaded models: {list(self.models.keys())}\n")

    def preprocess(
        self,
        model_role: str,
        images: ImageBatch,
        runtime: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[List[ImageLike], List[Dict[str, Any]]]:
        """preprocess the image(s) based on the preprocessing steps in model_role.

        Pairs with revert_preprocess() as its inverse.

        Args:
            model_role: the model role to be used for preprocessing.
            images: the image(s) to be preprocessed.
            runtime: optional `{id: value}` dict for ops that need caller-supplied data. Keys must match the `id` of a step in manifest.
                Supported values by op type:
                  - "crop-to-label": {"boxes": [[x1, y1, x2, y2], ...]} —
                    per-image boxes in original-image coordinates supplied by an upstream detector.

        Returns:
            list[ImageLike]: the preprocessed image(s).
            list[dict]: the preprocessing steps, each with keys:
                - "type" (str): the type of the preprocessing operation.
                - "metadata" (list): the metadata returned by the preprocessing operation, used for reconstruction.
                - "id" (str, optional): preserved from the manifest step.
        """
        if model_role not in self._preprocessing:
            raise ValueError(f"Not found global preprocessing steps for model role: {model_role}")

        return self.preprocessor.preprocess(images, self._preprocessing[model_role], runtime=runtime)

    def revert_preprocess(self, data, ops: List[Dict[str, Any]]):
        """Invert preprocessing transforms on either image data (AD) or detection coordinates (OD).

        Dispatches based on the type of ``data``:
        - ``list`` → invert spatial transforms on images (AD path, including anomaly score maps).
        - ``dict`` → revert coordinates to original image space (OD path).

        Pairs with ``preprocess()`` as its inverse.

        Args:
            data: Either a list of images (AD) or a batch results dict with keys
                  boxes, scores, classes, masks, segments, points (OD).
            ops (list[dict]): Preprocessing history returned by preprocess().

        Returns:
            list[ImageLike] for the AD path (reconstructed images), or
            dict for the OD path (results with coordinates reverted to original image space).
        """
        if isinstance(data, dict):
            return self.reconstructor.reconstruct_coordinates(data, ops)
        elif isinstance(data, list):
            return self.reconstructor.reconstruct_images(data, ops)
        else:
            raise ValueError(f"Unsupported data type for revert_preprocess: {type(data)}")

    def add_prediction(
        self,
        pred_type: str,
        value: object,
        score: float,
        label: str,
        image_height: int,
        image_width: int,
        **kwargs: Any,
    ) -> None:
        """add a single prediction to results for uploading to Label Studio.

        Args:
            pred_type (str): the type of the predictions to be updated, e.g., 'boxes', 'polygons', 'masks', 'keypoints'.
            value (object): the value of the prediction, e.g., a numpy array for masks type. a list for other types.
            score (float): the confidence score of the prediction.
            label (str): the label of the prediction.
            image_height (int): the height of the image that the prediction is made on.
            image_width (int): the width of the image that the prediction is made on.
            kwargs (dict): additional arguments to be passed to the _add_predictions method, such as 'key' and 'sub_key'.
        """
        if pred_type not in self._PREDICTION_HANDLERS:
            raise ValueError(f'Unsupported prediction type: "{pred_type}"')

        predictions = {pred_type: {"classes": [label], "objects": [value], "confidences": [score]}}
        self._add_predictions(predictions, image_height, image_width, **kwargs)

    def _add_predictions(
        self,
        predictions: dict,
        image_height: int,
        image_width: int,
        key="outputs",
        sub_key="labels",
    ) -> None:
        """a helper function to add a batch of predictions to results for Label Studio.

        Args:
            predictions (dict): a dictionary of predictions with one of these keys: boxes, polygons, masks and keypoints. e.g.,
                {"boxes": {"classes": [], "objects": [], "confidences": []}}.
            image_height (int): the height of the image.
            image_width (int): the width of the image.
            key (str, optional): the key of the self.results. Defaults to 'outputs'.
            sub_key (str, optional): the key of the sub dictionary to be updated. Defaults to 'labels'.
        """
        default_entry = {
            "type": "object",
            "format": "json",
            "extension": ".label.json",
            "content": {
                "height": image_height,
                "width": image_width,
                "predictions": [],
            },
        }
        target_dict = self.results.setdefault(key, {}).setdefault(sub_key, default_entry)

        ch, cw = target_dict["content"]["height"], target_dict["content"]["width"]
        if ch != image_height or cw != image_width:
            raise ValueError(f"Image size mismatch: {ch}x{cw} != {image_height}x{image_width}")

        prediction_list = target_dict["content"]["predictions"]
        for pred_type, handler in self._PREDICTION_HANDLERS.items():
            if pred_type in predictions:
                data = predictions[pred_type]
                self._validate_prediction_data(pred_type, data)

                for obj, cls, conf in zip(data["objects"], data["classes"], data["confidences"]):
                    value_object = handler["value_factory"](obj)
                    annotation = Annotation(
                        id=str(len(prediction_list)),
                        value=value_object,
                        label_id=cls,
                        confidence=float(conf),
                        type=handler["type"],
                    )
                    prediction_list.append(annotation.to_dict())

    def _validate_prediction_data(self, pred_type: str, data: dict) -> None:
        """validate the structure and lengths of prediction data.

        Args:
            pred_type (str): the type of the prediction.
            data (dict): the prediction data to validate.

        Raises:
            ValueError: if the structure or lengths are invalid.
        """
        required_keys = ["objects", "classes", "confidences"]
        if not all(k in data for k in required_keys):
            raise ValueError(f"Missing required keys for {pred_type}: {required_keys}")

        # Validate lengths match
        if not (len(data["classes"]) == len(data["confidences"]) == len(data["objects"])):
            raise ValueError(f"Array length of 'classes', 'confidences' and 'objects' mismatch for {pred_type}.")

    def init_results(self) -> None:
        """
        init the output results
        """
        self.results = {
            "outputs": {
                "annotated": None,
            },
            "automation_keys": [],
            "factory_keys": ["tags"],
            "tags": [],
            "should_archive": True,
            "errors": [],
        }

    @classmethod
    def track_exception(cls, logger: Optional[Logger] = None) -> Callable:
        """track exceptions and log the error message to GoFactory.

        Args:
            logger (Logger, optional): the logger to use. Defaults to logging.getLogger(__name__).
        """
        logger = logger or logging.getLogger(__name__)

        def deco(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except Exception:
                    logger.exception(f"Failed to run function: {func.__name__}")
                    if func.__name__ == "predict":
                        # upload error messages to GoFactory
                        err_msg = traceback.format_exc()
                        self.update_results("errors", err_msg, to_factory=True)
                        self.update_results("tags", "ERROR", to_factory=True)
                        self.update_results("should_archive", True)
                        return self.results
                    else:
                        raise

            return wrapper

        return deco

    @abstractmethod
    def warm_up(self, configs: dict) -> None:
        """
        warm up the pipeline
        """
        pass

    @abstractmethod
    def load(self, model_roles: dict, configs: dict) -> None:
        """
        load models
        """
        pass

    @abstractmethod
    def predict(self, configs: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        the main function to run the pipeline.
        """
        pass

    def clean_up(self) -> None:
        """
        clean up the pipeline in REVERSED order, i.e., the last models get destroyed first
        """
        while self.models:
            model_name, model = self.models.popitem(last=True)
            del model
            self.logger.info(f"{model_name} has been cleaned up")
        self.logger.info("pipeline is cleaned up")

        self.init_results()
        self._preprocessing.clear()
        self.logger.info("preprocessing is cleaned up")

    def update_results(self, key: str, value: Any, sub_key: Optional[str] = None, **kwargs: Any) -> None:
        """
        modifies self.results by applying rules for creation and updates.

        Args:
            key (str): the key of the self.results
            value (Any): the value of the key to be updated
            sub_key (str, optional): the key of sub dictionary to be updated. Defaults to None.
            to_factory (bool, optional): add the key to the gofactory. Defaults to False.
            to_automation (bool, optional): add the key to the automation. Defaults to False.
            overwrite (bool, optional): if self.results[key] is a list, overwrite it with value. Defaults to False.
        """
        # Handle appending to an existing list.
        if key in self.results and isinstance(self.results[key], list) and not kwargs.get("overwrite", False):
            self.results[key].append(value)
        elif sub_key is not None:
            self.results.setdefault(key, {})[sub_key] = value
        else:
            self.results[key] = value

        if kwargs.get("to_factory", False) and key not in self.results["factory_keys"]:
            self.results["factory_keys"].append(key)

        if kwargs.get("to_automation", False) and key not in self.results["automation_keys"]:
            self.results["automation_keys"].append(key)

    def check_return_types(self, check_sub_keys: Optional[List[str]] = None) -> bool:
        """check if the result dictionary is json serializable

        Args:
            check_sub_keys (list, optional): a list of sub keys to check in 'outputs'. Defaults to [].
                It checks the default sub key 'labels' anyway.
        """

        def is_json_serializable(obj, key):
            """Check if an object can be serialized to JSON."""
            try:
                json.dumps(obj)
                return True
            except (TypeError, OverflowError):
                self.logger.error(f"{key} is not json serializable.")
                return False

        check_sub_keys = check_sub_keys or []
        # check the default subkey in 'outputs', i.e., 'labels'
        DEFAULT_SUB_KEY = "labels"
        for k, v in self.results.items():
            if k == "outputs":
                for sub_key in set(check_sub_keys + [DEFAULT_SUB_KEY]):
                    if sub_key not in v:
                        if sub_key != DEFAULT_SUB_KEY:
                            self.logger.warning(f"{sub_key} is not found in outputs, skip checking it")
                    elif not is_json_serializable(v[sub_key], f'"{sub_key}" in "outputs"'):
                        return False
            else:
                if not is_json_serializable(v, k):
                    return False
        return True
