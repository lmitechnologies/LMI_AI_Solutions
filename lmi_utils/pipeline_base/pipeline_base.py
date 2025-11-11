import collections
import functools
import logging
import traceback
from abc import ABCMeta, abstractmethod
import json

# local module
# handle different model_roles schema according to gadget version
from .core.schemas.schema_2 import ModelSchemaV_2

# LMI AIS repo's modules
from od_core.object_detector import ObjectDetector
from ad_core.anomaly_detector import AnomalyDetector
from cls_core.classifier import Classifier
from dataset_utils.representations import Box, Polygon, Mask, Point2d, AnnotationType, Annotation


class PipelineBase(metaclass=ABCMeta):
    
    logger = logging.getLogger(__name__)
    models: collections.OrderedDict
    version: str
    results: dict
    
    # Maps prediction keys to the specific classes and types needed for annotation.
    # This is used for uploading labels to label studio.
    _PREDICTION_HANDLERS = {
        'boxes': {
            'value_factory': lambda v: Box(x_min=v[0], y_min=v[1], x_max=v[2], y_max=v[3], angle=v[4] if len(v) > 4 else 0),
            'type': AnnotationType.BOX.value
        },
        'polygons': {
            'value_factory': lambda v: Polygon(points=v),
            'type': AnnotationType.POLYGON.value
        },
        'masks': {
            'value_factory': lambda v: Mask(mask=v),
            'type': AnnotationType.MASK.value
        },
        'keypoints': {
            'value_factory': lambda v: Point2d(x=v[0], y=v[1]),
            'type': AnnotationType.KEYPOINT.value
        }
    }
    
    # Maps gadget version to model_roles handler
    _MODEL_ROLES_HANDLERS = {
        '1': None,  # currently handled by the base class
        '2': ModelSchemaV_2.from_dict,
    }
    
    
    def __init__(self, **kwargs):
        """
        init the pipeline.  
        it has the following attributes:
        
            models: a dictionary of model instances, e.g., {model_name: model_instance}
            results: a dictionary of the results, e.g., {'outputs':{}, 'automation_keys':[], 'factory_keys':[], 'tags':[], 'should_archive':True, 'decision':None}
            version: the gadget version. It determines which model_roles handler to be used.
        """
        self.models = collections.OrderedDict()
        self.version = kwargs.get('version', '2')
        self.init_results()
        
        
    def _load_model(self, model_name:str, metadata:dict, **kwargs):
        """ load a model with the given metadata. Set default image size if not provided.
        
        args:
            model_name (str): the name of the model to be loaded.
            metadata (dict): the metadata of the model to be loaded.
            kwargs (dict): additional arguments to be passed to the model constructor.
        Raises:
            ValueError: if the model_type is not supported.
        """
        if model_name in self.models:
            self.logger.info(f'{model_name} is already loaded')
        self.logger.info(f"Loading {model_name} from {metadata['model_path']}")
        
        if not metadata.get('image_size'):
            raise ValueError(f'image_size is required in metadata for model: {model_name}')
        
        # add version to metadata if not provided
        model_type = metadata.get('model_type', '').lower()
        if 'version' not in metadata:
            metadata['version'] = 'v1'
        if metadata.get('package') == 'detectron2':
            # detectron2 only has v0 models
            metadata['version'] = 'v0'
        if model_type == 'classification':
            # classification only has v0 models
            metadata['version'] = 'v0'
        
        # check model type
        if model_type == 'anomalydetection':
            self.models[model_name] = AnomalyDetector(metadata, **kwargs)
        elif model_type in ['objectdetection', 'instancesegmentation', 'keypointdetection', 'orientedobjectdetection']:
            self.models[model_name] = ObjectDetector(metadata, **kwargs)
        elif model_type == 'classification':
            self.models[model_name] = Classifier(metadata, **kwargs)
        else:
            raise ValueError(f'model_type {model_type} is not supported')
        
        
    def _parse_model_roles(self, model_roles: dict, **kwargs):
        """parse model_roles by version and convert it to match the required format for initializing AIS repo models.

        Args:
            model_roles (dict): the model roles to parse.

        Raises:
            ValueError: If the version is not supported.

        Returns:
            dict: The parsed model roles.
        """
        version = kwargs.get('version', self.version)
        if version not in self._MODEL_ROLES_HANDLERS:
            raise ValueError(f'Unsupported version: {version}. Supported versions are: {list(self._MODEL_ROLES_HANDLERS.keys())}')
        handler = self._MODEL_ROLES_HANDLERS[version]
        if handler is None:
            return model_roles
        else:
            return handler(model_roles).get_metadata()
        
        
    def load_models(self, model_roles: dict, configs: dict, filter: str = '-model', **kwargs):
        """load multiple models based on the provided model_roles, configs and filter.  
        The model_roles are used for loading models from the GoFactory, while the configs are used for local models.  
        This function loads models from the GoFactory if their "use_factory" flags are set to true in the configs.  
        Otherwise, it uses local models defined in the pipeline_def.json.  
        It also filters out not relevant models based on the provided filter string.

        Args:
            model_roles (dict): a dictionary from gofactory or static_models.
            configs (dict): the configs from pipeline_def.json or the runtime.
            filter (str, optional): filter models by name. Defaults to '-model'.
        """
        
        # the format of model_roles from factory is:
        # {
        #     "top-od-model": {
        #         "format": "pt",
        #         "configs": {},
        #         "details": {
        #             "deployed": "2025-08-18T02:26:53.063Z",
        #             "baseModel": "yolov8m.pt",
        #             "trainingPackage": "Ultralytics8",
        #             "trainingAlgorithm": "Yolo",
        #             "confidenceThreshold": 0.5,
        #             "globalPreprocessing": [
        #                 {
        #                     "type": "resize",
        #                     "configuration": {
        #                         "width": 640,
        #                         "height": 640,
        #                         "preserveAspect": true
        #                     }
        #                 }
        #             ]
        #         },
        #         "artifacts": {
        #             "pt": {
        #                 "imageSize": [],
        #                 "model_path": "/app/models/top-od-model/ObjectDetection/yolo/1/model.pt"
        #             }
        #         },
        #         "model_name": "yolo",
        #         "model_role": "top-od-model",
        #         "model_type": "ObjectDetection",
        #         "model_version": "1"
        #     }
        # }
        
        # However, the required format for initializing AIS repo models is:
        # {
        #     "name": "foreground-od",
        #     "default_value": {
        #         "use_factory": false,
        #         "metadata":{
        #             "version": "v1",
        #             "model_name": "yolov8",
        #             "model_type": "instancesegmentation",
        #             "framework": "ultralytics",
        #             "image_size": [640, 640],
        #             "model_path": "/home/gadget/pipeline/trt-engines/yolo11n-seg.pt"
        #         },
        #         "iou": 0.45,
        #         "object_configs": {
        #             "person": {"confidence": 0.5},
        #             "bicycle": {"confidence": 0.5},
        #         }
        #     }
        # }
        
        # parse model_roles to match the required format for initializing AIS repo models
        parsed_model_roles = self._parse_model_roles(model_roles, **kwargs)
        self.logger.info(f'Original Model Roles: {model_roles}\n')
        self.logger.info(f'Parsed Model Roles: {parsed_model_roles}\n')

        # filter configs to get target model keys
        target_model_keys = [k for k in model_roles.keys() if f'{filter}' in k]
        for model_key in target_model_keys:
            config_to_use = parsed_model_roles[model_key]
            model_source = "Static" if 'static' in config_to_use['model_path'].split('/') else "GoFactory"

            # Add initialization artifacts not parsed by Schema (tile, stride for AD)
            model_format = model_roles[model_key]["format"]
            expected_artifacts = model_roles[model_key]["artifacts"][model_format]
            for init_artifact in expected_artifacts:
                if init_artifact not in config_to_use:
                    config_to_use[init_artifact] = expected_artifacts[init_artifact]
            
            self._load_model(model_key, config_to_use, **kwargs)
            self.logger.info(f'Successfully loaded {model_source} model: {model_key}\n')
        self.logger.info(f'Final loaded models: {list(self.models.keys())}\n')
        
    
    def add_prediction(self, pred_type:str, value:object, score:float, label:str, image_height:int, image_width:int, **kwargs):
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
        
        predictions = {
            pred_type: {
                'classes': [label],
                'objects': [value],
                'confidences': [score]
            }
        }
        self._add_predictions(predictions, image_height, image_width, **kwargs)
        
        
    def _add_predictions(self, predictions:dict, image_height:int, image_width:int, key='outputs', sub_key='labels'):
        """a helper functiomn to add a batch of predictions to results for Label Studio.
        
        Args:
            predictions (dict): a dictionary of predictions with one of these keys: boxes, polygons, masks and keypoints. e.g., {"boxes": {"classes": [], "objects": [], "confidences": []}}.
            image_height (int): the height of the image.
            image_width (int): the width of the image.
            key (str, optional): the key of the self.results. Defaults to 'outputs'.
            sub_key (str, optional): the key of the sub dictionary to be updated. Defaults to 'labels'.
        """
        default_entry = {
            'type': 'object',
            'format': 'json',
            'extension': '.label.json',
            'content': {
                'height': image_height,
                'width': image_width,
                'predictions': [],
            }
        }
        target_dict = self.results.setdefault(key, {}).setdefault(sub_key, default_entry)
        
        ch,cw = target_dict['content']['height'], target_dict['content']['width']
        if ch != image_height or cw != image_width:
            raise ValueError(f'Image size mismatch: {ch}x{cw} != {image_height}x{image_width}')
        
        prediction_list = target_dict['content']['predictions']
        for pred_type, handler in self._PREDICTION_HANDLERS.items():
            if pred_type in predictions:
                data = predictions[pred_type]
                
                for idx, value_data in enumerate(data['objects']):
                    value_object = handler['value_factory'](value_data)
                    annotation = Annotation(
                        id=str(len(prediction_list)),
                        value=value_object,
                        label_id=data['classes'][idx],
                        confidence=float(data['confidences'][idx]),
                        type=handler['type']
                    )
                    prediction_list.append(annotation.to_dict())
                    
    
    def init_results(self):
        """
        init the output results
        """
        self.results = {
            "outputs": {
                "annotated": None,
            },
            "automation_keys": [],
            "factory_keys": ['tags'],
            "tags": [],
            "should_archive": True,
            "errors": [],
        }
    
    
    @classmethod
    def track_exception(cls, logger=logging.getLogger(__name__)):
        """track exceptions and log the error message to GoFactory.
        
        Args:
            logger (Logger, optional): the logger to use. Defaults to logging.getLogger(__name__).
        """
        def deco(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except Exception:
                    logger.exception(f'Failed to run function: {func.__name__}')
                    if func.__name__ == 'predict':
                        # upload error messages to GoFactory
                        err_msg = traceback.format_exc()
                        self.update_results('errors', err_msg, to_factory=True)
                        self.update_results('tags', 'ERROR', to_factory=True)
                        self.update_results('should_archive', True)
                        return self.results
            return wrapper
        return deco
    
    
    @abstractmethod
    def warm_up(self, configs: dict):
        """
        warm up the pipeline
        """
        pass
    
    
    @abstractmethod
    def load(self, model_roles:dict, configs: dict):
        """
        load models
        """
        pass
    
    
    @abstractmethod
    def predict(self, configs: dict, inputs: dict):
        """
        the main function to run the pipeline.
        """
        pass
    
    
    def clean_up(self):
        """
        clean up the pipeline in REVERSED order, i.e., the last models get destroyed first
        """
        L = list(reversed(self.models.keys())) if self.models else []
        for model_name in L:
            del self.models[model_name]
            self.logger.info(f'{model_name} has been cleaned up')
        self.models.clear()
        self.logger.info('pipeline is cleaned up')
        
    
    def update_results(self, key:str, value, sub_key=None, to_factory=False, to_automation=False, overwrite=False):
        """ 
        modifies self.results by applying rules for creation and updates.

        Args:
            key (str): the key of the self.results
            value (obj): the value of the key to be updated
            sub_key (str, optional): the key of sub dictionary to be updated. Defaults to None.
            to_factory (bool, optional): add the key to the gofactory. Defaults to False.
            to_automation (bool, optional): add the key to the automation. Defaults to False.
            overwrite (bool, optional): if self.results[key] is a list, overwrite it with value. Defaults to False.
        """
        # Handle appending to an existing list.
        if key in self.results and isinstance(self.results[key], list) and not overwrite:
            self.results[key].append(value)
        elif sub_key is not None:
            self.results.setdefault(key, {})[sub_key] = value
        else:
            self.results[key] = value
            
        if to_factory and key not in self.results['factory_keys']:
            self.results['factory_keys'].append(key)
            
        if to_automation and key not in self.results['automation_keys']:
            self.results['automation_keys'].append(key)
            
    
    def check_return_types(self, check_sub_keys=[]) -> bool:
        """check if the result dictionary is json serializable
        
        Args:
            check_sub_keys (list, optional): a list of sub keys to check in 'outputs'. Defaults to []. It checks the default sub key 'labels' anyway.
        """
        def is_json_serializable(obj, key):
            """Check if an object can be serialized to JSON."""
            try:
                json.dumps(obj)
                return True
            except (TypeError, OverflowError):
                self.logger.error(f'{key} is not json serializable.')
                return False
        
        # check the default subkey in 'outputs', i.e., 'labels'
        DEFAULT_SUB_KEY = 'labels'
        for k, v in self.results.items():
            if k == 'outputs':
                for sub_key in set(check_sub_keys + [DEFAULT_SUB_KEY]):
                    if sub_key not in v:
                        if sub_key != DEFAULT_SUB_KEY:
                            self.logger.warning(f'{sub_key} is not found in outputs, skip checking it')
                    elif not is_json_serializable(v[sub_key], f'"{sub_key}" in "outputs"'):
                        return False
            else:
                if not is_json_serializable(v,k):
                    return False
        return True
    