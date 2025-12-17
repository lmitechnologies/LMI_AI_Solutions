from od_core.od_base import ODBase
from od_core.object_detector_registry import ObjectDetectorRegistry
from od_core.results import Results
import gadget_utils.pipeline_utils as pipeline_utils

class RFDETR(ODBase):
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, model_path:str, device='gpu', data=None, fp16=False,**kwargs) -> None:
        self.image_size = kwargs.get('image_size', [640, 640])
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'File not found: {model_path}')
        
        self._setup_device(device)
        
        # TODO load model and support all types
        model_type = kwargs.get('model_type', 'medium').lower()
        if model_type == 'medium':
            model = RFDETRMedium(pretrain_weights=model_path)
        else:
            raise ValueError(f'Unsupported model type: {model_type}. Supported types are "medium".')
        self.model.to(self.device)
        self.model.optimize_for_inference()
        self.model.eval()
        
        
    def _setup_device(self, device):
        """set up the computation device (CPU or GPU).

        Args:
            device (str): The device to be used, either 'cpu' or 'gpu'.
        """
        if device.lower() not in ['cpu', 'gpu']:
            raise ValueError(f'Invalid device: {device}. Supported devices are "cpu" and "gpu".')
        
        self.device = torch.device('cpu')
        if device.lower() == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.logger.warning('GPU not available, falling back to CPU')
        
    def warmup(self):
        """Warm up the model by running a dummy inference."""
        self.model.infer(torch.zeros(1, 3, self.image_size[0], self.image_size[1]).to(self.device))
    
    def preprocess(self, image: np.ndarray, **kwargs):
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image in numpy array format.
        """
        pass # No preprocessing needed for RF-DETR        
    
    def forward(self, image, **kwargs) -> dict:
        """Perform object detection on a list of images.

        Args:
            image (np.ndarray): Input image in numpy array format.

        Returns:
            Results: Object containing detection results.
        """

        return self.model.infer(image)
    
    def _construct_results(self, preds, **kwargs) -> dict:
        return Results(
            boxes = preds.xyxy,
            scores = preds.confidence,
            classes = preds.class_id
        )

    def predict(self, image, configs, operators=[], iou=0.4, agnostic=False, max_det=300, **kwargs):
        """Perform object detection on a list of images.

        Args:
            image (np.ndarray): Input image in numpy array format.
            configs (dict): Configuration dictionary for confidence thresholding
            operators (list, optional): List of operators to apply. Defaults to [].
            iou (float, optional): IoU threshold for NMS. Defaults to 0.4.
            agnostic (bool, optional): Class-agnostic NMS flag. Defaults to False.
            max_det (int, optional): Maximum number of detections per image. Defaults to 300.

        Returns:
            Results: Object containing detection results.
        """
        outputs = self.forward(image, **kwargs)
        return self._construct_results(outputs)
        
        

    



    


    

