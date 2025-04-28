import os
import logging 
from collections import OrderedDict, namedtuple
import tensorrt as trt
import torch
import numpy as np
import albumentations as A

from .base import Anomalib_Base
import gadget_utils.pipeline_utils as pipeline_utils
from ad_core.anomaly_detector_registry import AnomalyDetectorRegistry


logging.basicConfig()

PASS = 'PASS'
FAIL = 'FAIL'
MINIMUM_QUANT=1e-12

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
@AnomalyDetectorRegistry.register(metadata=dict(frameworks=['anomalib'], model_names=['patchcore', 'padim'], tasks=['seg'], versions=['v0']))
class AnomalyModel(Anomalib_Base):
    '''
    Desc: Class used for AD model inference.  
     
    Args: 
        - model_path: path to .pt file or TRT engine
    
    '''
    logger = logging.getLogger('AnomalyModel v0')
    logger.setLevel(logging.INFO)
    
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise Exception(f'Cannot find the model file: {model_path}')
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.logger.warning('GPU device unavailable. Use CPU instead.')
            self.device = torch.device('cpu')
            
        _,ext=os.path.splitext(model_path)
        self.logger.info(f"Loading model: {model_path}")
        if ext=='.engine':
            with open(model_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            self.context = model.create_execution_context()
            self.bindings = OrderedDict()
            self.output_names = []
            self.fp16 = False
            for i in range(model.num_bindings):
                name = model.get_tensor_name(i)
                dtype = trt.nptype(model.get_tensor_dtype(name))
                shape = tuple(self.context.get_tensor_shape(name))
                self.logger.info(f'binding {name} ({dtype}) with shape {shape}')
                if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    self.output_names.append(name)
                im = self.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            self.model_shape = list(shape[-2:])
            self.image_size = list(shape[-2:])
            self.inference_mode='TRT'
        elif ext=='.pt':     
            model = torch.load(model_path,map_location=self.device)["model"]
            model.eval()
            self.pt_model=model.to(self.device)
            self.pt_metadata = torch.load(model_path, map_location=self.device)["metadata"] if model_path else {}
            self.pt_transform=A.from_dict(self.pt_metadata["transform"])
            for d in self.pt_metadata['transform']['transform']['transforms']:
                if d['__class_fullname__']=='Resize':
                    self.model_shape = [d['height'], d['width']]
                    self.image_size = [d['height'], d['width']]
            self.inference_mode='PT'
        else:
            raise Exception(f'Unknown model format: {ext}')
        
        
    @torch.inference_mode()
    def normalize(self,image: np.ndarray) -> np.ndarray:
        """
        Desc: Normalize the image to the given mean and standard deviation for consistency with pytorch backbone
        """
        image = image.astype(np.float32)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image /= 255.0
        image -= mean
        image /= std
        return image
    
    
    @torch.inference_mode()
    def preprocess(self, image):
        '''
        Desc: Preprocess input image.
        args:
            - image: numpy array [H,W,Ch]
        
        '''
        if self.inference_mode=='TRT':
            h,w =  self.model_shape
            img = pipeline_utils.resize_image(self.normalize(image), W=w, H=h)
            input_dtype = np.float16 if self.fp16 else np.float32
            input_batch = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(input_dtype)
            return self.from_numpy(input_batch)
        elif self.inference_mode=='PT':
            processed_image = self.pt_transform(image=image)["image"]
            if len(processed_image) == 3:
                processed_image = processed_image.unsqueeze(0)
            return processed_image.to(self.device)
        else:
            raise Exception(f'Unknown model format: {self.inference_mode}')
        
    
    @torch.inference_mode()
    def warmup(self):
        '''
        Desc: Warm up model using a np zeros array with shape matching model input size.
        Args: None
        '''
        shape=self.model_shape+[3,]
        self.predict(np.zeros(shape))
        
        
    @torch.inference_mode()
    def predict(self, image):
        '''
        Desc: Model prediction 
        Args: image: numpy array [H,W,Ch]
        
        Note: predict calls the preprocess method
        '''
        if self.inference_mode=='TRT':
            input_batch = self.preprocess(image)
            self.binding_addrs['input'] = int(input_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            outputs = {x:self.bindings[x].data.cpu().numpy() for x in self.output_names}
            output=outputs['output']
        elif self.inference_mode=='PT':
            preprocessed_image = self.preprocess(image)
            output=self.pt_model(preprocessed_image)[0].cpu().numpy()
        output=np.squeeze(output).astype(np.float32)
        return output



if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-a','--action', default="test", nargs='?', choices=['convert','test'], help='Action modes')
    ap.add_argument('-i','--model_path', default="/app/model/model.pt", help='Input model file path.')
    ap.add_argument('-e','--export_dir', default="/app/export")
    ap.add_argument('-d','--data_dir', default="/app/data", help='Data file directory.')
    ap.add_argument('-o','--annot_dir', default="/app/annotation_results", help='Annot file directory.')
    ap.add_argument('-g','--generate_stats', action='store_true',help='generate the data stats')
    ap.add_argument('-p','--plot',action='store_true', help='plot the annotated images')
    ap.add_argument('-t','--ad_threshold',type=float,default=None,help='AD patch threshold.')
    ap.add_argument('-m','--ad_max',type=float,default=None,help='AD patch max anomaly.')

    args = vars(ap.parse_args())
    action=args['action']
    model_path = args['model_path']
    export_dir = args['export_dir']
    
    ad = AnomalyModel(model_path)
    
    if action=='convert':
        os.makedirs(export_dir,exist_ok=True)
        ad.convert(model_path,export_dir)

    if action=='test':
        os.makedirs(args['annot_dir'],exist_ok=True)
        ad.test(args['data_dir'],args['annot_dir'],args['generate_stats'],
                args['plot'],args['ad_threshold'],args['ad_max'])
        