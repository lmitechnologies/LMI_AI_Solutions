import os
import logging 
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
import tensorrt as trt
import torch
import numpy as np
from torchvision.transforms import v2

from .base import Anomalib_Base
import gadget_utils.pipeline_utils as pipeline_utils


logging.basicConfig()


MINIMUM_QUANT=1e-12
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))


class AnomalyModel2(Anomalib_Base):
    '''
    Desc: Class used for AD model inference.
    '''
    logger = logging.getLogger('AnomalyModel v2')
    logger.setLevel(logging.INFO)
    
    def __init__(self, model_path):
        """_summary_

        Args:
            model_path (str): the path to the model file, either a pt or trt engine file

        attributes:
            - self.device: device to run model on
            - self.fp16: flag for half precision
            - self.shape_inspection: model input shape (h,w)
            - self.inference_mode: model inference mode (TRT or PT)
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.logger.warning('GPU device unavailable. Use CPU instead.')
            self.device = torch.device('cpu')
        _,ext = os.path.splitext(model_path)
        self.fp16 = False
        self.logger.info(f"Loading model: {model_path}")
        if ext=='.engine':
            with open(model_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            self.context = model.create_execution_context()
            self.bindings = OrderedDict()
            self.output_names = []
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
            self.shape_inspection=list(shape[-2:])
            self.inference_mode='TRT'
        elif ext=='.pt':  
            checkpoint = torch.load(model_path,map_location=self.device)
            self.pt_model = checkpoint['model']
            self.pt_model.eval()
            self.pt_model.to(self.device)
            self.pt_metadata = checkpoint["metadata"]
            self.logger.info(f"Model metadata: {self.pt_metadata}")
            for d in self.pt_model.transform.transforms:
                if isinstance(d, v2.Resize):
                    self.shape_inspection = d.size
            self.inference_mode='PT'
        else:
            raise Exception(f'Unknown model format: {ext}')
    
    
    @torch.inference_mode()
    def preprocess(self, image):
        '''
        Desc: Preprocess input image.
        args:
            - image: numpy array [H,W,Ch]
        '''
        img = self.from_numpy(image).float()
        # grayscale to rgb
        if img.ndim == 2:
            img = img.unsqueeze(-1).repeat(1,1,3)
        
        if self.inference_mode=='TRT':
            h, w =  self.shape_inspection
            img = pipeline_utils.resize_image(img, H=h, W=w)
            
        # resize baked into the pt model
        img = img.permute((2, 0, 1)).unsqueeze(0).contiguous()
        img = img / 255.0
        return img.half() if self.fp16 else img
        
        
    @torch.inference_mode()
    def predict(self, image):
        '''
        Desc: Model prediction 
        Args: image: numpy array [H,W,Ch]
        
        Note: predict calls the preprocess method
        returns:
            - output: resized output to match training data's size
        '''
        input_batch = self.preprocess(image)
        if self.inference_mode=='TRT':
            self.binding_addrs['input'] = int(input_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            outputs = {x:self.bindings[x].data for x in self.output_names}
            output = outputs['output']
        elif self.inference_mode=='PT':
            preds = self.pt_model(input_batch)
            if isinstance(preds, torch.Tensor):
                output = preds
            elif isinstance(preds, dict):
                output = preds['anomaly_map']
            elif isinstance(preds, Sequence):
                output = preds[1]
            else:
                raise Exception(f'Unknown prediction type: {type(preds)}')
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        output = np.squeeze(output)
        return output
        

    def warmup(self):
        '''
        Desc: Warm up model using a np zeros array with shape matching model input size.
        Args: None
        '''
        shape=self.shape_inspection+[3,]
        self.predict(np.zeros(shape))



if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-a','--action', default="test", help='Action: convert, test')
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
    if action=='convert':
        if not os.path.isfile(model_path):
            raise Exception('Cannot find the model file. Need a valid model file to convert.')
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        AnomalyModel2.convert(model_path,export_dir,fp16=True)

    if action=='test':
        if not os.path.isfile(model_path):
            raise Exception(f'Error finding {model_path}. Need a valid model file to test model.')
        if not os.path.exists(args['annot_dir']):
            os.makedirs(args['annot_dir'])
        AnomalyModel2.test(model_path, args['data_dir'],
             args['annot_dir'],
             generate_stats=args['generate_stats'],
             annotate_inputs=args['plot'],
             anom_threshold=args['ad_threshold'],
             anom_max=args['ad_max'])
