import os
import logging 
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from .base import Anomalib_Base, to_list
from image_utils.tiler import Tiler, ScaleMode, OverlapMode
import gadget_utils.pipeline_utils as pipeline_utils
from ad_core.anomaly_detector_registry import AnomalyDetectorRegistry
logging.basicConfig()


MINIMUM_QUANT=1e-12
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))


@AnomalyDetectorRegistry.register(metadata=dict(frameworks=['anomalib1'], model_names=['patchcore', 'padim', 'efficientad'], tasks=['anomalydetection','seg'], versions=['v1']))
class AnomalyModel2(Anomalib_Base):
    '''
    Desc: Class used for AD model inference.
    '''
    logger = logging.getLogger('AnomalyModel v1')
    logger.setLevel(logging.INFO)
    
    def __init__(self, model_path, tile=None, stride=None, tile_mode='padding', **kwargs):
        """_summary_

        Args:
            model_path (str): the path to the model file, either a pt or trt engine file
            tile (int | list, optional): tile size [h,w]. Must provide if using tiling
            stride (int | list, optional): stride size [h,w]. Must provide if using tiling
            tile_mode (str, optional): 'padding' or 'resize'. Defaults to 'padding'
        attributes:
            - self.device: device to run model on
            - self.fp16: flag for half precision
            - self.model_shape: model input shape (h,w)
            - self.inference_mode: model inference mode (TRT or PT)
            - self.tiler: tiling object
        """
        if not os.path.isfile(model_path):
            raise Exception(f'Cannot find the model file: {model_path}')
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.logger.warning('GPU device unavailable. Use CPU instead.')
            self.device = torch.device('cpu')
        self.pt_metadata = {}
        self.image_size = kwargs.get('image_size', [224,224])
            
        _,ext = os.path.splitext(model_path)
        self.fp16 = False
        self.logger.info(f"Loading model: {model_path}")
        if ext=='.engine':
            import tensorrt as trt
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
                    input_shape = shape
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    self.output_names.append(name)
                im = self.from_numpy(np.empty(shape, dtype=dtype))
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            self.model_shape=list(input_shape[-2:])
            self.image_size = self.model_shape
            self.batch_size = input_shape[0]
            self.inference_mode='TRT'
        elif ext=='.pt':
            try:  
                # try loading the model using torchscript
                self.pt_model = torch.jit.load(model_path,map_location=self.device)
                self.model_shape = self.image_size
                self.logger.info(f"Traced model shape: {self.model_shape}")
            except Exception as e:
                checkpoint = torch.load(model_path,map_location=self.device,weights_only=False)
                self.pt_model = checkpoint['model']
                self.pt_metadata = checkpoint["metadata"]
                self.logger.info(f"Model metadata: {self.pt_metadata}")
                for d in self.pt_model.transform.transforms:
                    if isinstance(d, v2.Resize):
                        self.model_shape = to_list(d.size)
                        self.image_size = to_list(d.size)
                        self.logger.info(f"Model shape: {self.model_shape}")
                
            self.pt_model.eval()
            self.inference_mode='PT'
        else:
            raise Exception(f'Unknown model format: {ext}')
        
        # init tiler
        if tile is not None:
            self.logger.info('Tiling is enabled.')
            if stride is None:
                raise Exception('Must provide stride using tiling')
            
            tile = to_list(tile)
            if self.model_shape != tile:
                raise Exception(f'tile shape {tile} mismatch with model expected shape: {self.model_shape}')
            
            self.tiler = Tiler(tile,stride)
            self.tile_mode = ScaleMode.PADDING if tile_mode=='padding' else ScaleMode.INTERPOLATION
            self.logger.info(f'init tiler with tile={tile}, stride={stride}, mode={self.tile_mode}')
            
            
    
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
            
        img = img.permute((2, 0, 1)).unsqueeze(0)
        img = img / 255.0
        
        if self.tiler is not None:
            img = self.tiler.tile(img,self.tile_mode)
        
        batch = img.shape[0]
        if self.inference_mode=='TRT' and batch != self.batch_size:
            self.logger.warning(f'Got batch size of {batch},  but trt expects {self.batch_size}. The trt engine might output weird results')
            img = F.interpolate(img, size=self.model_shape, mode='bilinear')
        
        # resize baked into the pt model (although some torchscript models dont have preprocessing so resizing here)
        if self.tiler is None and self.inference_mode == 'PT':
            if img.shape[1] != self.model_shape[0] or img.shape[2] != self.model_shape[1]:
                self.logger.debug(f'Input image shape {image.shape[:2]} does not match model shape {self.model_shape}. Resizing input image.')
                img = v2.Resize(self.model_shape, antialias=True)(img)
        
        img = img.contiguous()
        return img.half() if self.fp16 else img
    
    
    def _infer(self, input_batch):
        '''
        Desc: Run inference on the input batch.
        Args:
            - input_batch: preprocessed input batch
        Returns:
            - output: model output tensor
        '''
        if self.inference_mode == 'TRT':
            self.binding_addrs['input'] = int(input_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            output_tensor = self.bindings['output'].data
            
        elif self.inference_mode == 'PT':
            preds = self.pt_model(input_batch)
            if isinstance(preds, torch.Tensor):
                output_tensor = preds
            elif isinstance(preds, dict):
                output_tensor = preds['anomaly_map']
            elif isinstance(preds, Sequence):
                output_tensor = preds[1]
            else:
                raise Exception(f'Unknown prediction type: {type(preds)}')
        
        return output_tensor
        
        
    @torch.inference_mode()
    def predict(self, image, **kwargs):
        '''
        Desc: Model prediction
        Args: image: numpy array [H,W,Ch] or [N,H,W,Ch]
        kwargs:
            overlap_mode (str): "average", "max", "cosine", "linear", "gaussian". Default 'average'.
            batch_size (int, optional): If provided and the input batch contains more
                                        samples than this size, the input batch will be
                                        split and processed in chunks of this size.
                                        The results are then aggregated.

        Note: predict calls the preprocess method
        returns:
            - output: processed output, typically a numpy array.
                      If tiling is used, this is the untilled output.
                      The output is squeezed.
        '''
        if self.tiler is not None:
            tiling_settings = kwargs.get('tiling_settings', {})
            overlap_mode_str = tiling_settings.get('overlap_mode', 'average')
            current_overlap_mode = OverlapMode(overlap_mode_str)
            tiling_settings['overlap_mode'] = current_overlap_mode
            tiling_settings['scale_mode'] = self.tile_mode

        input_batch = self.preprocess(image)
        if kwargs.get('verbose', False):
            self.logger.info(f'Using overlap mode: {overlap_mode_str}')
            self.logger.info(f'Final input batch shape: {input_batch.shape}')
            
        num_samples_in_input = input_batch.shape[0]
        if num_samples_in_input == 0:
            return np.array([])
        
        inference_settings = kwargs.get('inference_settings', {})
        user_inference_batch_size = inference_settings.get('inference_batch_size', None)
        perform_mini_batch_inference = (
            user_inference_batch_size is not None and \
            user_inference_batch_size > 0
        )
        
        aggregated_output_tensor = None
        if perform_mini_batch_inference:
            all_mini_batch_outputs = []
            for i in range(0, num_samples_in_input, user_inference_batch_size):
                mini_batch = input_batch[i:min(i + user_inference_batch_size, num_samples_in_input)]
                current_mini_batch_output_tensor = None
                current_mini_batch_output_tensor = self._infer(mini_batch)
                
                if current_mini_batch_output_tensor is not None:
                    all_mini_batch_outputs.append(current_mini_batch_output_tensor)
                else:
                    raise Exception(f"Model failed to produce an output for a mini-batch.")

            if not all_mini_batch_outputs:
                raise Exception("Batched inference was performed, but no outputs were collected.")

            if isinstance(all_mini_batch_outputs[0], torch.Tensor):
                aggregated_output_tensor = torch.cat(all_mini_batch_outputs, dim=0)
            else:
                raise Exception(f"Unsupported output type for aggregation: {type(all_mini_batch_outputs[0])}")

        else: 
            aggregated_output_tensor = self._infer(input_batch)
        
        if aggregated_output_tensor is None:
            raise Exception("Model inference failed to produce an output tensor.")

        processed_output = aggregated_output_tensor
        
        if self.tiler is not None:
            processed_output = self.tiler.untile(processed_output, **tiling_settings)
    
        output_numpy = None
        if isinstance(processed_output, torch.Tensor):
            output_numpy = processed_output.cpu().numpy()
        elif isinstance(processed_output, np.ndarray): 
            output_numpy = processed_output
        else:
            raise Exception(f"Output from model/tiler is of unexpected type: {type(processed_output)}")

        final_squeezed_output = np.squeeze(output_numpy)
        
        return final_squeezed_output

        

    def warmup(self,input_hw=None):
        '''
        Desc: 
            Warm up model using a np zeros array with shape matching model input size.
        Args: 
            input_hw(int | list, optional): a int if h equals to w, or a list of [h,w]. Need to specify this if using tiling. Otherwise, use model's built-in shape.
        '''
        if input_hw is None:
            input_hw = self.model_shape
        input_hw = to_list(input_hw)
        zeros = np.zeros(input_hw+[3,])
        self.logger.info(f'Warming up model with input shape: {zeros.shape}')
        self.predict(zeros)



if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest='action',required=True,help='Action modes: test or convert')
    
    test_ap = subs.add_parser('test',help='test model')
    test_ap.add_argument('-i','--model_path', default="/app/model/model.pt", help='Input model file path.')
    test_ap.add_argument('-d','--data_dir', default="/app/data", help='Data file directory.')
    test_ap.add_argument('-o','--annot_dir', default="/app/annotation_results", help='Annot file directory.')
    test_ap.add_argument('-g','--generate_stats', action='store_true',help='generate the data stats')
    test_ap.add_argument('-p','--plot',action='store_true', help='plot the annotated images')
    test_ap.add_argument('-t','--ad_threshold',type=float,default=None,help='AD patch threshold.')
    test_ap.add_argument('-m','--ad_max',type=float,default=None,help='AD patch max anomaly.')
    test_ap.add_argument('--tile',type=int,nargs=2,default=None,help='tile size (h,w)')
    test_ap.add_argument('--stride',type=int,nargs=2,default=None,help='stride size (h,w)')
    test_ap.add_argument('--resize',action='store_true',help='use resize for tiling')
    test_ap.add_argument('-om', '--overlap_mode', default="gaussian", help='overlap mode for tiling, can be "average", "max", "cosine", "linear", "gaussian"')
    
    convert_ap = subs.add_parser('convert',help='convert model to trt engine')
    convert_ap.add_argument('-i','--model_path', default="/app/model/model.pt", help='Input model file path.')
    convert_ap.add_argument('-o','--export_dir', default="/app/export")
    convert_ap.add_argument('-c','--convert_type', default="trt", type=str, choices=['trt','onnx'], help='convert type: trt or onnx')
    convert_ap.add_argument('--hw',type=int,nargs=2,default=None,help='input image shape (h,w). Muse be provided if using tiling')
    convert_ap.add_argument('--tile',type=int,nargs=2,default=None,help='tile size (h,w)')
    convert_ap.add_argument('--stride',type=int,nargs=2,default=None,help='stride size (h,w)')
    convert_ap.add_argument('--resize',action='store_true',help='use resize for tiling, otherwise pad zeros')
    args = vars(ap.parse_args())
    
    action=args['action']
    model_path = args['model_path']
    
    mode = 'resize' if args['resize'] else 'padding'
    ad = AnomalyModel2(model_path,args['tile'],args['stride'],mode)
    
    if action=='convert':
        export_dir = args['export_dir']
        os.makedirs(export_dir, exist_ok=True)
        if args['convert_type']=='onnx':
            onnx_path = os.path.join(export_dir, 'model.onnx')
            ad.convert_to_onnx(onnx_path, args['hw'])
        if args['convert_type']=='trt':
            ad.convert(model_path,export_dir,args['hw'])
    elif action=='test':
        os.makedirs(args['annot_dir'], exist_ok=True)
        ad.test(args['data_dir'],args['annot_dir'],args['generate_stats'],
                args['plot'],args['ad_threshold'],args['ad_max'], args['overlap_mode'])
