import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import logging
import numpy as np
import ctypes
import abc


class TRT_Model:
    """
    the base class for all the tensorRT models
    """
    logger = logging.getLogger()

    def __init__(self, engine_file_path, plugin_path='', channel_first=True) -> None:
        """
        load the engine and plugin
        Args:
            engine_file_path (str): the path to the tensorRT engine
            plugin_path (str, optional): the path to the tensorRT engine plugin. Defaults to ''.
            channel_first (bool, optional): use the channel first format: [C,H,W], otherwise use channel last format: [H,W,C]. Defaults to True.
        """
        self.name = self.__class__.__name__
        
        #load the plugin
        if plugin_path:
            ctypes.CDLL(plugin_path)
        
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.batch_max_size = self.engine.max_batch_size
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = {}
        self.cuda_outputs = {}
        self.bindings = []

        for binding in self.engine:
            self.logger.info(f'{self.name} binding:, {binding}, {self.engine.get_binding_shape(binding)}')
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                if channel_first:
                    self.input_c = self.engine.get_binding_shape(binding)[0]
                    self.input_h = self.engine.get_binding_shape(binding)[1]
                    self.input_w = self.engine.get_binding_shape(binding)[2]
                else:
                    self.input_h = self.engine.get_binding_shape(binding)[0]
                    self.input_w = self.engine.get_binding_shape(binding)[1]
                    self.input_c = self.engine.get_binding_shape(binding)[2]
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs[binding] = host_mem
                self.cuda_outputs[binding] = cuda_mem
        
        
    @abc.abstractmethod
    def infer(self, images_raw:list):
        """Must need to be implemented in the child class
        Args:
            images_raw (list): a list of numpy arrays
        """
        pass
        
        
    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
        
    def get_raw_image_zeros(self):
        """
        description: Ready data for warmup
        """
        return np.zeros([self.batch_max_size, self.input_h, self.input_w, self.input_c], dtype=np.uint8)
    

    def preprocess_image(self,image_raw,BGR_to_RGB=False,normalize=True,HWC_to_NCHW=True):
        """
        description:    BGR to RGB,
                        normalize to [0,1],
                        transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image(np.array):  the processed image
        """
        errors = []

        image = image_raw.astype(np.float32)
        if BGR_to_RGB:
            image = image[:,:,::-1]
        if normalize:
            # Normalize to [0,1]
            image /= 255.0
        if HWC_to_NCHW:
            # HWC to CHW format:
            image = np.transpose(image, [2, 0, 1])
            # CHW to NCHW format
            image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, errors
    
