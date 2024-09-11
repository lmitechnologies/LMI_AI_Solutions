from model_base import ModelBase
import tensorrt as trt
from cuda import cudart
import numpy as np
import detectron2_lmi.utils.common_runtime as common
from typing import Optional

class Detectron2TRT(ModelBase):
    logger = trt.Logger(trt.Logger.ERROR)
    
    def __init__(self, model_path):
        self.model_path = model_path
        
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        
        # load the model
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        
        # create an execution context
        self.context = self.model.create_execution_context()
         
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.model.num_io_tensors):
            name = self.model.get_tensor_name(i)
            is_input = False
            if self.model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.model.get_tensor_dtype(name)
            shape = self.model.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
                "size": size,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        self.input_shape, self.data_type = self.inputs[0]["shape"], self.inputs[0]["dtype"]
    
    def warmup(self):
        image_h, image_w = self.input_shape[2], self.input_shape[3]
        input = np.random.rand(self.batch_size, 3, image_h, image_w).astype(self.data_type)
        self.forward(input)
        
    def preprocess(self, images):
        input = np.zeros(self.input_shape, dtype=self.data_type)
        # convert the image to channels first
        for i, image in enumerate(images):
            image = image.transpose(2, 0, 1).astype(np.float32)
            input[i] = image
        return input
    
    def forward(self, input):
        # prepare the outputs
        predictions = []
        for output in self.outputs:
            predictions.append(np.zeros(output["shape"], dtype=output["dtype"]))
        common.memcpy_host_to_device(
            self.inputs[0]["allocation"], np.ascontiguousarray(input)
        )
        self.context.execute_v2(self.allocations)
        
        # copy the outputs back to the host
        for i, output in enumerate(self.outputs):
            common.memcpy_device_to_host(predictions[i], output["allocation"])
        return predictions

    def postprocess(self):
        pass
    
    def predict(self, images):
        # preprocess the image
        input = self.preprocess(images)
        predictions = self.forward(input)
        
        results = {
            "instances": predictions[0],
            "boxes": predictions[1],
            "scores": predictions[2],
            "classes": predictions[3],
            "masks": predictions[4],
        }
        return results
            
        
        
        
        
    
        
if __name__ == "__main__":
    import argparse
    import cv2
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    
    model = Detectron2TRT(args.model_path)
    model.warmup()
    print(args.image_path)
    image = cv2.imread(args.image_path, -1)
    # print(image.shape)
    # model.predict(cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_GRAY2BGR))