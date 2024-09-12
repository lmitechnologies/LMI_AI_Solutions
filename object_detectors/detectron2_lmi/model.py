from model_base import ModelBase
import tensorrt as trt
from cuda import cudart
import numpy as np
import detectron2_lmi.utils.common_runtime as common
from typing import Optional
from detectron2.structures import Instances, Boxes, ROIMasks
import torch
from gadget_utils.pipeline_utils import plot_one_box, resize_image

class Detectron2TRT(ModelBase):
    
    def __init__(self, model_path):
        
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()

        self.model_inputs = []
        self.model_outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
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
                self.model_inputs.append(binding)
            else:
                self.model_outputs.append(binding)
        self.input_shape = self.model_inputs[0]["shape"]
        self.data_type = self.model_inputs[0]["dtype"]
        self.input_width = self.input_shape[3]
        self.input_height = self.input_shape[2]
    
    def warmup(self):
        """_summary_
        """
        image_h, image_w = self.input_height, self.input_width
        input = np.random.rand(self.batch_size, 3, image_h, image_w).astype(self.data_type)
        self.forward(input)
        
    def preprocess(self, images: list|np.ndarray):
        """_summary_

        Args:
            images (_type_): _description_

        Returns:
            _type_: _description_
        """
        image_h, image_w = self.input_shape[2], self.input_shape[3]
        input = np.zeros((self.batch_size, 3, image_h, image_w), dtype=np.float32)
        if isinstance(images, np.ndarray):
            images = [images]
        for i, image in enumerate(images):
            #HWC to CHW
            # nomalization is handled in the model
            input[i] = image.astype(np.float32).transpose(2, 0, 1)
        return input
    
    def forward(self, inputs):
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))
        # Process I/O and execute the network.
        common.memcpy_host_to_device(
            self.model_inputs[0]["allocation"], np.ascontiguousarray(inputs)
        )
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.model_outputs[o]["allocation"])
        return outputs

    def postprocess(self, image, predictions):
        """_summary_

        Args:
            image (_type_): _description_
            predictions (_type_): _description_

        Returns:
            _type_: _description_
        """
        # assuming batch size is 1
        if len(predictions) == 0 or len(predictions[0]) == 0:
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "masks": []
            }
        instances = predictions[0][0]
        boxes = predictions[1][0]
        masks = predictions[4][0]
        scores = predictions[2][0]
        classes = predictions[3][0]
        results = {
            "boxes": [],
            "scores": [],
            "classes": [],
            "masks": []
        }
        image_h, image_w = image.shape[0], image.shape[1]
        for i in range(0, instances):
            y1, x1, y2, x2 = boxes[i]
            x1, y1, x2, y2 = x1*image_w, y1*image_h, x2*image_w, y2*image_h
            mask = masks[i]
            results["boxes"].append([x1, y1, x2, y2])
            results["scores"].append(scores[i])
            results["classes"].append(classes[i])
            w = x2 - x1 # width
            h = y2 - y1 # height
            mask = mask.astype(np.uint8)
            mask = resize_image(mask, H=h, W=w)
            # crop the mask to the bbox size
            crop_mask = np.zeros((image_h, image_w), dtype=np.uint8)
            crop_mask[round(y1):round(y2), round(x1):round(x2)] = mask
            mask = crop_mask
            results["masks"].append(mask)
        return results 
            
    def predict(self, image):
        input = self.preprocess(image)
        predictions = self.forward(input)
        results = self.postprocess(image, predictions)
        return results