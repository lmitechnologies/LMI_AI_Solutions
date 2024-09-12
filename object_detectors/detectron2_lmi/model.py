from model_base import ModelBase
import tensorrt as trt
from cuda import cudart
import numpy as np
import detectron2_lmi.utils.common_runtime as common
from typing import Optional
from detectron2.structures import Instances, Boxes, ROIMasks
import torch
from gadget_utils.pipeline_utils import plot_one_box, resize_image
import cv2

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
        input = np.zeros((self.batch_size, 3, image_h, image_w), dtype=self.data_type)
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
            
        common.memcpy_host_to_device(
            self.model_inputs[0]["allocation"], np.ascontiguousarray(inputs)
        )
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.model_outputs[o]["allocation"])
        return outputs
    
    def mask_to_polygons(self, mask):
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        res = res[-2]
        res[:, :, 0] += 0.5
        res[:, :, 1] += 0.5
        return res

    def postprocess(self, image, predictions):
        """_summary_

        Args:
            image (_type_): _description_
            predictions (_type_): _description_

        Returns:
            _type_: _description_
        """
        results = {
                "boxes": [],
                "scores": [],
                "classes": [],
                "masks": []
            }
        # assuming batch size is 1
        if len(predictions) == 0 or len(predictions[0]) == 0:
            return results
        instances = predictions[0][0]
        boxes = predictions[1][0]
        boxes = boxes[:, [1, 0, 3, 2]]
        boxes[:, 0] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]
        scores = predictions[2][0]
        classes = predictions[3][0].astype(int)
        
        image_h, image_w = image.shape[0], image.shape[1]
        for i in range(0, instances):
            y1, x1, y2, x2 = boxes[i]
            w = x2 - x1 # width
            h = y2 - y1 # height
            mask = mask.astype(np.uint8)
            mask = resize_image(mask, H=h, W=w)
            results["masks"].append(mask)
            # crop the mask to the bbox size
            # crop_mask = np.zeros((image_h, image_w), dtype=np.uint8)
            # crop_mask[round(y1):round(y2), round(x1):round(x2)] = mask
            # mask = crop_mask
            # results["masks"].append(mask)
            # # get the segments from the mask using contours
            # polygons = self.mask_to_polygons(mask)
            # results["segments"].append(polygons)
        results["boxes"] = boxes
        results["scores"] = scores
        results["classes"] = classes
        return results 
            
    def predict(self, image):
        input = self.preprocess(image)
        predictions = self.forward(input)
        results = self.postprocess(image, predictions)
        return results
    
    def annotate_image(self, results, image):
        for i in range(len(results["boxes"])):
            plot_one_box(
                results["boxes"][i],
                image,
                label=f"{results['classes'][i]}",
                mask=results["masks"][i],
                color=[0, 255, 0],
            )
        return image


if __name__ == "__main__":
    import argparse
    import glob
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    args = parser.parse_args()
    model = Detectron2TRT(args.model_path)
    model.warmup()
    
    images = glob.glob(os.path.join(args.images_path, "*.png"))
    for image in images:
        image = cv2.imread(image)
        
        results = model.predict(image)
        annotated_image = model.annotate_image(results, image)
    
    