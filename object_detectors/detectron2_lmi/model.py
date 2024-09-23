from model_base import ModelBase
import tensorrt as trt
from cuda import cudart
import numpy as np
import detectron2_lmi.utils.common_runtime as common
from gadget_utils.pipeline_utils import plot_one_box
import cv2

class Detectron2TRT(ModelBase):
    
    def __init__(self, model_path, class_map):
        """source: https://github.com/NVIDIA/TensorRT/tree/release/10.4/samples/python/detectron2"""
        
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Setup I/O bindings
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
        self.input_dtype = self.model_inputs[0]["dtype"]

        self.class_map = {
            int(k): str(v) for k, v in class_map.items()
        }
    
    def warmup(self):
        """_summary_
        """
        for _ in range(10):
            image_h, image_w = self.input_shape[2], self.input_shape[3]
            input = np.random.rand(self.batch_size, 3, image_h, image_w).astype(self.input_dtype)
            self.forward(input)
        
    def preprocess(self, images: list):
        """_summary_

        Args:
            images (_type_): _description_

        Returns:
            _type_: _description_
        """
        image_h, image_w = self.input_shape[2], self.input_shape[3]
        input = np.zeros((self.batch_size, 3, image_h, image_w), dtype=self.input_dtype)
        if isinstance(images, np.ndarray):
            images = [images]
        for i in range(0, self.batch_size):
            image = images[i]
            input[i] = image.astype(np.float32).transpose(2, 0, 1)
        return input
    
    def forward(self, inputs):
        outputs = []
        for out in self.model_outputs:
            outputs.append(np.zeros(out["shape"], dtype=out["dtype"]))
        common.memcpy_host_to_device(
            self.model_inputs[0]["allocation"], np.ascontiguousarray(inputs)
        )
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.model_outputs[o]["allocation"])
        return outputs
    
    
    def postprocess(self, images, predictions,**kwargs):
        results = {
            "boxes": [],
            "scores": [],
            "classes": [],
            "masks": []
        }
        confs = kwargs.get("confs")
        image_h, image_w = images[0].shape[0], images[0].shape[1]
        num_preds = predictions[0]
        boxes = predictions[1]
        scores = predictions[2]
        classes = predictions[3]
        masks = predictions[4]
        keep = scores > [
            confs.get(self.class_map.get(int(c), c), 0.5) for c in classes
        ]
        classes = classes[keep]
        scores = scores[keep]
        boxes = boxes[keep]
        masks = masks[keep]
    

        # convert classes to human readable format
        classes = [self.class_map.get(int(c), c) for c in classes]
        processed_masks = []
        results = []
        if len(boxes) > 0:
            
            boxes *= [images[0].shape[1], images[0].shape[0], images[0].shape[1], images[0].shape[0]]
            
            if kwargs.get("tile_offsets", False):
                tile_offsets = kwargs.get("tile_offsets")
                tile_offsets = np.array(tile_offsets)
                boxes[:, 0] += tile_offsets[0]
                boxes[:, 1] += tile_offsets[1]
                boxes[:, 2] += tile_offsets[0]
                boxes[:, 3] += tile_offsets[1]
            
            boxes = boxes.astype(np.int32)
            
            if kwargs.get("process_masks", False):
                for idx in range(0, self.batch_size):
                    proc_batch_masks = []
                    for i in range(0, int(num_preds[idx])):
                        label = classes[i]
                        mask = masks[i]
                        box = boxes[i]
                        samples_w = box[2] - box[0] + 1
                        samples_h = box[3] - box[1] + 1
                        mask = mask > kwargs.get("mask_threshod_map", {}).get(label, 0.5)
                        mask = mask.astype(np.uint8)
                        mask = cv2.resize(mask, (samples_w, samples_h), interpolation=cv2.INTER_LINEAR)
                        x_0 = max(box[0], 0)
                        x_1 = min(box[2] + 1, )
                        y_0 = max(box[1], 0)
                        y_1 = min(box[3] + 1, image_h)
                        im_mask = np.zeros((image_h, image_w), dtype=np.uint8)
                        im_mask[y_0:y_1, x_0:x_1] = mask[
                            (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
                        ]
                        proc_batch_masks.append(im_mask)
                    processed_masks.append(proc_batch_masks)
        results = {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "masks": processed_masks
        }
        return results
    
    def get_batches(self, images):
        # get the number of batches add empty images to make it a multiple of batch size
        num_batches = len(images) // self.batch_size
        if len(images) % self.batch_size != 0:
            num_batches += 1
            for i in range(0, self.batch_size - len(images) % self.batch_size):
                images.append(np.zeros_like(images[0]))
        for i in range(0, num_batches):
            yield images[i * self.batch_size : (i + 1) * self.batch_size]
    

    def predict(self, images, **kwargs):
        predictions = self.forward(self.preprocess(images))
        predictions = self.postprocess(images, predictions,**kwargs)
        return predictions
    
    def annotate_images(self, results, images, color_map=None):
        for idx, classes in enumerate(results["classes"]):
            boxes = results["boxes"][idx]
            classes = results["classes"][idx]
            scores = results["scores"][idx]
            masks = results["masks"][idx] if len(results["masks"]) > 0 else None
            for i in range(len(classes)):
                plot_one_box(
                    boxes[i],
                    images[idx],
                    label=f"{classes[i]}",
                    mask=masks[i] if masks is not None else None,
                    color=color_map,
                )
        return images



if __name__ == "__main__":
    import argparse
    import glob
    import os
    import time
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/weights/engine.trt")
    parser.add_argument("--images_path", type=str, default="/home/data/images")
    parser.add_argument("--class-map", type=str, help="Class map json file", default="/home/data/class_map.json")

    
    args = parser.parse_args()
    with open(args.class_map, "r") as f:
        class_map = json.load(f)
    model = Detectron2TRT(args.model_path, class_map)
    model.warmup()
    
    images = glob.glob(os.path.join(args.images_path, "*.png"))
    image_batch = [
        cv2.imread(image)
        for image in images
    ]
    t0 = time.time()
    preds = model.predict(image_batch[:20], {}, {}, process_masks=True)
    t1 = time.time()
    print(f"Inference time: {(t1 - t0)* 1000} ms")
    model.annotate_images(preds, image_batch[:20])
    for image,pth in zip(image_batch[:20], images[:20]):
        cv2.imwrite(f"/home/data/output/{os.path.basename(pth)}_annotated.png", image)
    # for image, result, pth in zip(image_batch[:20], preds, images[:20]):
    #     annotated_image = model.annotate_image(result, image)
    #     cv2.imwrite(f"/home/data/output/{os.path.basename(pth)}_annotated.png", annotated_image)

    # for pred in preds:
    #     image_path = images[0]
    #     image = cv2.imread(image_path)
    #     annotated_image = model.annotate_image(pred, image)
    #     cv2.imwrite(f"/home/data/output/{os.path.basename(image_path)}_annotated.png", annotated_image)


    # # predictions = model.postprocess(image_batch, predictions)
    # t1 = time.time()
    # print(f"Time taken: {(t1 - t0)* 1000} ms")
    # for image, result, pth in zip(image_batch[:20], preds, images[:20]):
    #     annotated_image = model.annotate_image(result, image)
    #     cv2.imwrite(f"/home/data/output/{os.path.basename(pth)}_annotated.png", annotated_image)
    