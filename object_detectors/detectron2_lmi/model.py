from model_base import ModelBase
import tensorrt as trt
from cuda import cudart
import numpy as np
import detectron2_lmi.utils.common_runtime as common
from detectron2.layers.mask_ops import paste_mask_in_image_old
from gadget_utils.pipeline_utils import plot_one_box
import cv2
import logging
import torch
import time





class Detectron2TRT(ModelBase):
    logger = logging.getLogger(__name__)
    def __init__(self, model_path, class_map):
        """source: https://github.com/NVIDIA/TensorRT/tree/release/10.4/samples/python/detectron2"""
        
        trt_logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(trt_logger, namespace="")
        with open(model_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
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
        inputs = np.zeros((self.batch_size, 3, image_h, image_w), dtype=self.input_dtype)
        for i in range(0, self.batch_size):
            image = images[i]
            inputs[i] = image.astype(np.float32).transpose(2, 0, 1)
        return inputs
    
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
    
    
    def postprocess(self, images, predictions, **kwargs):
        results = {
            "boxes": [],
            "scores": [],
            "classes": [],
            "masks": []
        }
        
        if len(predictions) == 0:
            return results
        confs = kwargs.get("confs", {})
        mask_threshold_map = kwargs.get("mask_threshold_map", {})
        process_masks = kwargs.get("process_masks", False)
        image_h, image_w = images[0].shape[0], images[0].shape[1]
        
        num_preds, boxes, scores, classes, masks = predictions[:5]

        class_map = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))
        classes = class_map(classes)
        
        if len(boxes) > 0:
            scale_factors = np.array([image_w, image_h, image_w, image_h])
            boxes = (boxes * scale_factors).astype(np.int32)

        processed_masks = [] if process_masks else None
        processed_boxes = []
        processed_scores = []
        processed_classes = []
        # valid_scores = scores>= np.vectorize(confs.get)(classes, 0.5)
        # boxes, scores = boxes[valid_scores], scores[valid_scores]
        # classes = classes[valid_scores]
        # masks = masks[valid_scores] if process_masks else None
        

        for idx in range(self.batch_size):
            valid_scores = scores[idx] >= np.vectorize(confs.get)(classes[idx], 0.5)
            batch_boxes, batch_scores = boxes[idx][valid_scores], scores[idx][valid_scores]
            batch_classes = classes[idx][valid_scores]
            

            processed_boxes.append(batch_boxes)
            processed_scores.append(batch_scores)
            processed_classes.append(batch_classes)
            
            if process_masks:
                batch_masks = []
                filtered_masks = masks[idx][valid_scores]
                
                
                for i, box in enumerate(batch_boxes):
                    label = batch_classes[i]
                    mask = filtered_masks[i]
                    
                    w = box[2] - box[0] + 1
                    h = box[3] - box[1] + 1
                    mask = mask > mask_threshold_map.get(label, 0.5)
                    mask = mask.astype(np.uint8)
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    x_0, x_1 = max(box[0], 0), min(box[2] + 1, image_w)
                    y_0, y_1 = max(box[1], 0), min(box[3] + 1, image_h)
                    im_mask = np.zeros((image_h, image_w), dtype=np.uint8)
                    im_mask[y_0:y_1, x_0:x_1] = mask[
                        (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
                    ]
                    batch_masks.append(im_mask)
                processed_masks.append(batch_masks)
                
        results = {
            "boxes": processed_boxes,
            "scores": processed_scores,
            "classes": processed_classes,
            "masks": processed_masks if process_masks else []
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
        t0 = time.time()
        predictions = self.postprocess(images, predictions,**kwargs)
        t1 = time.time()
        self.logger.info(f"PostProcessing Time: {(t1-t0)*1000:.2f} ms")
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
                    label=f"{classes}:{scores:.2f}",
                    mask=masks[i] if masks is not None else None,
                    color=color_map,
                )
        return images
    
    def annotate_image(self, result, image, color_map=None):
        for i in range(len(result["classes"])):
            plot_one_box(
                result["boxes"][i],
                image,
                label=f"{result['classes'][i]}:{result['scores'][i]:.2f}",
                mask=result["masks"][i] if len(result["masks"]) > 0 else None,
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
    parser.add_argument("--model_path", type=str, default="/home/weights/model.trt")
    parser.add_argument("--images_path", type=str, default="/home/input")
    parser.add_argument("--class-map", type=str, help="Class map json file", default="/home/class_map.json")

    
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
    batches = model.get_batches(image_batch)
    current_image_count = 0
    for idx, batch in enumerate(batches):
        batch_preds = model.predict(batch)
        model.annotate_images(batch_preds, batch)
        num_images = len(batch_preds["classes"])
        batch_classes = batch_preds["classes"]
        # remove empty images
        for i in range(num_images):
            if i + current_image_count >= len(images):
                break
            image_path = images[i + current_image_count]
            image = cv2.imread(image_path)
            batch_pred = {
                "boxes": batch_preds["boxes"][i],
                "scores": batch_preds["scores"][i],
                "classes": batch_preds["classes"][i],
            }
            annotated_image = model.annotate_image(batch_pred, image)
            file_ext = os.path.basename(image_path).split('.')[-1]
            cv2.imwrite(f"/home/data/output/{os.path.basename(image_path)}_annotated.{file_ext}", annotated_image)
        current_image_count += num_images
    # t0 = time.time()
    # preds = model.predict(image_batch[:20], {}, {}, process_masks=True)
    # t1 = time.time()
    # print(f"Inference time: {(t1 - t0)* 1000} ms")
    # model.annotate_images(preds, image_batch[:20])
    # for image,pth in zip(image_batch[:20], images[:20]):
    #     cv2.imwrite(f"/home/data/output/{os.path.basename(pth)}_annotated.png", image)
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
    