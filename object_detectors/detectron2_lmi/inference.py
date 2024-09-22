import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.modeling import build_model, detector_postprocess
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, GenericMask
from model_base import ModelBase
from detectron2_lmi.utils.det_utils import merge_a_into_b
import yaml
import logging
import time
import cv2
import numpy as np
from gadget_utils.pipeline_utils import plot_one_box
from label_utils.shapes import Rect, Mask, Keypoint
from label_utils.csv_utils import write_to_csv
import json


"""
TODO Update for deploying to LMI AISolutions
"""

class Detectron2Model(ModelBase):
    logger = logging.getLogger(__name__)
    
    def __init__(self,weights_path: str, config_file: str, detectron2_config_file: str, class_map: dict):
        """
        The function initializes a model with specified weights and configuration for object detection tasks
        in Python.
        
        :param weights_path: The `weights_path` parameter is the file path to the pre-trained weights of the
        model that you want to load for inference. This file typically contains the learned parameters of
        the model after training on a specific dataset
        :type weights_path: str
        :param config_file: The `config_file` parameter is a string that represents the file path to a
        configuration file. This configuration file is used to configure the model and its components, such
        as the model architecture, input sizes, and other settings required for the model to make
        predictions
        :type config_file: str
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(detectron2_config_file))
        self.cfg.merge_from_file(config_file)
        self.cfg = self.cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(weights_path)
        # Get min max test size, to handle bigger inmages. 
        self.transforms = T.ResizeShortestEdge([self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST)
        self.class_map = {
           int(k): str(v) for k,v in class_map.items()
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def warmup(self):
        """
        The `warmup` function initializes a tensor input and runs the model on it for 10 iterations without
        gradient computation.
        """
        # create a tensor input for warmup
        input = dict(image=torch.randint(high=255, size=(3,self.cfg.INPUT.MAX_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST,)).to(torch.float32).to(self.device), height=self.cfg.INPUT.MAX_SIZE_TEST, width=self.cfg.INPUT.MAX_SIZE_TEST) 

        # 10 iterations of forward pass without gradient computation

        t0 = time.time()
        for i in range(10):
            self.forward(input)
        t1 = time.time()
        self.logger.info(f"Warmup time: {(t1-t0) * 1000} ms")


    def preprocess(self, image):
        """
        The `preprocess` function takes an image, applies transformations, converts it to a tensor, and
        returns the processed image along with its height and width.
        
        :param image: The `image` parameter is a NumPy array representing an image. The `preprocess` method
        takes this image as input and performs some preprocessing steps on it before returning a dictionary
        containing the preprocessed image as a PyTorch tensor, along with the height and width of the
        original image
        :return: The preprocess method returns a dictionary with the following keys and values:
        - "image": torch tensor of the preprocessed input image, converted to float32 and transposed to (C,
        H, W) format, then moved to the device specified in the class
        - "height": the height of the original image
        - "width": the width of the original image
        """
        t0 = time.time()
        height, width = image.shape[:2]
        inputs = self.transforms.get_transform(image).apply_image(image)
        t1 = time.time()
        return {
            "image": torch.as_tensor(inputs.astype("float32").transpose(2, 0, 1)).to(self.device),
            "height": height,
            "width": width
        }
    
    def forward(self, input,**kwargs):
        """
        The `forward` function takes an input, passes it through a model without gradient computation, and
        returns the outputs.
        
        :param input: The `input` parameter in the `forward` method is typically the input data that will be
        passed through the neural network model for inference or training. This input data could be a single
        data point, a batch of data points, or any other form of input that the model is designed to process
        :return: the outputs generated by passing the input through the model.
        """
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.inference([input], do_postprocess=kwargs.get(f"do_postprocess", False))
        t1 = time.time()
        return outputs
    
    def postprocess(self, results, image_height, image_width, confs,return_segments=False, **kwargs):
        """
        The `postprocess` function takes the outputs of the model and processes them to generate the final
        predictions.
        
        :param outputs: The `outputs` parameter is the output of the model, which contains the detected
        features and their properties
        :return: The `postprocess` method returns the final predictions generated by processing the outputs
        of the model
        """

        if isinstance(results, list):
            results = results[0]
        
        results = detector_postprocess(results, image_height, image_width, mask_threshold=kwargs.get("mask_threshold", 0.5))
        
        boxes = results.pred_boxes.tensor if results.has("pred_boxes") else torch.tensor([]).to(results.pred_boxes.tensor.device)
        scores = results.scores if results.has("scores") else torch.tensor([]).to(results.pred_boxes.tensor.device)
        classes = results.pred_classes if results.has("pred_classes") else torch.tensor([]).to(results.pred_boxes.tensor.device)
        keypoints = results.pred_keypoints if results.has("pred_keypoints") else None
        masks = results.pred_masks if results.has("pred_masks") else torch.tensor([]).to(results.pred_boxes.tensor.device)
        
        postprocessed_results = {}
        
        thresholds = torch.tensor([confs.get(self.class_map.get(int(classes[k])), 1.0) for k in range(len(classes))])
        
        keep = scores > thresholds.to(scores.device)
        
        postprocessed_results["boxes"] = boxes[keep].cpu().numpy() if boxes is not None else np.array([])
        postprocessed_results["scores"] = scores[keep].cpu().numpy() if scores is not None else np.array([])
        postprocessed_results["classes"] = [self.class_map.get(int(label), str(label)) for label in classes[keep].cpu().numpy()] if classes is not None else np.array([])
        postprocessed_results["keypoints"] = keypoints[keep].cpu().numpy() if keypoints is not None else np.array([])
        postprocessed_results["masks"] = masks[keep].cpu().numpy() if masks is not None else np.array([])
        

        if return_segments:
            # Runs contour detection
            postprocessed_results["segments"] = [GenericMask(x, image_height, image_width).polygons[0].reshape(-1, 2) for x in  postprocessed_results["masks"]]
        
        return postprocessed_results
    
    def predict(self, image, confs, **kwargs):
        """
        The `predict` function preprocesses an image and then passes it through a neural network for forward
        propagation to make a prediction.
        
        :param image: The `image` parameter is the input image that will be passed to the `predict` method
        for making predictions
        :return: The `predict` method is returning the output of the `forward` method applied to the
        preprocessed input image.
        """
        input = self.preprocess(image)
        orig_height, orig_width = input["height"], input["width"]
        predictions = self.forward(input)
        results = self.postprocess(predictions, orig_height, orig_width, confs,return_segments=kwargs.get("return_segments", False))
        return results
    
    
    def annotate_image(self, image, results, **kwargs):
        """
        The `annotate_image` function takes an image and the outputs of the model and visualizes the detected
        objects on the image.
        
        :param image: The `image` parameter is the input image on which the detected objects will be
        visualized
        :param outputs: The `outputs` parameter is the output of the model, which contains the detected
        objects and their properties
        """
        colormap = kwargs.get("color_map", None)
        for i in range(len(results["classes"])):
            box = results["boxes"][i]
            class_id = results["classes"][i]
            color = colormap[class_id] if colormap is not None else None
            mask = results["masks"][i] if len(results["masks"]) > 0 else None
            label = f"{class_id}"
            plot_one_box(box=box, img=image, mask=mask, mask_threshold=0.5, color=color, label=label)
        return image
        

if __name__ == "__main__":
    import argparse
    import glob
    import cv2
    import os
    import tqdm
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="Path to the weights file", default="/home/weights/model_final.pth")
    parser.add_argument("--config-file", type=str, help="Path to the config file", default="/home/config.yaml")
    parser.add_argument("--detectron2-config", type=str, help="Detectron2 config file", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--input-path", type=str, help="Path to the input folder", default="/home/input")
    parser.add_argument("--output-path", type=str, help="Path to the output folder", default="/home/output")
    parser.add_argument("--confidence-threshold", type=float, help="Path to the confidence map file", default=0.5)
    parser.add_argument("--class-map", type=str, help="Class map json file", default="/home/class_map.json")
    
    args = parser.parse_args()
    
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    with open(args.class_map, "r") as f:
        class_map = json.load(f)
        
    # with open(args.confidence_map, "r") as f:
    #     confidences = json.load(f)
    
    confidence_map = {
        str(v): args.confidence_threshold for k,v in class_map.items()
    }
    
    model = Detectron2Model(weights_path=args.weights, config_file=args.config_file, detectron2_config_file=args.detectron2_config, class_map=class_map)
    
    model.warmup()
    
    images = glob.glob(os.path.join(args.input_path, "*")) # TODO change this to the correct extension
    shapes = {}

    for image_path in tqdm.tqdm(images):
        results = []

        try:
            image = cv2.imread(image_path) # read in as BGR
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue
        t0 = time.time()
        outputs = model.predict(image=image, confs=confidence_map)
        t1 = time.time()
        fname = os.path.basename(image_path)
        
        
        # annotate the image
        annotated_image = model.annotate_image(image, outputs)
        cv2.imwrite(os.path.join(args.output_path, fname), annotated_image)
        
        # write the output to a file
        for i in range(len(outputs["classes"])):
            box = outputs["boxes"][i]
            class_id = outputs["classes"][i]
            score = outputs["scores"][i]
            results.append(
                Rect(im_name=fname, category=class_id, up_left=box[:2].astype(int).tolist(), bottom_right=box[2:].astype(int).tolist(), confidence=score, angle=0)
            )
        shapes[fname] = results
            
    write_to_csv(shapes, os.path.join(args.output_path, f"predictions.csv"), overwrite=True)
            
        
        