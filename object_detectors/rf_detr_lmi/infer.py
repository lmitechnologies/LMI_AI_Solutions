import argparse
import json
import logging
import os
import time

import cv2
from dataset_utils.file_utils import get_images
from rf_detr_lmi.model import RfdetrModel

# setup the logger
logger = logging.getLogger("RFDETR-INFER")
logger.setLevel(logging.INFO)


def setup_parser():
    parser = argparse.ArgumentParser(description="RF-DETR-LMI Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--input", type=str, required=True, help="Path to input images")
    parser.add_argument("--output", type=str, required=True, help="Path to save output results")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--class_map", type=str, required=False, help="Path to class map JSON file")
    parser.add_argument("--image_size", type=int, default=640, help="Input image size for the model")
    return parser


def inference_run(args):
    model_path = args.get("weights")
    imgs_path = args.get("input")
    out_path = args.get("output")
    class_map_path = args.get("class_map", None)
    image_size = args.get("image_size", 640)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    class_map = None
    if class_map_path:
        with open(class_map_path, "r") as f:
            class_map = json.load(f)
        logger.info(f"Loaded class map with {len(class_map)} classes from {class_map_path}")
        class_map = {int(k): v for k, v in class_map.items()}

    # load model
    model = RfdetrModel(model_path, class_map=class_map, image_size=[image_size, image_size])
    # model warmup
    model.warmup()
    # find images
    img_list = get_images(imgs_path)
    logger.info(f"Found {len(img_list)} images in {imgs_path}")
    inference_times = []
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        outputs = model.predict(image, configs=args.get("conf", 0.5))
        t1 = time.time()
        inference_times.append(t1 - t0)
        logger.info(f"Processed image: {img_name}, found {outputs} objects")
        annotated_image = model.annotate_image(outputs, image)

        output_image_path = os.path.join(out_path, img_name)
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, annotated_image_bgr)
        logger.info(f"Saved annotated image to {output_image_path}")

    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    logger.info(f"Average inference time per image: {avg_time:.4f} seconds | in ms: {avg_time * 1000:.2f} ms")
    max_time = max(inference_times) if inference_times else 0
    logger.info(f"Max inference time for an image: {max_time:.4f} seconds | in ms: {max_time * 1000:.2f} ms")
    min_time = min(inference_times) if inference_times else 0
    logger.info(f"Min inference time for an image: {min_time:.4f} seconds | in ms: {min_time * 1000:.2f} ms")


def main():
    parser = setup_parser()
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    inference_run(vars(args))


if __name__ == "__main__":
    main()
