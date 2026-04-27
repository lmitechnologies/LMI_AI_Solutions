import argparse
import json
import logging
import os
import time

import cv2

from lmi_utils.dataset_utils.file_utils import get_images
from object_detectors.rf_detr_lmi.model import RfdetrModel

# setup the logger
logger = logging.getLogger("RFDETR-INFER")


def setup_parser():
    parser = argparse.ArgumentParser(description="RF-DETR-LMI Inference")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to model weights")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input images")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save output results")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument(
        "--class_map",
        "-m",
        type=str,
        default=None,
        help="Path to class map JSON file (optional; auto-discovered from <model>.classes.json)",
    )
    parser.add_argument("--image_size", "-s", type=int, nargs=2, required=True, help="Input image size [h,w] for the model")
    parser.add_argument("--model_type", "-t", type=str, required=False, help="Type of the model to use for inference")
    return parser


def inference_run(args):
    model_path = args.weights
    imgs_path = args.input
    out_path = args.output
    class_map_path = args.class_map
    image_size = args.image_size
    model_type = args.model_type
    model_ext = os.path.splitext(model_path)[1].lower()
    if model_ext == ".pth":
        if not model_type:
            raise ValueError("Model type must be specified when using .pth weights")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    class_map = None
    if class_map_path:
        with open(class_map_path, "r") as f:
            class_map = json.load(f)
        logger.info(f"Loaded class map with {len(class_map)} classes from {class_map_path}")
        class_map = {int(k): v for k, v in class_map.items()}

    # load model
    model = RfdetrModel(model_path, class_map=class_map, image_size=image_size, model_type=model_type)
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
        image = cv2.resize(image, (image_size[1], image_size[0]))
        t0 = time.time()
        batch_outputs, time_info = model.predict(image, configs=args.conf)
        t1 = time.time()
        inference_times.append(t1 - t0)
        # Extract single image results from batch output
        outputs = {k: v[0] for k, v in batch_outputs.items()}
        logger.info(f"Processed image: {img_name}, found {len(outputs.get('boxes', []))} objects")
        annotated_image = model.annotate_image(outputs, image)

        output_image_path = os.path.join(out_path, img_name)
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, annotated_image_bgr)

    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    logger.info(f"Average inference time per image: {avg_time:.4f} seconds | in ms: {avg_time * 1000:.2f} ms")
    max_time = max(inference_times) if inference_times else 0
    logger.info(f"Max inference time for an image: {max_time:.4f} seconds | in ms: {max_time * 1000:.2f} ms")
    min_time = min(inference_times) if inference_times else 0
    logger.info(f"Min inference time for an image: {min_time:.4f} seconds | in ms: {min_time * 1000:.2f} ms")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = setup_parser()
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    inference_run(args)


if __name__ == "__main__":
    main()
