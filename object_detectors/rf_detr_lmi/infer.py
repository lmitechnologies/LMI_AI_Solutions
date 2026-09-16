import argparse
import json
import logging
import os
import time

import cv2

from lmi_utils.dataset_utils.file_utils import get_images
from object_detectors.od_core.infer_cli import JSON_NAME, PredictionsJson, add_infer_args, check_infer_args, predict_tiled, save_tile_plot
from object_detectors.rf_detr_lmi.model import RfdetrModel

# setup the logger
logger = logging.getLogger("RFDETR-INFER")


def setup_parser():
    parser = argparse.ArgumentParser(description="RF-DETR-LMI Inference")
    add_infer_args(parser, confidence=0.5)
    parser.add_argument(
        "--class_map",
        "-m",
        type=str,
        default=None,
        help="Optional override; by default class names come from the model file itself (embedded at export, or the .pth checkpoint)",
    )
    parser.add_argument(
        "--model_type",
        "-t",
        type=str,
        default=None,
        help="Optional override for .pth weights; by default the variant is read from the checkpoint",
    )
    return parser


def inference_run(args):
    model_path = args.weights
    imgs_path = args.input
    out_path = args.output
    class_map_path = args.class_map
    model_type = args.model_type
    tile = args.tile_step
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    class_map = None
    if class_map_path:
        with open(class_map_path, "r") as f:
            class_map = json.load(f)
        logger.info(f"Loaded class map with {len(class_map)} classes from {class_map_path}")
        class_map = {int(k): v for k, v in class_map.items()}

    # load model
    # .pth weights record no input size and need a square one; onnx/engine carry their own and ignore it
    model = RfdetrModel(model_path, class_map=class_map, model_type=model_type, image_size=args.image_size)
    image_size = model.image_size  # onnx/engine report their own input size, ignoring the argument
    logger.info(f"Model loaded with image size: {image_size}")
    # model warmup
    model.warmup()
    # find images
    img_list = get_images(imgs_path)
    logger.info(f"Found {len(img_list)} images in {imgs_path}")
    inference_times = []
    predictions_json = PredictionsJson(model.class_map.values())
    for img_path in sorted(img_list):
        img_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        if tile is not None:
            batch_outputs, time_info, tile_boxes = predict_tiled(model, image, tile, configs=args.confidence)
        else:
            batch_outputs, time_info = model.predict(image, configs=args.confidence)
        t1 = time.time()
        inference_times.append(t1 - t0)
        # Extract single image results from batch output
        outputs = {k: v[0] for k, v in batch_outputs.items()}
        if tile is not None:
            save_tile_plot(out_path, img_name, image, outputs, tile_boxes, hide_label=args.no_label, line_thickness=args.line_thickness)
        logger.info(f"Processed image: {img_name}, found {len(outputs.get('boxes', []))} objects")
        annotated_image = model.annotate_image(outputs, image, hide_label=args.no_label, line_thickness=args.line_thickness)
        if args.json:
            predictions_json.add(os.path.relpath(img_path, imgs_path), image.shape[0], image.shape[1], outputs)

        output_image_path = os.path.join(out_path, img_name)
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, annotated_image_bgr)

    if args.json:
        predictions_json.save(os.path.join(out_path, JSON_NAME))

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
    check_infer_args(parser, args)
    logger.info(f"Arguments: {args}")

    inference_run(args)


if __name__ == "__main__":
    main()
