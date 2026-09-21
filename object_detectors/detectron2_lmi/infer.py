import argparse
import glob
import json
import logging
import os
import time

import cv2

from object_detectors.od_core.infer_cli import JSON_NAME, PredictionsJson, add_infer_args, check_infer_args, predict_tiled, save_tile_plot

# setup the logger
logger = logging.getLogger(__name__)


def find_images(path: str, exts=("jpg", "jpeg", "png")):
    """find all images with the given extensions in the path

    Args:
        path (str): the input path
        exts (list): the list of extensions

    Returns:
        list: the list of image paths
    """
    import os

    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(path, f"*.{ext}")))
    return imgs


def add_args(parser: argparse.ArgumentParser, **defaults) -> None:
    """Add the inference flags. ``defaults`` may give weights, input, output and class_map paths."""
    add_infer_args(parser, confidence=0.5, weights=defaults.get("weights"), input=defaults.get("input"), output=defaults.get("output"))
    class_map = defaults.get("class_map")
    parser.add_argument("-m", "--class_map", default=class_map, required=class_map is None, help="the path to the class map json file")


def inference_run(args):
    model_path = args.get("weights")
    imgs_path = args.get("input")
    out_path = args.get("output")
    class_map_path = args.get("class_map")
    confidence = args.get("confidence")
    tile = args.get("tile_step")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(class_map_path, "r") as f:
        class_map = json.load(f)

    from object_detectors.detectron2_lmi.model import Detectron2Model  # imports torch and detectron2: keep cli.py startup fast

    # load model
    size = {"image_size": args["image_size"]} if args.get("image_size") else {}  # .pt defaults to 640x640; an engine has its own
    model = Detectron2Model(model_path, class_map=class_map, **size)

    # model warmup
    model.warmup()

    # find images
    imgs = find_images(imgs_path)
    predictions_json = PredictionsJson(model.class_map.values())

    for img_path in imgs:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        if tile is not None:
            outputs, _, tile_boxes = predict_tiled(model, img, tile, configs=confidence, return_segments=True)
        else:
            outputs, _ = model.predict(img, configs=confidence, return_segments=True)
        t1 = time.time()
        outputs = {k: v[0] for k, v in outputs.items()}  # get the first batch output
        if tile is not None:
            save_tile_plot(
                out_path,
                os.path.basename(img_path),
                img,
                outputs,
                tile_boxes,
                hide_label=args.get("no_label"),
                line_thickness=args.get("line_thickness"),
            )
        if args.get("json"):
            predictions_json.add(os.path.relpath(img_path, imgs_path), img.shape[0], img.shape[1], outputs)

        n_boxes = len(outputs["boxes"])
        if n_boxes == 0:
            logger.warning(f"No detections found for image: {img_path}")
            continue
        logger.info(f"Found {n_boxes} detections for image: {os.path.basename(img_path)} in {t1 - t0:.2f} seconds")

        # save the image
        annotated_image = model.annotate_image(outputs, img, hide_label=args.get("no_label"), line_thickness=args.get("line_thickness"))
        fname = os.path.basename(img_path)
        out_img_path = os.path.join(out_path, fname)
        cv2.imwrite(out_img_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if args.get("json"):
        predictions_json.save(os.path.join(out_path, JSON_NAME))


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Detectron2-LMI Inference")
    add_args(parser)
    args = parser.parse_args()
    check_infer_args(parser, args)
    inference_run(vars(args))


if __name__ == "__main__":
    main()
