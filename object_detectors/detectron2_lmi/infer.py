import glob
import json
import logging
import os
import time

import cv2

from lmi_utils.label_utils.csv_utils import write_to_csv
from lmi_utils.label_utils.shapes import Mask, Rect
from object_detectors.detectron2_lmi.model import Detectron2Model

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


def inference_run(args):
    model_path = args.get("weights")
    imgs_path = args.get("input")
    out_path = args.get("output")
    class_map_path = args.get("class_map")
    confidence = args.get("confidence")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(class_map_path, "r") as f:
        class_map = json.load(f)

    # load model
    model = Detectron2Model(model_path, class_map=class_map)

    # model warmup
    model.warmup()

    # find images
    imgs = find_images(imgs_path)
    results = {}

    for img_path in imgs:
        csv_results = []
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        outputs, _ = model.predict(img, configs=confidence, return_segments=True)
        t1 = time.time()
        outputs = {k: v[0] for k, v in outputs.items()}  # get the first batch output

        n_boxes = len(outputs["boxes"])
        if n_boxes == 0:
            logger.warning(f"No detections found for image: {img_path}")
            continue
        logger.info(f"Found {n_boxes} detections for image: {os.path.basename(img_path)} in {t1 - t0:.2f} seconds")

        # save the image
        annotated_image = model.annotate_image(outputs, img)
        fname = os.path.basename(img_path)
        out_img_path = os.path.join(out_path, fname)
        cv2.imwrite(out_img_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # save to csv file
        for idx, box in enumerate(outputs["boxes"]):
            score = outputs["scores"][idx]
            csv_results.append(
                Rect(
                    im_name=fname,
                    category=outputs["classes"][idx],
                    up_left=box[:2].astype(int).tolist(),
                    bottom_right=box[2:].astype(int).tolist(),
                    confidence=score,
                    angle=0,
                )
            )
            if "segments" in outputs and len(outputs["segments"]) > 0:
                segments = outputs["segments"][idx].astype(int)
                csv_results.append(
                    Mask(
                        im_name=fname,
                        category=outputs["classes"][idx],
                        x_vals=segments[:, 0].tolist(),
                        y_vals=segments[:, 1].tolist(),
                        confidence=score,
                    )
                )

        results[fname] = csv_results
    write_to_csv(results, os.path.join(out_path, "predictions.csv"), overwrite=True)
