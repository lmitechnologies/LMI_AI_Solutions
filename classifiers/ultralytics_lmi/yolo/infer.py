import logging
import os

import cv2
import numpy as np

from classifiers.ultralytics_lmi.yolo.model import YoloCls
from lmi_utils.gadget_utils.pipeline_utils import get_img_path_batches

BATCH_SIZE = 1


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--weights",
        required=True,
        help='the path to the model weights file. The type of supported files are: ".pt" or ".engine"',
    )
    parser.add_argument("-i", "--input", required=True, help="the path to the input images")
    parser.add_argument("-o", "--output", required=True, help="the path to the output folder")
    parser.add_argument(
        "-s",
        "--image_size",
        nargs="+",
        type=int,
        metavar="N",
        help="[optional] the model input size: one int for a square, or two ints: h w. By default it is read from the model",
    )
    args = parser.parse_args()
    if args.image_size is not None and (len(args.image_size) not in (1, 2) or min(args.image_size) <= 0):
        parser.error(f"--image_size takes one or two positive ints (h w), got {args.image_size}")
    size = None if args.image_size is None else args.image_size * (3 - len(args.image_size))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = YoloCls(args.weights, image_size=size)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # warm up
    t1 = time.time()
    model.warmup()
    t2 = time.time()
    logger.info(f"warmup input shape: {model.image_size}")
    logger.info(f"warmup proc time -> {t2 - t1:.4f}")

    batches = get_img_path_batches(batch_size=BATCH_SIZE, img_dir=args.input, fmt="jpg") + get_img_path_batches(
        batch_size=BATCH_SIZE, img_dir=args.input, fmt="png"
    )
    logger.info(f"loaded {len(batches)} with a batch size of {BATCH_SIZE}")
    for batch in batches:
        for p in batch:
            t1 = time.time()
            # load image
            fname = os.path.basename(p)
            im0 = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # BGR format
            if len(im0.shape) == 2:
                im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

            # inference: predict() resizes to the model input size
            results, _ = model.predict(im0)
            t2 = time.time()

            cls, conf = results["classes"][0], results["scores"][0]
            logger.info(f"file: {fname}")
            logger.info(f"class: {cls}, conf: {conf:.4f}")

            # save images according to the classes
            out_path = os.path.join(args.output, cls)
            os.makedirs(out_path, exist_ok=True)
            save_path = os.path.join(out_path, fname)
            im_out = np.copy(im0)

            # save output image from RGB to BGR
            cv2.imwrite(save_path, im_out[:, :, ::-1])
            t3 = time.time()
            logger.info(f"proc time: {t2 - t1:.4f}, cycle time: {t3 - t1:.4f}\n")
