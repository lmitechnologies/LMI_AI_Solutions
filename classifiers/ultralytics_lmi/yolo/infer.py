import logging
import os

import cv2

from classifiers.cls_core.infer_cli import add_infer_args, check_infer_args, predict_tiled, save_tile_plot
from classifiers.ultralytics_lmi.yolo.model import YoloCls
from lmi_utils.gadget_utils.pipeline_utils import get_img_path_batches

BATCH_SIZE = 1


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    add_infer_args(parser)
    args = parser.parse_args()
    check_infer_args(parser, args)

    size = args.image_size

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

    tiling = args.tile_step is not None
    if tiling:
        logger.info(f"tiling enabled: tile={args.tile_step.tile_size}, stride={args.tile_step.stride}")

    batches = get_img_path_batches(batch_size=BATCH_SIZE, img_dir=args.input, fmt="jpg") + get_img_path_batches(
        batch_size=BATCH_SIZE, img_dir=args.input, fmt="png"
    )
    logger.info(f"loaded {len(batches)} with a batch size of {BATCH_SIZE}")
    for batch in batches:
        for p in batch:
            t1 = time.time()
            fname = os.path.basename(p)
            im0 = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # BGR format
            if len(im0.shape) == 2:
                im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

            if tiling:
                results, _, tile_boxes = predict_tiled(model, im0, args.tile_step)
                save_tile_plot(args.output, fname, im0, tile_boxes)
            else:
                results, _ = model.predict(im0)
            t2 = time.time()

            cls, conf = results["classes"][0], results["scores"][0]
            logger.info(f"file: {fname}")
            logger.info(f"class: {cls}, conf: {conf:.4f}")

            out_path = os.path.join(args.output, cls)
            os.makedirs(out_path, exist_ok=True)
            save_path = os.path.join(out_path, fname)
            cv2.imwrite(save_path, im0[:, :, ::-1])
            t3 = time.time()
            logger.info(f"proc time: {t2 - t1:.4f}, cycle time: {t3 - t1:.4f}\n")
