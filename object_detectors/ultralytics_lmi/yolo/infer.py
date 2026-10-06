import collections
import logging
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from lmi_utils.gadget_utils.pipeline_utils import (
    _reconstructor,
    fit_im_to_size,
    get_img_path_batches,
    plot_one_box,
    plot_one_rbox,
    resize_image,
    revert_to_origin,
)
from lmi_utils.preprocess_utils import steps
from object_detectors.od_core.infer_cli import JSON_NAME, PredictionsJson, add_infer_args, check_infer_args, predict_tiled, save_tile_plot
from object_detectors.ultralytics_lmi.yolo.model import Yolo, YoloObb, YoloPose, YoloSeg

BATCH_SIZE = 1
COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (102, 51, 153),
    (255, 140, 0),
    (105, 105, 105),
    (127, 25, 27),
    (9, 200, 100),
]


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    add_infer_args(parser, confidence=0.25)
    parser.add_argument(
        "--obb",
        action="store_true",
        help="[optional] whether to run Oriented Bounding Box model",
    )
    parser.add_argument("--pose", action="store_true", help="[optional] whether to run Pose model")
    parser.add_argument("--seg", action="store_true", help="[optional] whether to run Segmentation model")
    parser.add_argument(
        "--el",
        action="store_false",
        help="[optional] log level default is ERROR",
        default=False,
    )
    parser.add_argument("--resize", required=False, nargs=2, type=int, help="resize")
    parser.add_argument("--no-box", action="store_true", help="[optional] do not show bounding box")
    parser.add_argument("--pad", required=False, nargs=2, type=int, help="pad")
    args = parser.parse_args()
    check_infer_args(parser, args)
    tile = args.tile_step
    if tile is not None and (args.obb or args.pose):
        parser.error("--tile does not support --obb or --pose")
    if tile is not None and (args.resize or args.pad):
        parser.error("--tile cannot be combined with --resize or --pad")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    if args.el:
        logger.propagate = False

    model_cls = YoloPose if args.pose else YoloObb if args.obb else YoloSeg if args.seg else Yolo
    model = model_cls(args.weights, image_size=args.image_size)
    sz = model.image_size

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # get color map
    color_map = {}
    for k in sorted(model.names.keys()):
        v = model.names[k]
        if len(color_map) < len(COLORS):
            color_map[v] = COLORS[len(color_map)]
        else:
            color_map[v] = tuple([random.randint(0, 255) for _ in range(3)])

    # warm up
    t1 = time.time()
    model.warmup(sz)
    t2 = time.time()
    logger.info(f"warmup input shape: {sz}")
    logger.info(f"warmup proc time -> {t2 - t1:.4f}")

    predictions_json = PredictionsJson(model.names.values())
    batches = get_img_path_batches(batch_size=BATCH_SIZE, img_dir=args.input, fmt="jpg") + get_img_path_batches(
        batch_size=BATCH_SIZE, img_dir=args.input, fmt="png"
    )
    logger.info(f"loaded {len(batches)} with a batch size of {BATCH_SIZE}")

    for batch in tqdm(batches):
        for p in batch:
            t1 = time.time()
            # load image
            im0 = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # BGR format
            if len(im0.shape) == 2:
                im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

            # warp image
            operators = []
            im1 = im0
            fname = os.path.basename(p)
            save_path = os.path.join(args.output, fname)
            if args.resize:
                im1 = resize_image(
                    im=im0,
                    H=args.resize[0] if args.resize[0] != 0 else None,
                    W=args.resize[1] if args.resize[1] != 0 else None,
                )
                logger.warning(f"{im1.shape}, resizing")
                operators.append(
                    steps.revert_resize(
                        src_sizes=[[im0.shape[1], im0.shape[0]]],
                        dst_sizes=[[im1.shape[1], im1.shape[0]]],
                        pads=[[0, 0, 0, 0]],
                    )
                )

            if args.pad:
                im1, pad_L, pad_R, pad_T, pad_B = fit_im_to_size(im=im1, H=args.pad[0], W=args.pad[1])
                operators.append(steps.revert_pad(pads=[[pad_L, pad_R, pad_T, pad_B]]))
                logger.warning(f"{im1.shape}, padding")

            if tile is not None:
                rh, rw = 1.0, 1.0  # predict_tiled returns results in image coordinates
            elif sz[0] != im1.shape[0] or sz[1] != im1.shape[1]:
                logger.warning(f"{im1.shape}, warping")
                rh, rw = sz[0] / im0.shape[0], sz[1] / im0.shape[1]
                im1 = cv2.resize(im0, (sz[1], sz[0]))
            else:
                rh, rw = 1.0, 1.0

            # inference
            if tile is not None:
                results, _, tile_boxes = predict_tiled(model, im0, tile, configs=args.confidence)
                outputs = {k: v[0] for k, v in results.items()}
                save_tile_plot(args.output, fname, im0, outputs, tile_boxes, hide_label=args.no_label, line_thickness=args.line_thickness)
            else:
                results, _ = model.predict(im1, configs=args.confidence)
            t2 = time.time()

            im_out = np.copy(im0)
            final = collections.defaultdict(list)  # per-instance results in image coordinates, for the json

            if len(results["boxes"]):
                # uppack results for a single image
                use_revert_to_origin = len(operators) > 0

                boxes, scores, classes = (
                    results["boxes"][0],
                    results["scores"][0],
                    results["classes"][0],
                )
                masks = results["masks"][0] if "masks" in results else None
                segments = results["segments"][0] if "segments" in results else []
                points = results["points"][0] if "points" in results else []
                if use_revert_to_origin:
                    boxes = revert_to_origin(boxes, operators)
                    if masks is not None and len(masks):
                        # revert the whole mask stack once via the coordinate path (bilinear + re-threshold)
                        masks = _reconstructor().reconstruct_coordinates({"masks": [masks]}, operators)["masks"][0]

                # loop through each box
                for j in range(len(boxes) - 1, -1, -1):
                    # convert box,mask to original image size
                    mask = None
                    if masks is not None:
                        mask = masks[j]
                        if not use_revert_to_origin:
                            mask = cv2.resize(mask, (im_out.shape[1], im_out.shape[0]))
                    box = boxes[j]
                    if not use_revert_to_origin:
                        if args.obb:
                            for b in range(len(box)):
                                box[b] = [box[b][0] / rw, box[b][1] / rh]
                        else:
                            box[[0, 2]] /= rw
                            box[[1, 3]] /= rh

                    final["boxes"].append(box.copy())
                    final["scores"].append(scores[j])
                    final["classes"].append(classes[j])
                    if mask is not None:
                        final["masks"].append(mask)
                    box = box.astype(np.int32)
                    # annotation
                    color = color_map[classes[j]]
                    label = None if args.no_label else f"{classes[j]}: {scores[j]:.2f}"
                    if args.obb:
                        plot_one_rbox(box, im_out, color=color, label=label, line_thickness=args.line_thickness, hide_bbox=args.no_box)
                    else:
                        plot_one_box(
                            box,
                            im_out,
                            mask,
                            color=color,
                            label=label,
                            line_thickness=args.line_thickness,
                            hide_bbox=args.no_box,
                        )

                    if segments and len(segments[j]):
                        seg = segments[j]
                        # convert segments to original image size
                        if not use_revert_to_origin:
                            seg[:, 0] /= rw
                            seg[:, 1] /= rh
                        else:
                            seg = revert_to_origin(seg, operators)
                        cv2.drawContours(im_out, [seg.reshape((-1, 1, 2)).astype(np.int32)], -1, color, 1)

                    if len(points):
                        pts = points[j]
                        # convert points to original image size
                        if not use_revert_to_origin:
                            pts[:, 0] /= rw
                            pts[:, 1] /= rh
                        else:
                            pts = revert_to_origin(pts, operators)
                        final["points"].append(pts.copy())
                        pts = pts.astype(np.int32)
                        for pt in pts[:, :2]:  # a third column, when present, is visibility
                            cv2.circle(im_out, tuple(pt), 4, color, -1)

                # log
                cnts = collections.Counter(classes)
                logger.info(f"fname: {fname}")
                for c in cnts:
                    logger.info(f"found {cnts[c]} {c}")
            else:
                logger.info(f"fname: {fname} --- no object detected")
            if args.json:
                predictions_json.add(os.path.relpath(p, args.input), im0.shape[0], im0.shape[1], final)
            # save output image from RGB to BGR
            cv2.imwrite(save_path, im_out[:, :, ::-1])
            t3 = time.time()
            logger.info(f"proc time: {t2 - t1:.4f}, cycle time: {t3 - t1:.4f}\n")

    if args.json:
        predictions_json.save(os.path.join(args.output, JSON_NAME))
