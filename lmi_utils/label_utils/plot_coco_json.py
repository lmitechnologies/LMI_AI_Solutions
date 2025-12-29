import os
import argparse
import logging
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from label_utils.bbox_utils import rotate
from label_utils.plot_utils import plot_one_polygon, plot_one_brush, get_distinct_colors
from system_utils import path_utils


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_annotations_from_json(path_json, path_imgs, path_out):
    coco = COCO(path_json)
    category = coco.dataset["categories"]
    colors = get_distinct_colors(len(category))
    colormap = {cat["id"]: colors[i] for i, cat in enumerate(category)}
    relative_paths = path_utils.get_relative_paths(path_imgs, recursive=True)
    fnames = [os.path.basename(p) for p in relative_paths]
    for m in coco.imgs.values():
        fname = m["file_name"]
        idx = fnames.index(fname)
        path_img = os.path.join(path_imgs, relative_paths[idx])
        if not os.path.isfile(path_img):
            raise Exception(f"Cannot find the file: {path_img}")

        # load img
        im = cv2.imread(path_img)

        # get annotations
        img_id = m["id"]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        for annot in annotations:
            bbox = annot["bbox"]
            cat_id = int(annot["category_id"])
            cat_name = coco.loadCats(cat_id)[0]["name"]
            img_id = annot["image_id"]
            segs = annot["segmentation"]
            if "keypoints" in annot:
                logger.warning("Does not support keypoints")

            if len(bbox) == 5:
                x, y, w, h, angle = bbox
                angle = float(angle)
            else:
                x, y, w, h = bbox
                angle = 0

            # plot box
            x, y, w, h = list(map(int, [x, y, w, h]))
            pts = rotate(x, y, w, h, angle, unit="radian", rot_center="center")
            pts = pts.reshape((-1, 1, 2))
            plot_one_polygon(pts, im, label=f"{cat_name}", color=colormap[cat_id])

            # plot segments
            if isinstance(segs, list):
                for seg in segs:
                    pts = np.array(list(map(int, seg)))
                    pts = pts.reshape((-1, 1, 2))
                    plot_one_polygon(pts, im, label=f"{cat_name}", color=colormap[cat_id])
            elif isinstance(segs, dict):
                mask = coco_mask.decode(segs)
                ys, xs = np.where(mask)
                plot_one_brush(xs, ys, im, label=f"{cat_name}", color=colormap[cat_id])
            else:
                raise Exception(f"Unknown segmentation type: {type(segs)}")

        # write the annotated image
        path = os.path.join(path_out, fname)
        logger.info(f"writing to {path}")
        cv2.imwrite(path, im)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--path_imgs", required=True)
    ap.add_argument("-j", "--path_json", required=True)
    ap.add_argument("-o", "--path_out", required=True, help="output path")
    args = ap.parse_args()

    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)

    get_annotations_from_json(args.path_json, args.path_imgs, args.path_out)
