import argparse
import json
import logging
from pathlib import Path

import cv2

from lmi_utils.label_utils.csv_labels.csv_utils import load_csv
from lmi_utils.label_utils.csv_labels.shapes import Mask, Rect
from lmi_utils.label_utils.lst_project import CONTROL_NAMES, XML_NAME, labeling_interface, to_image_root

logger = logging.getLogger(__name__)


OUT_NAME = "lst.json"
RECT_NAME = CONTROL_NAMES["rectanglelabels"][0]
POLYGON_NAME = CONTROL_NAMES["polygonlabels"][0]


def rect_to_lst(rect_obj, width, height, is_pred):
    x1, y1 = rect_obj.up_left
    x2, y2 = rect_obj.bottom_right
    w, h = x2 - x1, y2 - y1
    box = {
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": x1 / width * 100,
            "y": y1 / height * 100,
            "width": w / width * 100,
            "height": h / height * 100,
            "rotation": rect_obj.angle,
            "rectanglelabels": [rect_obj.category],
        },
        "from_name": RECT_NAME,
        "to_name": "image",
        "type": "rectanglelabels",
    }
    if is_pred:
        box["value"]["score"] = rect_obj.confidence
    return box


def mask_to_lst(mask_obj, width, height, is_pred):
    X, Y = mask_obj.X, mask_obj.Y
    polygon = {
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {"points": [], "polygonlabels": [mask_obj.category]},
        "from_name": POLYGON_NAME,
        "to_name": "image",
        "type": "polygonlabels",
    }
    for i in range(0, len(X)):
        x = X[i] / width * 100
        y = Y[i] / height * 100
        polygon["value"]["points"].append([x, y])
    if is_pred:
        polygon["value"]["score"] = mask_obj.confidence
    return polygon


def init_label_obj(path_img, is_pred):
    # TODO: support BOTH annotations and predictions
    label_obj = {}
    if not is_pred:
        label_obj["annotations"] = [{"result": []}]
    else:
        label_obj["predictions"] = [{"model_version": "prediction", "result": []}]
    label_obj["data"] = {"image": path_img}
    return label_obj


def write_to_lst(shapes: dict, out_path: Path, img_dir: Path, gs_path: str, is_pred: bool):
    image_root = to_image_root(gs_path)

    labels = []
    cnt_box = 0
    cnt_polygon = 0
    cnt_img = len(shapes)
    # sort the shapes by filename
    shapes = {k: v for k, v in sorted(shapes.items(), key=lambda item: item[0])}
    img_dt = dict(
        (p.name, p)
        for p in img_dir.rglob("*")
        if p.suffix.lower()
        in [
            ".jpg",
            ".jpeg",
            ".png",
        ]
    )
    for fname in shapes:
        if fname not in img_dt:
            raise Exception(f"Image {fname} not found in {str(img_dir)}")

        img_path = img_dt[fname]
        im = cv2.imread(img_path)
        height, width = im.shape[:2]

        gs_fname = image_root + fname
        label_obj = init_label_obj(gs_fname, is_pred)
        target = "predictions" if is_pred else "annotations"

        for shape in shapes[fname]:
            if isinstance(shape, Rect):
                box = rect_to_lst(shape, width, height, is_pred)
                label_obj[target][0]["result"].append(box)
                cnt_box += 1
            elif isinstance(shape, Mask):
                polygon = mask_to_lst(shape, width, height, is_pred)
                label_obj[target][0]["result"].append(polygon)
                cnt_polygon += 1
            else:
                raise Exception(f"Invalid shape type: {type(shape)}")
        labels.append(label_obj)

    # save to json
    with open(out_path / OUT_NAME, "w") as f:
        json.dump(labels, f, indent=4)

    # write the xml file
    with open(out_path / XML_NAME, "w") as f:
        f.write(labeling_interface(labels))

    logger.info(f"Number of images: {cnt_img}")
    logger.info(f"Number of boxes: {cnt_box}")
    logger.info(f"Number of polygons: {cnt_polygon}")
    logger.info(
        f'In the label studio labeling interface, ensure that PolygonLabels name="{POLYGON_NAME}" and RectangleLabels name="{RECT_NAME}"'
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser(description="Convert csv annotations to label studio lst format")
    ap.add_argument("--path_imgs", "-i", type=Path, help="path to the images")
    ap.add_argument("--csv", required=True, type=Path, help="path to the csv file")
    ap.add_argument(
        "--gs_path",
        type=str,
        required=True,
        help="the gs path will be output in the json. Either a gs path (start with gs://) or local absolute path (start with /)",
    )
    ap.add_argument("--out_dir", "-o", type=Path, required=True, help="the output directory")
    ap.add_argument("--pred", action="store_true", help="if the csv file is a prediction file")
    args = ap.parse_args()

    if args.out_dir.is_file():
        raise Exception("The output path should be a directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shapes = load_csv(args.csv)[0]
    write_to_lst(shapes, args.out_dir, args.path_imgs, args.gs_path, args.pred)
