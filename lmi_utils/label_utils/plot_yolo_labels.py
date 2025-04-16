import cv2
import argparse
import glob
import os
import numpy as np
from label_utils.plot_utils import plot_one_polygon, plot_one_pt
import json

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# This helper function must exist; it converts segment coordinates to bounding boxes.
def segments2boxes(segments):
    # Dummy implementation: replace with your actual conversion code.
    # For example, this could compute the bounding box from a set of (x,y) points.
    boxes = []
    for seg in segments:
        x_coords = seg[:, 0]
        y_coords = seg[:, 1]
        x_min, y_min = x_coords.min(), y_coords.min()
        x_max, y_max = x_coords.max(), y_coords.max()
        # Convert to YOLO format (center_x, center_y, width, height)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        boxes.append(np.array([cx, cy, w, h]))
    return np.array(boxes)

def load_yolo_labels(lb_file, im_file, shape, keypoint=False, nkpt=0, ndim=2, num_cls=1, prefix=""):
    """
    Loads a YOLO label txt file and returns a dictionary containing label information.

    Args:
        lb_file (str): Path to the YOLO label file.
        im_file (str): Name or path of the corresponding image file.
        shape (tuple): Image shape (e.g., (height, width, channels)).
        keypoint (bool): If True, labels include keypoint data.
        nkpt (int): Number of keypoints (used if keypoint=True).
        ndim (int): Dimensionality of keypoints (usually 2).
        num_cls (int): Total number of classes in the dataset.
        prefix (str): Optional prefix string for warning messages.

    Returns:
        dict: A dictionary with the following keys:
            - "im_file": the image file name/path.
            - "labels": a numpy array of labels, each row as [class, x, y, w, h].
            - "shape": the provided image shape.
            - "segments": if applicable, the segment coordinates converted to bounding boxes; else None.
            - "keypoints": keypoints array if keypoint=True; else None.
            - "nm": flag indicating label missing (1 if missing, else 0).
            - "nf": flag indicating label found (1 if found, else 0).
            - "ne": flag indicating label empty (1 if empty, else 0).
            - "nc": the dataset class count.
            - "msg": any warning message generated during processing.
    """
    segments = None
    keypoints = None
    nm = 0  # label missing flag
    nf = 0  # label found flag
    ne = 0  # label empty flag
    msg = ""
    logger.info(f"Loading YOLO labels from {lb_file}")
    logger.info(f'keypoint: {keypoint}, nkpt: {nkpt}, ndim: {ndim}, num_cls: {num_cls}')
    if os.path.isfile(lb_file):
        nf = 1  # label found
        with open(lb_file) as f:
            # Read each non-empty line and split by whitespace
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            # If any label line has more than 6 values and keypoint data is not expected,
            # assume these are segment labels and convert segments to boxes.
            if any(len(x) > 6 for x in lb) and (not keypoint):
                classes = np.array([x[0] for x in lb], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), axis=1)
            lb = np.array(lb, dtype=np.float32)
            classes = lb[:, 0]
            if len(classes) == 0:
                msg = f"{prefix}WARNING ⚠️ {im_file}: negative class labels {classes[classes < 0]}"
        nl = len(lb)
        if nl:
            if keypoint:
                expected_cols = 5 + nkpt * ndim
                assert lb.shape[1] == expected_cols, f"labels require {expected_cols} columns each"
                points = lb[:, 5:].reshape(-1, ndim)[:, :2]
            else:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
            assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
            assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

            # Verify that the maximum class id is within range.
            max_cls = lb[:, 0].max()
            assert max_cls < num_cls, (
                f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                f"Possible class labels are 0-{num_cls - 1}"
            )
            # Remove duplicate rows if any.
            _, i = np.unique(lb, axis=0, return_index=True)
            if len(i) < nl:
                lb = lb[i]
                if segments is not None:
                    segments = [segments[x] for x in i]
                msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
        else:
            ne = 1  # label empty
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
    else:
        nm = 1  # label missing
        lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
    
    if keypoint:
        keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
        if ndim == 2:
            kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
            keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)
    
    # Keep only the first 5 columns (class and bounding box info)
    lb = lb[:, :5]
    nc = num_cls  # number of classes
    
    return {
        "classes": classes,
        "im_file": im_file,
        "labels": lb,
        "shape": shape,
        "segments": segments,
        "keypoints": keypoints,
        "nm": nm,
        "nf": nf,
        "ne": ne,
        "nc": nc,
        "msg": msg,
    }

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset", type=str, help="path to the root folder contains images and labels")
    parser.add_argument("--path_out", type=str, help="path to the output directory")
    parser.add_argument('--kp', action='store_true', help='whether the labels contain keypoint data')
    parser.add_argument('--nkpt', type=int, default=0, help='number of keypoints')

    print(f'args: {parser.parse_args()}')
    args = parser.parse_args()
    txt_files = glob.glob(os.path.join(args.path_dataset, "labels/train/*.txt"))
    print(f"Found {len(txt_files)} label files.")
    os.makedirs(args.path_out, exist_ok=True)
    # load class map json
    class_map = {}
    with open(os.path.join(args.path_dataset, "class_map.json")) as f:
        class_map = json.load(f)
    num_classes = len(class_map)
    id_to_class = {v: k for k, v in class_map.items()}
    color_map ={}
    for txt_file in txt_files:
        logger.info(f"Processing {txt_file}")
        image_path = txt_file.replace(".txt", ".png")
        image_path = image_path.replace("labels/train", "images/train")
        if not os.path.isfile(image_path):
            image_path = image_path.replace("images/train", "images/val")
            if not os.path.isfile(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        labels = load_yolo_labels(txt_file, image_path, (h, w), num_cls=num_classes, keypoint=args.kp, nkpt=args.nkpt)
        for id in labels['classes']:
            if id not in color_map:
                color_map[id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        # logger.info(f"Labels: {labels['keypoints']}")
        if labels['segments'] is not None:
            for idx, segment in enumerate(labels["segments"]):
                for i in range(len(segment)):
                    segment[i][0] = segment[i][0] * w
                    segment[i][1] = segment[i][1] * h
                class_id = int(labels["classes"][idx])
                color = color_map[class_id]
                plot_one_polygon(segment, image, color=color, label=id_to_class[int(class_id)])
        
        if labels['keypoints'] is not None:
            for idx, keypoint in enumerate(labels["keypoints"]):
                for i in range(len(keypoint)):
                    keypoint[i][0] = keypoint[i][0] * w
                    keypoint[i][1] = keypoint[i][1] * h
                class_id = int(labels["classes"][idx])
                color = color_map[class_id]
                keypoint = keypoint[:, :2].astype(np.int32)
                for kpt in keypoint:
                    plot_one_pt(kpt, image, color=color, label=id_to_class[int(class_id)])
                
        
        if labels['segments'] is None:
            for label in labels['labels']:
                cx, cy, width, height = label[1], label[2], label[3], label[4]
                x1 = cx - width/2
                y1 = cy - height/2
                x2 = cx + width/2
                y2 = cy + height/2
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                class_id = int(label[0])
                color = color_map[class_id]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image, id_to_class[class_id], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                

        
        cv2.imwrite(os.path.join(args.path_out, os.path.basename(image_path)), image)