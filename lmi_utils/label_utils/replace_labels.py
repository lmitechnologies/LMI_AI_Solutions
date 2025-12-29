import argparse
import copy
from pathlib import Path
from label_utils.csv_utils import load_csv, write_to_csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def replace_labels(name_to_shapes, label_mapping):
    out_dt = {}
    for name, shapes in name_to_shapes.items():
        new_shapes = []
        for shape in shapes:
            if shape.category in label_mapping:
                new_shape = copy.deepcopy(shape)
                new_shape.category = label_mapping[shape.category]
            else:
                new_shape = shape
            new_shapes.append(new_shape)
        out_dt[name] = new_shapes
    return out_dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    ap.add_argument("--output", "-o", type=str, required=True, help="Path to output CSV file")
    ap.add_argument(
        "--target_labels",
        "-t",
        type=str,
        nargs="+",
        required=True,
        help="List of target labels to replace",
    )
    ap.add_argument(
        "--new_labels",
        "-n",
        type=str,
        nargs="+",
        required=True,
        help="List of new labels to replace with",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    name_to_shapes, _ = load_csv(input_path)
    label_mapping = dict(zip(args.target_labels, args.new_labels))

    out_dt = replace_labels(name_to_shapes, label_mapping)

    write_to_csv(out_dt, output_path)
