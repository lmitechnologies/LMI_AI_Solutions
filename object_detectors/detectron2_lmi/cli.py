import argparse
import logging
import os

logger = logging.getLogger(__name__)


DET2_ONNX_EXPORT = "model.onnx"
DET2_TRT_EXPORT = "model.engine"
DET2_PT_EXPORT = "model.pt"
DET2_DEFAULT_DIR = "/home/weights"
DET2_DATASET_DIR = "/home/data"
DET2_INPUT_DIR = "/home/input"
DET2_OUTPUT_DIR = "/home/output"
DET2_CLASS_MAP = "/home/class_map.json"
DET2_CONFIG_FILE = "config.yaml"
DET2_SAMPLE_IMAGE = "sample_image.png"
DET2_PTH_EXPORT = "model_final.pth"


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest="action", required=True, help="Action modes: train, test or convert")

    wpath = lambda f: os.path.join(DET2_DEFAULT_DIR, f)  # noqa: E731

    train_ap = subs.add_parser("train", help="train model")
    train_ap.add_argument(
        "-c", "--config-file", metavar="FILE", default=os.path.join("/home", DET2_CONFIG_FILE), help="path to config file"
    )
    train_ap.add_argument(
        "--detectron2-config", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="Detectron2 config file"
    )
    train_ap.add_argument("--dataset_dir", type=str, default=DET2_DATASET_DIR, help="Dataset dir")
    train_ap.add_argument("--output", type=str, default=DET2_DEFAULT_DIR, help="Path to the output directory")

    test = subs.add_parser("test", help="test model")
    test.add_argument("-w", "--weights", type=str, default=wpath(DET2_PT_EXPORT), help="The path to the model")
    test.add_argument("-i", "--input", type=str, default=DET2_INPUT_DIR, help="The path to the images")
    test.add_argument("-o", "--output", type=str, default=DET2_OUTPUT_DIR, help="The path to the outputs")
    test.add_argument("--class_map", type=str, default=DET2_CLASS_MAP, help="The path to the class map")
    test.add_argument("--confidence", type=float, default=0.5, help="The confidence threshold")

    convert_ap = subs.add_parser("convert", help="convert model")
    convert_ap.add_argument("-c", "--config-file", metavar="FILE", default=wpath(DET2_CONFIG_FILE), help="path to config file")
    convert_ap.add_argument("-o", "--output", type=str, default=DET2_DEFAULT_DIR, help="The output directory for the converted model")
    convert_ap.add_argument(
        "-w", "--weights", type=str, default=wpath(DET2_PTH_EXPORT), help="The Detectron 2 model weights (.pth or .pkl)"
    )
    convert_ap.add_argument(
        "-s", "--sample_image", type=str, default=wpath(DET2_SAMPLE_IMAGE), help="Sample image for anchors generation/predictions"
    )
    convert_ap.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size for the model")
    convert_ap.add_argument("--fp16", action="store_true", help="Use fp16")
    convert_ap.add_argument("--pt", action="store_true", help="Convert to pt")
    convert_ap.add_argument("--onnx", action="store_true", help="Convert to onnx")
    convert_ap.add_argument("--trt", action="store_true", help="Convert to TensorRT")

    args = vars(ap.parse_args())
    action = args["action"]

    if action == "train":
        from object_detectors.detectron2_lmi.trainer import training_run

        logger.info("Training model")
        training_run(args)
    elif action == "test":
        from object_detectors.detectron2_lmi.infer import inference_run

        inference_run(args)
    elif action == "convert":
        from object_detectors.detectron2_lmi.convert import convert

        weights = args.get("weights", "")
        if not (weights.endswith(".pth") or weights.endswith(".pkl")):
            ap.error(f"Weights file must be a .pth or .pkl file, got: {weights}")
        convert(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
