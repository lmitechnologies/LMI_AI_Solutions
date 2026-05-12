import argparse
import logging
import os


def run_cli(model_cls):
    """Argparse-based CLI for AnomalyModel test/convert actions."""
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest="action", required=True, help="Action modes: test or convert")

    test_ap = subs.add_parser("test", help="test model")
    test_ap.add_argument("-i", "--model_path", default="/app/model/model.pt", help="Input model file path.")
    test_ap.add_argument("-d", "--data_dir", default="/app/data", help="Data file directory.")
    test_ap.add_argument("-o", "--annot_dir", default="/app/annotation_results", help="Annot file directory.")
    test_ap.add_argument("-g", "--generate_stats", action="store_true", help="generate the data stats")
    test_ap.add_argument("-p", "--plot", action="store_true", help="plot the annotated images")
    test_ap.add_argument("-t", "--ad_threshold", type=float, default=None, help="AD patch threshold.")
    test_ap.add_argument("-m", "--ad_max", type=float, default=None, help="AD patch max anomaly.")
    test_ap.add_argument("--tile", type=int, nargs=2, default=None, help="tile size (h,w)")
    test_ap.add_argument("--stride", type=int, nargs=2, default=None, help="stride size (h,w)")
    test_ap.add_argument(
        "-om",
        "--overlap_mode",
        default="gaussian",
        help='overlap blending mode for tiling: "average", "max", "cosine", "linear", "gaussian"',
    )
    test_ap.add_argument(
        "--scale_mode", default="padding", choices=["padding", "interpolation"], help="tile scaling mode: padding or interpolation"
    )
    test_ap.add_argument("--limit", type=int, default=None, help="process only the first N images")

    convert_ap = subs.add_parser("convert", help="convert model to trt engine")
    convert_ap.add_argument("-i", "--model_path", default="/app/model/model.pt", help="Input model file path.")
    convert_ap.add_argument("-o", "--export_dir", default="/app/export")
    convert_ap.add_argument("-c", "--convert_type", default="trt", choices=["trt", "onnx"], help="convert type: trt or onnx")
    convert_ap.add_argument("--fp32", action="store_true", help="disable fp16 and use fp32 for TRT conversion")
    args = vars(ap.parse_args())

    action = args["action"]
    model_path = args["model_path"]
    ad = model_cls(model_path)

    if action == "convert":
        export_dir = args["export_dir"]
        os.makedirs(export_dir, exist_ok=True)
        if args["convert_type"] == "onnx":
            ad.export_onnx(os.path.join(export_dir, "model.onnx"))
        else:
            ad.export_trt(export_dir, fp16=not args["fp32"])
    elif action == "test":
        os.makedirs(args["annot_dir"], exist_ok=True)
        ad.test(
            args["data_dir"],
            args["annot_dir"],
            args["generate_stats"],
            args["plot"],
            args["ad_threshold"],
            args["ad_max"],
            args["tile"],
            args["stride"],
            args["overlap_mode"],
            args["scale_mode"],
            args["limit"],
        )
