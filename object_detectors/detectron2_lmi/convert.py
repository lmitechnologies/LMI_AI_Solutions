import logging
import os
import tempfile

import cv2

from lmi_common.trt_convert import onnx_to_trt

logger = logging.getLogger(__name__)


def engine_input_size(cfg, height, width):
    """detectron2's test-time resize of a (height, width) image, rounded to multiples of 32 for the fixed-size engine."""
    import detectron2.data.transforms as T

    if cfg.INPUT.MIN_SIZE_TEST:
        height, width = T.ResizeShortestEdge.get_output_shape(height, width, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    return tuple(max(32, round(x / 32) * 32) for x in (height, width))


def write_engine_sample(sample_image, config_file, path):
    """Write ``sample_image`` resized to the engine input size; the ONNX is traced and the anchors built at this size."""
    from .converter.detectron2_exporter import setup_cfg

    image = cv2.imread(sample_image)
    if image is None:
        raise ValueError(f"Could not read image {sample_image}")
    height, width = engine_input_size(setup_cfg({"config_file": config_file}), *image.shape[:2])
    logger.info(f"Engine input size (h, w): {height}, {width}; sample image was {image.shape[:2]}")
    cv2.imwrite(path, cv2.resize(image, (width, height)))


def convert(args):
    from .cli import DET2_ONNX_EXPORT, DET2_PT_EXPORT, DET2_TRT_EXPORT

    output = args["output"]
    os.makedirs(output, exist_ok=True)
    args["onnx_file_path"] = os.path.join(output, DET2_ONNX_EXPORT)
    args["trt_file_path"] = os.path.join(output, DET2_TRT_EXPORT)
    args["pt_file_path"] = os.path.join(output, DET2_PT_EXPORT)

    if args.get("pt", False):
        from .converter.detectron2_exporter import det2export

        args["format"] = "pt"
        det2export(args)

    # the engine is built from the ONNX
    if args.get("onnx", False) or args.get("trt", False):
        from .converter.detectron2_exporter import det2export
        from .converter.detectron2_onnx_trtonnx import onnx_gs

        with tempfile.TemporaryDirectory() as tmp:
            sample = os.path.join(tmp, "engine_sample.png")
            write_engine_sample(args["sample_image"], args["config_file"], sample)
            onnx_args = {**args, "format": "onnx", "sample_image": sample}
            det2export(onnx_args)
            onnx_gs(onnx_args)

    if args.get("trt", False):
        onnx_to_trt(
            args["onnx_file_path"],
            args["trt_file_path"],
            fp16=args.get("fp16", False),
            workspace_gb=args.get("workspace_size", 4),
        )
