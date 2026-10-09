import json
import logging
import os

from lmi_common.model_metadata import embed_onnx_metadata
from lmi_common.trt_convert import onnx_to_trt

logger = logging.getLogger(__name__)

CLASS_MAP_NAME = "class_map.json"
INPUT_FORMATS = ("RGB", "BGR")


def check_input_format(input_format: str) -> None:
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"Unsupported INPUT.FORMAT '{input_format}'; only {' and '.join(INPUT_FORMATS)} are supported")


def convert(args):
    from .cli import DET2_ONNX_EXPORT, DET2_PT_EXPORT, DET2_TRT_EXPORT

    # the engine is built from the ONNX
    build_onnx = args.get("onnx", False) or args.get("trt", False)
    image_size = args.get("image_size")
    if build_onnx and (not image_size or len(image_size) != 2 or any(x <= 0 or x % 32 for x in image_size)):
        raise ValueError(f"ONNX/TensorRT conversion needs image_size (h, w) in positive multiples of 32, got {image_size}")

    output = args["output"]
    os.makedirs(output, exist_ok=True)
    args["onnx_file_path"] = os.path.join(output, DET2_ONNX_EXPORT)
    args["trt_file_path"] = os.path.join(output, DET2_TRT_EXPORT)
    args["pt_file_path"] = os.path.join(output, DET2_PT_EXPORT)

    if args.get("pt", False):
        from .converter.detectron2_exporter import det2export

        args["format"] = "pt"
        det2export(args)

    if build_onnx:
        from .converter.detectron2_exporter import det2export, setup_cfg
        from .converter.detectron2_onnx_trtonnx import onnx_gs

        # fail on bad metadata before the export
        metadata = export_metadata(setup_cfg(args), args.get("class_map"), args["config_file"])
        args["format"] = "onnx"
        det2export(args)
        onnx_gs(args)
        # graph surgery drops metadata_props
        embed_onnx_metadata(args["onnx_file_path"], metadata)

    if args.get("trt", False):
        onnx_to_trt(
            args["onnx_file_path"],
            args["trt_file_path"],
            fp16=args.get("fp16", False),
            workspace_gb=args.get("workspace_size", 4),
        )


def export_metadata(cfg, class_map_path, config_file) -> dict:
    """The payload embedded in the ONNX and engine: the input color order, plus the class names when a class map is found.

    Args:
        cfg: The detectron2 config.
        class_map_path: A class map json ({"0": name, ...}), or None to use the class_map.json beside ``config_file`` if any.
        config_file: The config file path.
    """
    check_input_format(cfg.INPUT.FORMAT)
    metadata = {"input_format": cfg.INPUT.FORMAT}
    if class_map_path is None:
        class_map_path = os.path.join(os.path.dirname(config_file), CLASS_MAP_NAME)
        if not os.path.isfile(class_map_path):
            logger.warning(f"No {CLASS_MAP_NAME} beside {config_file}; exporting without class names, so loading needs class_map")
            return metadata
    with open(class_map_path) as f:
        raw = json.load(f)
    try:
        class_map = {int(k): v for k, v in raw.items()}
    except (AttributeError, ValueError):
        class_map = None
    if class_map is None or not all(isinstance(v, str) for v in class_map.values()):
        raise ValueError(f'{class_map_path} must map 0-based class ids to names, e.g. {{"0": "dent", "1": "scratch"}}')
    num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    if sorted(class_map) != list(range(num_classes)):
        raise ValueError(f"{class_map_path} must map exactly the ids 0..{num_classes - 1}, to match MODEL.ROI_HEADS.NUM_CLASSES")
    metadata["class_names"] = [class_map[i] for i in range(num_classes)]
    return metadata
