import logging
import os

import tensorrt as trt

trt.init_libnvinfer_plugins(None, "")

logger = logging.getLogger(__name__)


def convert_to_trt(args):
    """Convert an ONNX model to a TensorRT engine using the TensorRT Python API.

    Args:
        args: dict with keys:
            - onnx_file_path (str): Path to the input ONNX model.
            - trt_file_path (str): Path to save the output TensorRT engine.
            - fp16 (bool, optional): Enable FP16 precision. Default False.
            - workspace_size (int, optional): Max workspace size in GB. Default 4.
    """
    onnx_path = args["onnx_file_path"]
    trt_path = args["trt_file_path"]
    fp16 = args.get("fp16", False)
    workspace_gb = args.get("workspace_size", 4)

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if fp16:
        if not builder.platform_has_fast_fp16:
            logger.warning("FP16 requested but not natively supported on this platform")
        config.set_flag(trt.BuilderFlag.FP16)

    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    logger.info(f"Building TensorRT engine from {onnx_path} (fp16={fp16}) ...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
    logger.info(f"TensorRT engine saved to {trt_path}")


def convert(args):
    from .cli import DET2_ONNX_EXPORT, DET2_PT_EXPORT, DET2_TRT_EXPORT

    output = args["output"]
    args["onnx_file_path"] = os.path.join(output, DET2_ONNX_EXPORT)
    args["trt_file_path"] = os.path.join(output, DET2_TRT_EXPORT)
    args["pt_file_path"] = os.path.join(output, DET2_PT_EXPORT)

    if args.get("pt", False):
        from .converter.detectron2_exporter import det2export

        args["format"] = "pt"
        det2export(args)

    if args.get("onnx", False):
        from .converter.detectron2_exporter import det2export
        from .converter.detectron2_onnx_trtonnx import onnx_gs

        args["format"] = "onnx"
        det2export(args)
        onnx_gs(args)

    if args.get("trt", False):
        convert_to_trt(args)
