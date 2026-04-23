import json
import logging
import os

logger = logging.getLogger(__name__)


def _write_class_names(model_path: str, class_names: list) -> None:
    """Write class names to a sidecar JSON file next to model_path."""
    sidecar = os.path.splitext(model_path)[0] + ".classes.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(list(class_names), f)
    logger.info(f"Class names written to: {sidecar}")


def build_trt_engine(onnx_dir: str, **kwargs) -> None:
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("tensorrt is required for TensorRT conversion. Install it with: pip install tensorrt") from e

    engine_dir = onnx_dir.replace(".onnx", ".engine")
    workspace_mb = 4096
    verbose = kwargs.get("verbose", False)

    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_dir, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx_dir}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    logger.info(f"Building TensorRT engine, saving to {engine_dir} ...")
    engine_bytes = builder.build_serialized_network(network, config)
    with open(engine_dir, "wb") as f:
        f.write(engine_bytes)
    logger.info("Done.")


def convert_to_tensorrt(onnx_path: str, **kwargs) -> None:
    """
    Convert an ONNX model to TensorRT engine.

    Args:
        onnx_path (str): Path to the ONNX model file.
        **kwargs: Additional keyword arguments for conversion options.
    """
    build_trt_engine(onnx_path, **kwargs)


def convert_to_onnx(model, output_dir: str, **kwargs) -> None:
    model.export(output_dir=output_dir, opset_version=kwargs.get("opset_version", 17))
