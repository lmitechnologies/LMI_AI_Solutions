"""ONNX → TensorRT engine conversion via the TensorRT Python API.

Requires TensorRT >= 8.5 (matches `lmi_common/trt_engine.py`).
"""

import logging

logger = logging.getLogger(__name__)

_trt_logger_singleton = None


def _get_trt_logger():
    """Return a process-wide ``trt.ILogger`` that forwards to Python ``logging``.

    All TRT C++ messages appear under the ``tensorrt`` Python logger, interleaved with regular Python logs in a single stream.
    """
    global _trt_logger_singleton
    if _trt_logger_singleton is not None:
        return _trt_logger_singleton

    import tensorrt as trt

    class _PyTRTLogger(trt.ILogger):
        _SEVERITY_TO_LEVEL = {
            trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
            trt.Logger.ERROR: logging.ERROR,
            trt.Logger.WARNING: logging.WARNING,
            trt.Logger.INFO: logging.INFO,
            trt.Logger.VERBOSE: logging.DEBUG,
        }

        def __init__(self):
            trt.ILogger.__init__(self)
            self._log = logging.getLogger("tensorrt")

        def log(self, severity, msg):
            self._log.log(self._SEVERITY_TO_LEVEL.get(severity, logging.INFO), msg)

    _trt_logger_singleton = _PyTRTLogger()
    return _trt_logger_singleton


def onnx_to_trt(
    onnx_path: str,
    engine_path: str,
    *,
    fp16: bool = True,
    workspace_gb: int = 4,
    min_batch: int = 1,
    opt_batch: int = None,
    max_batch: int = 1,
) -> None:
    """Build a serialized TensorRT engine from an ONNX file.

    Static-batch ONNX (no dynamic dims) ignores the batch kwargs.
    Dynamic-batch ONNX (axis 0 == -1) gets an optimization profile from the kwargs;
    other dynamic axes are not supported and will raise.

    Args:
        onnx_path: source .onnx path.
        engine_path: destination engine path.
        fp16: enable FP16 precision.
        workspace_gb: builder workspace memory pool size, in GB.
        min_batch: minimum batch in the optimization profile.
        opt_batch: batch size to optimize for; defaults to ``max_batch``.
        max_batch: maximum batch in the optimization profile.
    """
    import tensorrt as trt

    trt_logger = _get_trt_logger()
    trt.init_libnvinfer_plugins(trt_logger, "")

    builder = trt.Builder(trt_logger)
    # EXPLICIT_BATCH is required by OnnxParser on TRT 8.x; on TRT 10 it's deprecated but still accepted.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if fp16:
        if not builder.platform_has_fast_fp16:
            logger.warning("FP16 requested but not natively supported on this platform")
        config.set_flag(trt.BuilderFlag.FP16)

    dynamic_inputs = []
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = tuple(inp.shape)
        if any(d == -1 for d in shape[1:]):
            raise ValueError(f"Input '{inp.name}' has dynamic non-batch dims {shape}; only dynamic batch (axis 0) is supported")
        if shape[0] == -1:
            dynamic_inputs.append((inp.name, shape))

    if dynamic_inputs:
        opt = opt_batch if opt_batch is not None else max_batch
        if not (min_batch <= opt <= max_batch):
            raise ValueError(f"Require min_batch <= opt_batch <= max_batch, got ({min_batch}, {opt}, {max_batch})")
        profile = builder.create_optimization_profile()
        for name, shape in dynamic_inputs:
            tail = shape[1:]
            profile.set_shape(name, (min_batch, *tail), (opt, *tail), (max_batch, *tail))
        config.add_optimization_profile(profile)
        logger.info(f"Dynamic-batch profile: min={min_batch}, opt={opt}, max={max_batch} for inputs {[n for n, _ in dynamic_inputs]}")

    logger.info(f"Building TensorRT engine from {onnx_path} (fp16={fp16}, workspace={workspace_gb} GB) ...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    logger.info(f"TensorRT engine saved to {engine_path}")
