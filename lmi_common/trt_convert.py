"""ONNX → TensorRT engine conversion via the TensorRT Python API.

Requires TensorRT >= 8.5 (matches `lmi_common/trt_engine.py`).
"""

import inspect
import logging
from typing import Dict, Tuple

from lmi_common.model_metadata import engine_props_header

logger = logging.getLogger(__name__)

_trt_logger_singleton = None

_ULTRALYTICS_AUTHOR = "Ultralytics"


def _inspect_onnx(onnx_path: str) -> Tuple[Dict[str, str], bool]:
    """(metadata_props map, whether onnx2engine should build this ONNX) — one load, one decision.

    Dynamic-shape models are excluded: the builder below owns the batch profile, and onnx2engine cannot express a batch-only one.
    """
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    props = {prop.key: prop.value for prop in model.metadata_props}
    static_shape = all(
        dim.HasField("dim_value") and dim.dim_value > 0 for inp in model.graph.input for dim in inp.type.tensor_type.shape.dim
    )
    return props, static_shape and props.get("author") == _ULTRALYTICS_AUTHOR


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

    A static-shape ultralytics ONNX is built by ``ultralytics.utils.export.onnx2engine`` so its export metadata is embedded in the
    plan file; everything else is built here, and the source ONNX's whole metadata_props map is carried over to the engine's header
    in the same layout. Static-batch ONNX (no dynamic dims) ignores the batch kwargs. Dynamic-batch ONNX (axis 0 == -1) gets an
    optimization profile from the kwargs; other dynamic axes are not supported and will raise.

    Args:
        onnx_path: source .onnx path.
        engine_path: destination engine path.
        fp16: enable FP16 precision.
        workspace_gb: builder workspace memory pool size, in GB.
        min_batch: minimum batch in the optimization profile.
        opt_batch: batch size to optimize for; defaults to ``max_batch``.
        max_batch: maximum batch in the optimization profile.
    """
    # Read before the build: an unreadable ONNX must not cost a full engine build first.
    props, use_onnx2engine = _inspect_onnx(onnx_path)

    if use_onnx2engine:
        try:
            from ultralytics.utils.export import onnx2engine
        except ImportError:
            # Pre-8.4 ultralytics: no writer for the metadata, so fall through and build without it.
            logger.warning("ultralytics.utils.export.onnx2engine unavailable; building without embedded export metadata")
        else:
            logger.info(f"Ultralytics ONNX: building via onnx2engine to embed export metadata in {engine_path}")
            # Mid-8.4 replaced the half/int8 flags with a single `quantize` precision selector.
            params = inspect.signature(onnx2engine).parameters
            precision = {"quantize": 16 if fp16 else None} if "quantize" in params else {"half": fp16}
            onnx2engine(onnx_path, engine_path, workspace=workspace_gb, metadata=props, **precision)
            return

    import tensorrt as trt

    trt_logger = _get_trt_logger()
    trt.init_libnvinfer_plugins(trt_logger, "")

    builder = trt.Builder(trt_logger)
    # EXPLICIT_BATCH is required by OnnxParser on TRT 8.x; on TRT 10 it's deprecated but still accepted.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    # By path, not by buffer: a buffer has no directory, so external weights beside the model cannot resolve.
    if not parser.parse_from_file(str(onnx_path)):
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
        if props:
            f.write(engine_props_header(props))
        f.write(serialized_engine)
    logger.info(f"TensorRT engine saved to {engine_path}")
