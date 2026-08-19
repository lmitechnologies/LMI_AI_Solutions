import logging
from typing import Any, Dict, List, Tuple

import torch

from lmi_common.model_metadata import split_engine_metadata

logger = logging.getLogger(__name__)


class TRTEngine:
    """TensorRT engine wrapper for static and dynamic-batch engines (TensorRT 8.5+).

    I/O buffers are allocated as torch CUDA tensors at init (at max-batch shape for
    dynamic engines). Only a dynamic batch dim (dim 0) is supported; dynamic spatial
    dims raise ``NotImplementedError``.

    A metadata header written by ``lmi_common.trt_convert`` is stripped before deserializing and
    exposed as ``self.metadata`` ({} for an engine without one).

    Not thread-safe — create one instance per thread.

    Args:
        engine_path: Path to the serialized ``.engine`` file.
        device: CUDA device string, e.g. ``"cuda"`` or ``"cuda:0"``.
        log_level: TensorRT logger severity. Defaults to ``trt.Logger.WARNING``.
    """

    def __init__(self, engine_path: str, device: str = "cuda", log_level=None) -> None:
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError("tensorrt is required. Install with: pip install tensorrt") from e

        dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }

        trt_logger = trt.Logger(log_level if log_level is not None else trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, namespace="")

        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            metadata, plan = split_engine_metadata(f.read())
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        if metadata:
            logger.info(f"Engine metadata: {sorted(metadata)}")

        self._engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context for: {engine_path}")
        self.device = torch.device(device)

        def resolve_dtype(name: str) -> torch.dtype:
            trt_dtype = engine.get_tensor_dtype(name)
            if trt_dtype not in dtype_map:
                raise NotImplementedError(f"Unsupported TensorRT dtype {trt_dtype} for tensor '{name}'")
            return dtype_map[trt_dtype]

        input_names: List[str] = []
        output_names: List[str] = []
        tensor_order: List[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            tensor_order.append(name)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        if not input_names:
            raise ValueError(f"TensorRT engine '{engine_path}' has no inputs.")
        for n in input_names:
            if len(engine.get_tensor_shape(n)) == 0:
                raise NotImplementedError(f"Scalar input '{n}' (rank 0) is not supported; at least a batch dimension is required.")

        is_dynamic = any(engine.get_tensor_shape(n)[0] == -1 for n in input_names)

        # Allocate input buffers at max shape; record max_batch from first input.
        input_buffers: Dict[str, torch.Tensor] = {}
        max_batch = 0
        for i, name in enumerate(input_names):
            shape = tuple(engine.get_tensor_shape(name))
            if any(d == -1 for d in shape[1:]):
                raise NotImplementedError(
                    f"Dynamic spatial dimensions in input tensor '{name}' (shape={shape}) are not "
                    f"supported. Only a dynamic batch dimension (dim 0) is supported."
                )
            alloc_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2]) if is_dynamic else shape
            input_buffers[name] = torch.empty(alloc_shape, dtype=resolve_dtype(name), device=self.device)
            logger.info(f"TRT input  '{name}': shape={alloc_shape}, dtype={input_buffers[name].dtype}")
            if i == 0:
                max_batch = alloc_shape[0]
            elif alloc_shape[0] != max_batch:
                raise ValueError(
                    f"Input '{name}' batch dim ({alloc_shape[0]}) differs from first input ({max_batch}). "
                    f"All inputs must share the same batch dimension."
                )

        # Output shapes depend on input shapes for dynamic engines — set context to max first.
        if is_dynamic:
            for name, buf in input_buffers.items():
                self.context.set_input_shape(name, tuple(buf.shape))

        output_buffers: List[torch.Tensor] = []
        for name in output_names:
            static_shape = tuple(engine.get_tensor_shape(name))
            if is_dynamic and (len(static_shape) == 0 or static_shape[0] != -1):
                raise NotImplementedError(
                    f"Output '{name}' has static dim 0 ({static_shape[0] if static_shape else 'scalar'}) but the engine "
                    f"has a dynamic input batch. Outputs whose dim 0 does not scale with batch are not supported."
                )
            alloc_shape = tuple(self.context.get_tensor_shape(name)) if is_dynamic else static_shape
            output_buffers.append(torch.empty(alloc_shape, dtype=resolve_dtype(name), device=self.device))
            logger.info(f"TRT output '{name}': shape={alloc_shape}, dtype={output_buffers[-1].dtype}")

        self._input_names = input_names
        self._output_names = output_names
        self._input_buffers = input_buffers
        self._output_buffers = output_buffers
        all_bufs: Dict[str, torch.Tensor] = {**input_buffers, **dict(zip(output_names, output_buffers))}
        self._bindings: List[int] = [all_bufs[name].data_ptr() for name in tensor_order]

        # Public attributes consumed by model __init__ and preprocess.
        first_input_buf = input_buffers[input_names[0]]
        self.max_batch: int = max_batch
        self.input_dtype: torch.dtype = first_input_buf.dtype
        self.input_shape: Tuple[int, ...] = tuple(first_input_buf.shape[1:])  # (C, H, W)
        self.fp16: bool = first_input_buf.dtype == torch.float16
        self.is_dynamic: bool = is_dynamic
        self.metadata: Dict[str, Any] = metadata

    def release(self) -> None:
        """Release TensorRT resources deterministically (context before engine) and drop CUDA I/O buffers.

        Safe to call more than once. The engine must not be used afterwards.
        """
        # Drop in dependency order: TRT requires the execution context to be destroyed before the engine.
        self.context = None
        self._engine = None
        self._input_buffers = {}
        self._output_buffers = []
        self._bindings = []

    def infer(self, *inputs: torch.Tensor, copy: bool = True) -> List[torch.Tensor]:
        """Run synchronous inference.

        Args:
            *inputs: One CUDA tensor per engine input, in ``self._input_names`` order.
                Each must live on ``self.device`` with ``shape[0] <= self.max_batch``.
            copy: If True (default), return independent clones — safe to hold across
                calls. If False, return views into internal buffers; the caller must
                consume them before the next ``infer()`` overwrites the memory.

        Returns:
            Output tensors in engine output order. Dynamic engines slice to ``[:actual_batch]``.
        """
        if torch.cuda.current_stream(self.device) != torch.cuda.default_stream(self.device):
            raise RuntimeError(
                "TRTEngine.infer() must run on the default CUDA stream; do not call inside a torch.cuda.stream(...) context."
            )

        if len(inputs) != len(self._input_names):
            raise ValueError(f"Expected {len(self._input_names)} input(s), got {len(inputs)}")

        actual_batch = inputs[0].shape[0]
        if actual_batch > self.max_batch:
            raise ValueError(f"Batch size {actual_batch} exceeds engine max_batch {self.max_batch}")
        if not self.is_dynamic and actual_batch != self.max_batch:
            raise ValueError(f"Static engine requires batch size {self.max_batch}, got {actual_batch}")

        for name, x in zip(self._input_names, inputs):
            buf = self._input_buffers[name]
            expected = (buf.device, buf.dtype, (actual_batch, *buf.shape[1:]))
            got = (x.device, x.dtype, tuple(x.shape))
            if got != expected:
                raise ValueError(f"Input '{name}': expected (device, dtype, shape)={expected}, got={got}")
            if self.is_dynamic:
                self.context.set_input_shape(name, tuple(x.shape))
            buf[:actual_batch].copy_(x)

        with torch.cuda.device(self.device):
            if not self.context.execute_v2(self._bindings):
                raise RuntimeError("TensorRT execute_v2 failed")

        outs = [out[:actual_batch] for out in self._output_buffers] if self.is_dynamic else list(self._output_buffers)
        return [o.clone() for o in outs] if copy else outs
