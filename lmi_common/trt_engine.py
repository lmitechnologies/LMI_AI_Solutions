import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


class TRTEngine:
    """TensorRT engine wrapper for static and dynamic-batch engines (TensorRT 8.5+).

    I/O buffers are allocated as torch CUDA tensors at init (at max-batch shape for
    dynamic engines). Only a dynamic batch dim (dim 0) is supported; dynamic spatial
    dims raise ``NotImplementedError``.

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

        _dtype_map = {
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
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self._engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context for: {engine_path}")
        self.device = torch.device(device)

        # Enumerate I/O tensors in engine order — order matters for execute_v2.
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

        # Detect dynamic batch and validate no dynamic spatial dims.
        is_dynamic = False
        max_batch: int = 0
        for i, name in enumerate(input_names):
            shape = tuple(engine.get_tensor_shape(name))
            if shape[0] == -1:
                is_dynamic = True
            if any(d == -1 for d in shape[1:]):
                raise NotImplementedError(
                    f"Dynamic spatial dimensions in input tensor '{name}' (shape={shape}) are not "
                    f"supported. Only a dynamic batch dimension (dim 0) is supported."
                )
            if is_dynamic:
                resolved_batch = int(engine.get_tensor_profile_shape(name, 0)[2][0])
            else:
                resolved_batch = shape[0]
            if i == 0:
                max_batch = resolved_batch
            elif resolved_batch != max_batch:
                raise ValueError(
                    f"Input '{name}' batch dim ({resolved_batch}) differs from first input ({max_batch}). "
                    f"All inputs must share the same batch dimension."
                )

        def _resolve_dtype(name: str) -> torch.dtype:
            trt_dtype = engine.get_tensor_dtype(name)
            if trt_dtype not in _dtype_map:
                raise NotImplementedError(f"Unsupported TensorRT dtype {trt_dtype} for tensor '{name}'")
            return _dtype_map[trt_dtype]

        # Allocate input buffers.
        input_buffers: Dict[str, torch.Tensor] = {}
        for name in input_names:
            shape = tuple(engine.get_tensor_shape(name))
            dtype = _resolve_dtype(name)
            if is_dynamic:
                alloc_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2])  # max shape
            else:
                alloc_shape = shape
            input_buffers[name] = torch.empty(alloc_shape, dtype=dtype, device=self.device)
            logger.info(f"TRT input  '{name}': shape={alloc_shape}, dtype={dtype}")

        # For dynamic engines, output shapes depend on input shapes. Set the context to max
        # input shapes so we can query each output's max shape via the context.
        if is_dynamic:
            for name, buf in input_buffers.items():
                self.context.set_input_shape(name, tuple(buf.shape))

        # Allocate output buffers.
        output_buffers: List[torch.Tensor] = []
        for name in output_names:
            dtype = _resolve_dtype(name)
            out_shape = tuple(engine.get_tensor_shape(name))
            if is_dynamic and (len(out_shape) == 0 or out_shape[0] != -1):
                raise NotImplementedError(
                    f"Output '{name}' has static dim 0 ({out_shape[0] if out_shape else 'scalar'}) but the engine "
                    f"has a dynamic input batch. Outputs whose dim 0 does not scale with batch are not supported."
                )
            if is_dynamic:
                alloc_shape = tuple(self.context.get_tensor_shape(name))
            else:
                alloc_shape = out_shape
            output_buffers.append(torch.empty(alloc_shape, dtype=dtype, device=self.device))
            logger.info(f"TRT output '{name}': shape={alloc_shape}, dtype={dtype}")

        self._input_names = input_names
        self._output_names = output_names
        self._input_buffers = input_buffers
        self._output_buffers = output_buffers

        # Build binding address list once — pointers are stable for the object's lifetime.
        all_bufs: Dict[str, torch.Tensor] = {**input_buffers, **dict(zip(output_names, output_buffers))}
        self._bindings: List[int] = [all_bufs[name].data_ptr() for name in tensor_order]

        # Public attributes consumed by model __init__ and preprocess.
        first_input_buf = input_buffers[input_names[0]]
        self.max_batch: int = max_batch
        self.input_dtype: torch.dtype = first_input_buf.dtype
        self.input_shape: Tuple[int, ...] = tuple(first_input_buf.shape[1:])  # (C, H, W)
        self.fp16: bool = first_input_buf.dtype == torch.float16
        self.is_dynamic: bool = is_dynamic

    def infer(self, *inputs: torch.Tensor, copy: bool = True) -> List[torch.Tensor]:
        """Run synchronous inference.

        Args:
            *inputs: One CUDA tensor per engine input, in ``self._input_names`` order.
                Each must live on ``self.device`` with ``shape[0] <= self.max_batch``.
                Non-contiguous inputs incur a ``.contiguous()`` copy.
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

        for name, x in zip(self._input_names, inputs):
            buf = self._input_buffers[name]
            if x.device != buf.device:
                raise ValueError(f"Input '{name}': expected device {buf.device}, got {x.device}")
            if x.dtype != buf.dtype:
                raise ValueError(f"Input '{name}': expected dtype {buf.dtype}, got {x.dtype}")
            if x.shape[1:] != buf.shape[1:]:
                raise ValueError(f"Input '{name}': expected spatial shape {buf.shape[1:]}, got {x.shape[1:]}")

        actual_batch = inputs[0].shape[0]
        if any(x.shape[0] != actual_batch for x in inputs[1:]):
            batch_sizes = {name: x.shape[0] for name, x in zip(self._input_names, inputs)}
            raise ValueError(f"All inputs must have the same batch size, got: {batch_sizes}")
        if actual_batch > self.max_batch:
            raise ValueError(f"Batch size {actual_batch} exceeds engine max_batch {self.max_batch}")
        if not self.is_dynamic and actual_batch != self.max_batch:
            raise ValueError(f"Static engine requires batch size {self.max_batch}, got {actual_batch}")

        for name, x in zip(self._input_names, inputs):
            if self.is_dynamic:
                self.context.set_input_shape(name, tuple(x.shape))
            self._input_buffers[name][:actual_batch].copy_(x.contiguous())

        with torch.cuda.device(self.device):
            if not self.context.execute_v2(self._bindings):
                raise RuntimeError("TensorRT execute_v2 failed")

        if self.is_dynamic:
            outs = [out[:actual_batch] for out in self._output_buffers]
        else:
            outs = list(self._output_buffers)
        return [o.clone() for o in outs] if copy else outs
