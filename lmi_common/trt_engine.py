import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


class TRTEngine:
    """TensorRT engine wrapper supporting static and dynamic-batch engines (TensorRT 8.5+).

    Allocates all I/O buffers as torch CUDA tensors at init time (at max shape for dynamic
    engines). Binding addresses are computed once and reused — tensor pointers are stable for
    the lifetime of this object.

    Dynamic shape support
    ---------------------
    If the input tensor's batch dimension is -1 (dynamic), the engine is treated as dynamic:
    - I/O buffers are allocated at the max-batch shape from optimization profile 0.
    - ``set_input_shape`` is called before every ``execute_v2``.
    - Outputs are sliced to ``[:actual_batch]`` before being returned.
    Only a dynamic batch dimension (dim 0) is supported. Dynamic spatial dimensions (H, W)
    raise ``NotImplementedError`` at init time.

    Output aliasing
    ---------------
    ``infer()`` returns views into internal buffers. Clone the outputs if you need to hold
    them across multiple ``infer()`` calls — the next call overwrites the same memory.

    Thread safety
    -------------
    NOT thread-safe. The execution context and I/O buffers are shared mutable state.
    Create one ``TRTEngine`` instance per thread for concurrent inference.

    Args:
        engine_path: Path to the serialized TensorRT ``.engine`` file.
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
        for name in input_names:
            shape = tuple(engine.get_tensor_shape(name))
            if shape[0] == -1:
                is_dynamic = True
            if any(d == -1 for d in shape[1:]):
                raise NotImplementedError(
                    f"Dynamic spatial dimensions in input tensor '{name}' (shape={shape}) are not "
                    f"supported. Only a dynamic batch dimension (dim 0) is supported."
                )

        # Allocate input buffers.
        input_buffers: Dict[str, torch.Tensor] = {}
        for name in input_names:
            shape = tuple(engine.get_tensor_shape(name))
            dtype = _dtype_map.get(engine.get_tensor_dtype(name), torch.float32)
            if is_dynamic:
                alloc_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2])  # max shape
            else:
                alloc_shape = shape
            input_buffers[name] = torch.empty(alloc_shape, dtype=dtype, device=self.device)
            logger.info(f"TRT input  '{name}': shape={alloc_shape}, dtype={dtype}")

        # Allocate output buffers.
        output_buffers: List[torch.Tensor] = []
        for name in output_names:
            shape = tuple(engine.get_tensor_shape(name))
            dtype = _dtype_map.get(engine.get_tensor_dtype(name), torch.float32)
            if is_dynamic:
                alloc_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2])  # max shape
            else:
                alloc_shape = shape
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
        self.max_batch: int = first_input_buf.shape[0]
        self.input_dtype: torch.dtype = first_input_buf.dtype
        self.input_shape: Tuple[int, ...] = tuple(first_input_buf.shape[1:])  # (C, H, W)
        self.fp16: bool = first_input_buf.dtype == torch.float16
        self.is_dynamic: bool = is_dynamic

    def infer(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """Run synchronous TensorRT inference.

        Args:
            *inputs: One contiguous CUDA tensor per engine input, in the same order as
                     ``self._input_names``. Each tensor's batch size must not exceed
                     ``self.max_batch``.

        Returns:
            List of output tensors in engine output order. For dynamic engines the tensors
            are sliced to ``[:actual_batch]``. For static engines they are the full
            pre-allocated buffers.

            **The returned tensors are views into internal buffers.** Clone them with
            ``.clone()`` if you need to hold them across multiple ``infer()`` calls.

        Raises:
            ValueError: If the number of inputs doesn't match, or batch size exceeds max.
            RuntimeError: If ``execute_v2`` reports failure.
        """
        if len(inputs) != len(self._input_names):
            raise ValueError(f"Expected {len(self._input_names)} input(s), got {len(inputs)}")

        for name, x in zip(self._input_names, inputs):
            buf = self._input_buffers[name]
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

        if not self.context.execute_v2(self._bindings):
            raise RuntimeError("TensorRT execute_v2 failed")

        if self.is_dynamic:
            return [out[:actual_batch] for out in self._output_buffers]
        return self._output_buffers
