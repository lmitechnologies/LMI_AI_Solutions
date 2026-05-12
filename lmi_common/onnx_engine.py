import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


_NP_TO_TORCH_DTYPE: Dict[str, torch.dtype] = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int8)": torch.int8,
    "tensor(uint8)": torch.uint8,
    "tensor(int16)": torch.int16,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(bool)": torch.bool,
}

_TORCH_TO_NP_DTYPE: Dict[torch.dtype, np.dtype] = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
}


def _is_static_dim(d) -> bool:
    return isinstance(d, int) and d > 0


class ONNXEngine:
    """ONNX Runtime wrapper with the same public surface as ``lmi_common.trt_engine.TRTEngine``.

    On CUDA, both inputs and outputs are bound to torch CUDA tensors via ORT's IOBinding so
    the entire pipeline stays on GPU — zero host/device copies. Output buffers are pre-allocated
    once at the model's max shape and reused, matching ``TRTEngine``'s contract.

    Dynamic shape support
    ---------------------
    Symbolic dimensions are detected from ``session.get_inputs()`` / ``session.get_outputs()``:
    - Symbolic batch dimension (dim 0) is supported. ``is_dynamic`` reports True.
    - Symbolic non-batch dimensions raise ``NotImplementedError``.
    For dynamic-batch outputs, the symbolic dim is assumed to track the input batch size
    (standard CV-model behavior); ``infer()`` returns ``out[:actual_batch]``.

    Output aliasing
    ---------------
    On CUDA, returned tensors are **views into pre-allocated internal buffers**. Clone them
    with ``.clone()`` if you need to hold them across multiple ``infer()`` calls — the next
    call overwrites the same memory. On CPU, outputs are fresh tensors.

    Thread safety
    -------------
    NOT thread-safe. The IO binding and CUDA buffers are shared mutable state.

    Args:
        onnx_path: Path to the ``.onnx`` model file.
        device: ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.
        providers: Optional ORT execution providers list. Defaults to
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` on CUDA, ``["CPUExecutionProvider"]`` on CPU.
        session_options: Optional ``onnxruntime.SessionOptions`` instance.
        dynamic_max_batch: Max batch size to advertise for dynamic-batch models. Defaults to 32.
    """

    def __init__(
        self,
        onnx_path: str,
        device: str = "cuda",
        providers: Optional[Sequence[str]] = None,
        session_options=None,
        dynamic_max_batch: int = 32,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                f"Failed to import onnxruntime ({e}). If the package is installed, this usually means a "
                "CUDA/cuDNN version mismatch with onnxruntime-gpu. Otherwise install with: "
                "pip install onnxruntime-gpu (or onnxruntime)"
            ) from e

        self.device = torch.device(device)
        self._is_cuda = self.device.type == "cuda"
        self._device_id = self.device.index if self.device.index is not None else 0

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._is_cuda else ["CPUExecutionProvider"]
        resolved_providers = []
        for p in providers:
            if p == "CUDAExecutionProvider":
                resolved_providers.append((p, {"device_id": self._device_id}))
            else:
                resolved_providers.append(p)

        self._session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=resolved_providers)
        active = self._session.get_providers()
        logger.info(f"ONNX session loaded with providers: {active}")
        if self._is_cuda and "CUDAExecutionProvider" not in active:
            logger.warning("CUDA requested but CUDAExecutionProvider unavailable; running on CPU.")
            self.device = torch.device("cpu")
            self._is_cuda = False

        # Inputs.
        input_names: List[str] = []
        is_dynamic = False
        first_shape: Optional[Tuple[int, ...]] = None
        first_dtype: Optional[torch.dtype] = None
        max_batch = 0
        for meta in self._session.get_inputs():
            input_names.append(meta.name)
            shape = meta.shape
            dtype = _NP_TO_TORCH_DTYPE.get(meta.type, torch.float32)
            batch_dim = shape[0]
            dyn = not _is_static_dim(batch_dim)
            if dyn:
                is_dynamic = True
            for d in shape[1:]:
                if not _is_static_dim(d):
                    raise NotImplementedError(
                        f"Dynamic non-batch dimensions in input '{meta.name}' (shape={shape}) are not "
                        f"supported. Only a dynamic batch dimension (dim 0) is supported."
                    )
            resolved_batch = dynamic_max_batch if dyn else batch_dim
            max_batch = max(max_batch, resolved_batch)
            if first_shape is None:
                first_shape = tuple(shape[1:])
                first_dtype = dtype
            logger.info(f"ONNX input  '{meta.name}': shape={shape}, dtype={dtype}, dynamic_batch={dyn}")

        # Outputs — pre-allocate at max shape; track which outputs have a dynamic batch dim.
        output_names: List[str] = []
        output_buffers: Dict[str, torch.Tensor] = {}
        output_dtypes: Dict[str, np.dtype] = {}
        output_dynamic_batch: Dict[str, bool] = {}
        for meta in self._session.get_outputs():
            output_names.append(meta.name)
            shape = meta.shape
            dtype = _NP_TO_TORCH_DTYPE.get(meta.type, torch.float32)
            dyn_batch = len(shape) > 0 and not _is_static_dim(shape[0])
            for d in shape[1:]:
                if not _is_static_dim(d):
                    raise NotImplementedError(
                        f"Dynamic non-batch dimensions in output '{meta.name}' (shape={shape}) are not "
                        f"supported. Only a dynamic batch dimension (dim 0) is supported."
                    )
            if not shape:
                alloc_shape: Tuple[int, ...] = ()
            else:
                alloc_shape = ((dynamic_max_batch if dyn_batch else shape[0]),) + tuple(shape[1:])
            buf = torch.empty(alloc_shape, dtype=dtype, device=self.device) if self._is_cuda else torch.empty(alloc_shape, dtype=dtype)
            output_buffers[meta.name] = buf
            output_dtypes[meta.name] = _TORCH_TO_NP_DTYPE[dtype]
            output_dynamic_batch[meta.name] = dyn_batch
            logger.info(f"ONNX output '{meta.name}': alloc_shape={alloc_shape}, dtype={dtype}, dynamic_batch={dyn_batch}")

        self._input_names = input_names
        self._output_names = output_names
        self._input_meta = {m.name: m for m in self._session.get_inputs()}
        self._output_buffers = output_buffers
        self._output_dtypes = output_dtypes
        self._output_dynamic_batch = output_dynamic_batch

        # Public attributes — mirror TRTEngine.
        self.max_batch: int = max_batch
        self.input_dtype: torch.dtype = first_dtype or torch.float32
        self.input_shape: Tuple[int, ...] = first_shape or ()
        self.fp16: bool = self.input_dtype == torch.float16
        self.is_dynamic: bool = is_dynamic

        # Reusable IOBinding (CUDA only). Output buffer pointers are stable, so bind once.
        if self._is_cuda:
            self._io_binding = self._session.io_binding()
            for name in self._output_names:
                buf = self._output_buffers[name]
                self._io_binding.bind_output(
                    name=name,
                    device_type="cuda",
                    device_id=self._device_id,
                    element_type=self._output_dtypes[name],
                    shape=tuple(buf.shape),
                    buffer_ptr=buf.data_ptr(),
                )
        else:
            self._io_binding = None

    def infer(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """Run synchronous inference.

        Args:
            *inputs: One contiguous tensor per engine input, in ``self._input_names`` order.
                On CUDA, inputs must be CUDA tensors; on CPU, CPU tensors. Each input's
                batch size must not exceed ``self.max_batch``.

        Returns:
            List of output tensors in engine output order. For dynamic-batch outputs the
            tensors are sliced to ``[:actual_batch]``. On CUDA they are views into internal
            buffers — clone if held across calls.
        """
        if len(inputs) != len(self._input_names):
            raise ValueError(f"Expected {len(self._input_names)} input(s), got {len(inputs)}")

        actual_batch = inputs[0].shape[0]
        if any(x.shape[0] != actual_batch for x in inputs[1:]):
            batch_sizes = {n: x.shape[0] for n, x in zip(self._input_names, inputs)}
            raise ValueError(f"All inputs must have the same batch size, got: {batch_sizes}")
        if actual_batch > self.max_batch:
            raise ValueError(f"Batch size {actual_batch} exceeds engine max_batch {self.max_batch}")
        if not self.is_dynamic and actual_batch != self.max_batch:
            raise ValueError(f"Static engine requires batch size {self.max_batch}, got {actual_batch}")

        for name, x in zip(self._input_names, inputs):
            expected_dtype = _NP_TO_TORCH_DTYPE.get(self._input_meta[name].type, torch.float32)
            if x.dtype != expected_dtype:
                raise ValueError(f"Input '{name}': expected dtype {expected_dtype}, got {x.dtype}")

        if self._is_cuda:
            return self._infer_cuda(inputs, actual_batch)
        return self._infer_cpu(inputs)

    def _infer_cuda(self, inputs: Sequence[torch.Tensor], actual_batch: int) -> List[torch.Tensor]:
        binding = self._io_binding
        binding.clear_binding_inputs()

        for name, x in zip(self._input_names, inputs):
            if not x.is_cuda:
                raise ValueError(f"Input '{name}' must be a CUDA tensor on a CUDA engine, got device {x.device}")
            x = x.contiguous()
            binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=self._device_id,
                element_type=_TORCH_TO_NP_DTYPE[x.dtype],
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )

        # Output bindings persist across calls (buffer pointers are stable). For dynamic-batch
        # outputs, ORT writes to the first actual_batch rows of the pre-allocated buffer.
        self._session.run_with_iobinding(binding)

        outputs: List[torch.Tensor] = []
        for name in self._output_names:
            buf = self._output_buffers[name]
            if self._output_dynamic_batch[name]:
                outputs.append(buf[:actual_batch])
            else:
                outputs.append(buf)
        return outputs

    def _infer_cpu(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        feeds = {name: x.contiguous().cpu().numpy() for name, x in zip(self._input_names, inputs)}
        results = self._session.run(self._output_names, feeds)
        return [torch.from_numpy(r) for r in results]
