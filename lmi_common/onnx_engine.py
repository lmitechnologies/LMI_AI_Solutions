import logging
from dataclasses import dataclass
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

_TORCH_TO_NP_DTYPE: Dict[torch.dtype, type] = {
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


def _torch_dtype_from_ort(ort_type: str, where: str) -> torch.dtype:
    try:
        return _NP_TO_TORCH_DTYPE[ort_type]
    except KeyError:
        raise NotImplementedError(
            f"Unsupported ONNX tensor type '{ort_type}' for {where}. Supported types: {sorted(_NP_TO_TORCH_DTYPE)}"
        ) from None


@dataclass
class _OutputBinding:
    buffer: torch.Tensor
    np_dtype: type
    dynamic_batch: bool


class ONNXEngine:
    """ONNX Runtime wrapper mirroring ``lmi_common.trt_engine.TRTEngine``'s public surface.

    On CUDA, I/O is bound to torch CUDA tensors via ORT's IOBinding for a zero-copy GPU
    pipeline. Output buffers are pre-allocated once at max shape. Only a symbolic batch
    dim (dim 0) is supported; symbolic non-batch dims raise ``NotImplementedError``.

    Not thread-safe — create one instance per thread.

    Args:
        onnx_path: Path to the ``.onnx`` model file.
        device: ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.
        dynamic_max_batch: Max batch size for dynamic-batch models. Defaults to 32.
    """

    def __init__(
        self,
        onnx_path: str,
        device: str = "cuda",
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
        if self._is_cuda:
            # Resolve unindexed "cuda" against the current CUDA device so torch allocations
            # and ORT bindings agree on which GPU we're on.
            if self.device.index is None:
                self._device_id = torch.cuda.current_device()
                self.device = torch.device(f"cuda:{self._device_id}")
            else:
                self._device_id = self.device.index
        else:
            self._device_id = 0

        if self._is_cuda:
            providers = [("CUDAExecutionProvider", {"device_id": self._device_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(onnx_path, providers=providers)
        active = self._session.get_providers()
        logger.info(f"ONNX session loaded with providers: {active}")
        if self._is_cuda and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                f"device='cuda' was requested but CUDAExecutionProvider is unavailable "
                f"(active providers: {active}). Install onnxruntime-gpu matching your CUDA/cuDNN, "
                f"or construct with device='cpu'."
            )

        # Inputs.
        input_names: List[str] = []
        input_dtypes: Dict[str, torch.dtype] = {}
        input_spatial: Dict[str, Tuple[int, ...]] = {}
        is_dynamic = False
        first_shape: Optional[Tuple[int, ...]] = None
        first_dtype: Optional[torch.dtype] = None
        max_batch = 0
        ort_inputs = self._session.get_inputs()
        if not ort_inputs:
            raise ValueError(f"ONNX model '{onnx_path}' has no inputs.")
        for meta in ort_inputs:
            input_names.append(meta.name)
            shape = meta.shape
            if not shape:
                raise NotImplementedError(f"Scalar input '{meta.name}' (rank 0) is not supported; at least a batch dimension is required.")
            dtype = _torch_dtype_from_ort(meta.type, f"input '{meta.name}'")
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
            input_dtypes[meta.name] = dtype
            input_spatial[meta.name] = tuple(shape[1:])
            if first_shape is None:
                first_shape = tuple(shape[1:])
                first_dtype = dtype
                max_batch = resolved_batch
            elif resolved_batch != max_batch:
                raise ValueError(
                    f"Input '{meta.name}' batch dim ({resolved_batch}) differs from first input ({max_batch}). "
                    f"All inputs must share the same batch dimension."
                )
            logger.info(f"ONNX input  '{meta.name}': shape={shape}, dtype={dtype}, dynamic_batch={dyn}")

        # Outputs — pre-allocate at max shape; track which outputs have a dynamic batch dim.
        output_names: List[str] = []
        outputs: Dict[str, _OutputBinding] = {}
        for meta in self._session.get_outputs():
            output_names.append(meta.name)
            shape = meta.shape
            dtype = _torch_dtype_from_ort(meta.type, f"output '{meta.name}'")
            dyn_batch = len(shape) > 0 and not _is_static_dim(shape[0])
            if is_dynamic and not dyn_batch:
                raise NotImplementedError(
                    f"Output '{meta.name}' has static dim 0 ({shape[0] if shape else 'scalar'}) but the model "
                    f"has a dynamic input batch. Outputs whose dim 0 does not scale with batch are not supported."
                )
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
            buf = torch.empty(alloc_shape, dtype=dtype, device=self.device)
            outputs[meta.name] = _OutputBinding(buffer=buf, np_dtype=_TORCH_TO_NP_DTYPE[dtype], dynamic_batch=dyn_batch)
            logger.info(f"ONNX output '{meta.name}': alloc_shape={alloc_shape}, dtype={dtype}, dynamic_batch={dyn_batch}")

        self._input_names = input_names
        self._output_names = output_names
        self._input_dtypes = input_dtypes
        self._input_spatial = input_spatial
        self._outputs = outputs

        # Public attributes
        assert first_dtype is not None and first_shape is not None
        self.max_batch: int = max_batch
        self.input_dtype: torch.dtype = first_dtype
        self.input_shape: Tuple[int, ...] = first_shape
        self.fp16: bool = self.input_dtype == torch.float16
        self.is_dynamic: bool = is_dynamic

        # Reusable IOBinding (CUDA only). Static outputs are bound once; dynamic-batch outputs
        # must be rebound each call so ORT's reported shape matches the actual computed shape.
        if self._is_cuda:
            self._io_binding = self._session.io_binding()
            for name in self._output_names:
                b = self._outputs[name]
                if b.dynamic_batch:
                    continue
                self._io_binding.bind_output(
                    name=name,
                    device_type="cuda",
                    device_id=self._device_id,
                    element_type=b.np_dtype,
                    shape=tuple(b.buffer.shape),
                    buffer_ptr=b.buffer.data_ptr(),
                )
        else:
            self._io_binding = None

    @property
    def _output_buffers(self) -> List[torch.Tensor]:
        """List of output buffers in engine output order. Mirrors ``TRTEngine._output_buffers``."""
        return [self._outputs[n].buffer for n in self._output_names]

    def infer(self, *inputs: torch.Tensor, copy: bool = True) -> List[torch.Tensor]:
        """Run synchronous inference.

        Args:
            *inputs: One tensor per engine input, in ``self._input_names`` order. Must
                match the engine device (CUDA or CPU) with ``shape[0] <= self.max_batch``.
                Non-contiguous inputs incur a ``.contiguous()`` copy.
            copy: If True (default), return independent clones — safe to hold across
                calls. If False, on CUDA return views into internal buffers that must be
                consumed before the next ``infer()``. Ignored on CPU (always fresh).

        Returns:
            Output tensors in engine output order. Dynamic-batch outputs slice to ``[:actual_batch]``.
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
            expected_dtype = self._input_dtypes[name]
            if x.dtype != expected_dtype:
                raise ValueError(f"Input '{name}': expected dtype {expected_dtype}, got {x.dtype}")
            expected_spatial = self._input_spatial[name]
            if tuple(x.shape[1:]) != expected_spatial:
                raise ValueError(f"Input '{name}': expected spatial shape {expected_spatial}, got {tuple(x.shape[1:])}")

        if self._is_cuda:
            return self._infer_cuda(inputs, actual_batch, copy=copy)
        return self._infer_cpu(inputs)

    def _infer_cuda(self, inputs: Sequence[torch.Tensor], actual_batch: int, *, copy: bool) -> List[torch.Tensor]:
        binding = self._io_binding
        binding.clear_binding_inputs()

        # Keep strong refs to any freshly allocated contiguous copies until after run_with_iobinding
        contiguous_tensors: List[torch.Tensor] = []
        for name, x in zip(self._input_names, inputs):
            if not x.is_cuda:
                raise ValueError(f"Input '{name}' must be a CUDA tensor on a CUDA engine, got device {x.device}")
            xc = x.contiguous()
            contiguous_tensors.append(xc)
            binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=self._device_id,
                element_type=_TORCH_TO_NP_DTYPE[xc.dtype],
                shape=tuple(xc.shape),
                buffer_ptr=xc.data_ptr(),
            )

        # Rebind dynamic-batch outputs each call with the actual batch (buffer_ptr is stable,
        # only the shape field changes). Static outputs were bound once in __init__.
        for name in self._output_names:
            b = self._outputs[name]
            if not b.dynamic_batch:
                continue
            out_shape = (actual_batch,) + tuple(b.buffer.shape[1:])
            binding.bind_output(
                name=name,
                device_type="cuda",
                device_id=self._device_id,
                element_type=b.np_dtype,
                shape=out_shape,
                buffer_ptr=b.buffer.data_ptr(),
            )

        self._session.run_with_iobinding(binding)
        del contiguous_tensors

        outputs: List[torch.Tensor] = []
        for name in self._output_names:
            b = self._outputs[name]
            buf = b.buffer
            if b.dynamic_batch:
                outputs.append(buf[:actual_batch])
            else:
                outputs.append(buf)
        if copy:
            outputs = [o.clone() for o in outputs]
        return outputs

    def _infer_cpu(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        for name, x in zip(self._input_names, inputs):
            if x.device.type != "cpu":
                raise ValueError(f"Input '{name}' must be a CPU tensor on a CPU engine, got device {x.device}")
        feeds = {name: x.contiguous().numpy() for name, x in zip(self._input_names, inputs)}
        results = self._session.run(self._output_names, feeds)
        return [torch.from_numpy(r) for r in results]
