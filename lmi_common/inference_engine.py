from typing import List, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class InferenceEngine(Protocol):
    """Common contract for runtime inference backends (TensorRT, ONNX Runtime, ...).

    Implementations accept torch tensors as inputs and return torch tensors as outputs,
    so domain code can swap backends without changing pre/postprocess.
    """

    _input_names: List[str]
    _output_names: List[str]
    max_batch: int
    input_dtype: torch.dtype
    input_shape: Tuple[int, ...]
    fp16: bool
    is_dynamic: bool

    def infer(self, *inputs: torch.Tensor) -> List[torch.Tensor]: ...
