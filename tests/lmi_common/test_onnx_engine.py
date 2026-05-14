import numpy as np
import pytest
import torch
import torch.nn as nn

ort = pytest.importorskip("onnxruntime")

from lmi_common.onnx_engine import ONNXEngine  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


@pytest.fixture(scope="module")
def dynamic_onnx_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "dyn_model.onnx"
    model = _Tiny().eval()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return str(path)


def _reference_output(onnx_path: str, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


@pytest.mark.parametrize("actual_batch", [1, 3, 8])
def test_dynamic_batch_output_matches_reference_cpu(dynamic_onnx_path, actual_batch):
    engine = ONNXEngine(dynamic_onnx_path, device="cpu", dynamic_max_batch=8)
    assert engine.is_dynamic
    assert engine.max_batch == 8

    x = torch.randn(actual_batch, 3, 32, 32, dtype=torch.float32)
    out = engine.infer(x)[0].cpu().numpy()

    ref = _reference_output(dynamic_onnx_path, x.numpy())
    assert out.shape == ref.shape == (actual_batch, 4)
    np.testing.assert_allclose(out, ref, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("actual_batch", [1, 3, 8])
def test_dynamic_batch_output_matches_reference_cuda(dynamic_onnx_path, actual_batch):
    """The key check for bug #2: with actual_batch < dynamic_max_batch and a one-time
    output binding sized at max, does ORT still write correct values into our buffer?"""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda", dynamic_max_batch=8)
    assert engine.is_dynamic

    x_cpu = torch.randn(actual_batch, 3, 32, 32, dtype=torch.float32)
    x_cuda = x_cpu.cuda()
    out = engine.infer(x_cuda)[0].clone().cpu().numpy()

    ref = _reference_output(dynamic_onnx_path, x_cpu.numpy())
    assert out.shape == ref.shape == (actual_batch, 4)
    np.testing.assert_allclose(out, ref, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_copy_true_returns_independent_tensors_cuda(dynamic_onnx_path):
    """copy=True (default): two sequential infer() calls must produce tensors that do not
    alias each other. The first result must stay stable after the second call overwrites
    the engine's internal buffer."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda", dynamic_max_batch=8)
    x1 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")
    x2 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")

    out1 = engine.infer(x1)[0]
    snapshot = out1.clone()
    out2 = engine.infer(x2)[0]

    assert out1.data_ptr() != out2.data_ptr()
    torch.testing.assert_close(out1, snapshot)
    assert not torch.allclose(out1, out2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_copy_false_returns_aliased_views_cuda(dynamic_onnx_path):
    """copy=False: two sequential infer() calls return views into the same internal buffer.
    Pins the opt-out contract."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda", dynamic_max_batch=8)
    x1 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")
    x2 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")

    out1 = engine.infer(x1, copy=False)[0]
    snapshot = out1.clone()
    out2 = engine.infer(x2, copy=False)[0]

    assert out1.data_ptr() == out2.data_ptr()
    torch.testing.assert_close(out1, out2)
    assert not torch.allclose(out1, snapshot)


@pytest.fixture(scope="module")
def static_output_dim0_onnx_path(tmp_path_factory):
    """A model with a symbolic input batch but a static output dim 0. Gather with a fixed
    indices initializer of length 1 along axis 0 produces output shape (1, 4)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    path = tmp_path_factory.mktemp("onnx_static_out") / "gather.onnx"
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    indices = helper.make_tensor("indices", TensorProto.INT64, [1], [0])
    node = helper.make_node("Gather", inputs=["input", "indices"], outputs=["output"], axis=0)
    graph = helper.make_graph([node], "static_out", [inp], [out], initializer=[indices])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


def test_static_output_dim0_with_dynamic_input_rejected(static_output_dim0_onnx_path):
    """Dynamic input batch + static output dim 0 is unsupported (would silently mis-slice)."""
    with pytest.raises(NotImplementedError, match="static dim 0"):
        ONNXEngine(static_output_dim0_onnx_path, device="cpu", dynamic_max_batch=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unindexed_cuda_device_is_resolved(dynamic_onnx_path):
    """device='cuda' (no index) must resolve to a concrete cuda:N so torch allocations
    and ORT bindings target the same GPU."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda", dynamic_max_batch=4)
    assert engine.device.index is not None
    assert engine.device.index == engine._device_id
    # Sanity: output buffer ended up on the resolved device.
    assert engine._output_buffers[0].device.index == engine._device_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input_cuda(dynamic_onnx_path):
    """Regression test for bug #1: non-contiguous input must not produce a dangling pointer."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda", dynamic_max_batch=8)

    base = torch.randn(4, 3, 32, 64, dtype=torch.float32, device="cuda")
    x = base[:, :, :, :32]  # non-contiguous view
    assert not x.is_contiguous()

    out = engine.infer(x)[0].clone().cpu().numpy()
    ref = _reference_output(dynamic_onnx_path, x.contiguous().cpu().numpy())
    np.testing.assert_allclose(out, ref, atol=1e-4)
