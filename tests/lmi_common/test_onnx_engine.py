import subprocess
import sys

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
    engine = ONNXEngine(dynamic_onnx_path, device="cpu")
    assert engine.is_dynamic
    assert engine.max_batch is None

    x = torch.randn(actual_batch, 3, 32, 32, dtype=torch.float32)
    out = engine.infer(x)[0].cpu().numpy()

    ref = _reference_output(dynamic_onnx_path, x.numpy())
    assert out.shape == ref.shape == (actual_batch, 4)
    np.testing.assert_allclose(out, ref, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("actual_batch", [1, 3, 8])
def test_dynamic_batch_output_matches_reference_cuda(dynamic_onnx_path, actual_batch):
    """ORT must write correct values into our reused output buffer at every batch size."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda")
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

    engine = ONNXEngine(dynamic_onnx_path, device="cuda")
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

    engine = ONNXEngine(dynamic_onnx_path, device="cuda")
    x1 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")
    x2 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")

    out1 = engine.infer(x1, copy=False)[0]
    snapshot = out1.clone()
    out2 = engine.infer(x2, copy=False)[0]

    assert out1.data_ptr() == out2.data_ptr()
    torch.testing.assert_close(out1, out2)
    assert not torch.allclose(out1, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dynamic_batch_has_no_limit_cuda(dynamic_onnx_path):
    """Output buffers grow by doubling, or to the batch when that is larger; smaller batches reuse them."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda")
    for batch, capacity in [(2, 2), (3, 4), (40, 40), (3, 40)]:
        x_cpu = torch.randn(batch, 3, 32, 32, dtype=torch.float32)
        out = engine.infer(x_cpu.cuda())[0].cpu().numpy()
        np.testing.assert_allclose(out, _reference_output(dynamic_onnx_path, x_cpu.numpy()), atol=1e-4)
        assert engine._output_buffers[0].shape[0] == capacity


@pytest.fixture(scope="module")
def static_onnx_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx_static") / "static_model.onnx"
    torch.onnx.export(
        _Tiny().eval(), torch.randn(2, 3, 32, 32), str(path), input_names=["input"], output_names=["output"], opset_version=17
    )
    return str(path)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_static_batch_is_exact(static_onnx_path, device):
    if device == "cuda" and (not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers()):
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(static_onnx_path, device=device)
    assert not engine.is_dynamic
    assert engine.max_batch == 2
    assert engine.infer(torch.randn(2, 3, 32, 32, device=device))[0].shape == (2, 4)
    for batch in (1, 3):
        with pytest.raises(ValueError, match="requires batch size 2"):
            engine.infer(torch.randn(batch, 3, 32, 32, device=device))


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
        ONNXEngine(static_output_dim0_onnx_path, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unindexed_cuda_device_is_resolved(dynamic_onnx_path):
    """device='cuda' (no index) must resolve to a concrete cuda:N so torch allocations
    and ORT bindings target the same GPU."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda")
    assert engine.device.index is not None
    assert engine.device.index == engine._device_id
    # Sanity: output buffer ended up on the resolved device.
    assert engine._output_buffers[0].device.index == engine._device_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input_cuda(dynamic_onnx_path):
    """Regression test for bug #1: non-contiguous input must not produce a dangling pointer."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda")

    base = torch.randn(4, 3, 32, 64, dtype=torch.float32, device="cuda")
    x = base[:, :, :, :32]  # non-contiguous view
    assert not x.is_contiguous()

    out = engine.infer(x)[0].clone().cpu().numpy()
    ref = _reference_output(dynamic_onnx_path, x.contiguous().cpu().numpy())
    np.testing.assert_allclose(out, ref, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_still_being_written_by_torch_cuda(dynamic_onnx_path):
    """ORT must not read a bound input before torch's queued kernels have written it."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda")
    base_cpu = torch.randn(8, 3, 32, 32, dtype=torch.float32)
    base = base_cpu.cuda()
    busy = torch.randn(4096, 4096, device="cuda")

    for shift in range(1, 8):
        for _ in range(4):  # queue slow kernels ahead of the input's
            busy = (busy @ busy).clamp(-1, 1)
        x = base.roll(shift, dims=0)  # differs per pass so a stale buffer gives a wrong answer
        out = engine.infer(x)[0].cpu().numpy()
        ref = _reference_output(dynamic_onnx_path, base_cpu.roll(shift, dims=0).numpy())
        np.testing.assert_allclose(out, ref, atol=1e-4)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_input_on_other_gpu_rejected(dynamic_onnx_path):
    """An input on another GPU would skip the input sync, which only waits on the engine's GPU."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    engine = ONNXEngine(dynamic_onnx_path, device="cuda:0")
    x = torch.randn(1, 3, 32, 32, dtype=torch.float32, device="cuda:1")
    with pytest.raises(ValueError, match="expected"):
        engine.infer(x)


def test_indexed_cpu_device_accepts_cpu_inputs(dynamic_onnx_path):
    engine = ONNXEngine(dynamic_onnx_path, device="cpu:0")
    assert engine.infer(torch.randn(2, 3, 32, 32))[0].shape == (2, 4)


def test_cpu_output_buffers_hold_no_memory(dynamic_onnx_path):
    engine = ONNXEngine(dynamic_onnx_path, device="cpu")
    assert [(b.device.type, tuple(b.shape)) for b in engine._output_buffers] == [("meta", (1, 4))]


def test_non_cuda_non_cpu_device_rejected(dynamic_onnx_path):
    with pytest.raises(ValueError, match="CUDA or CPU device"):
        ONNXEngine(dynamic_onnx_path, device="meta")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_session_in_fresh_process(dynamic_onnx_path):
    """Other tests load the CUDA libs into this process, so only a fresh one shows whether ONNXEngine finds them itself."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    code = f"from lmi_common.onnx_engine import ONNXEngine; ONNXEngine({dynamic_onnx_path!r}, device='cuda')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


_CUDA_EP = torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers()
_needs_cuda_ep = pytest.mark.skipif(not _CUDA_EP, reason="onnxruntime-gpu / CUDAExecutionProvider not available")
_DEVICES = ["cpu", pytest.param("cuda", marks=_needs_cuda_ep)]


def _save_graph(path, nodes, initializers, inp_shape, out_shape):
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper

    inp = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, inp_shape)
    out = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, out_shape)
    graph = helper.make_graph(nodes, "g", [inp], [out], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


def _resize_onnx(path, rows_follow_batch=True, out_shape=("batch", "n")):
    """6 columns, and one row per image or, without rows_follow_batch, one per batch; ORT can't size it, as with F.interpolate."""
    onnx = pytest.importorskip("onnx")
    TensorProto, helper = onnx.TensorProto, onnx.helper

    rows = "batch" if rows_follow_batch else "rows"
    nodes = [
        helper.make_node("Shape", ["input"], ["shape"]),
        helper.make_node("Slice", ["shape", "zero", "one"], ["batch"]),
        helper.make_node("Div", ["batch", "batch"], ["rows"]),
        helper.make_node("Concat", [rows, "width"], ["sizes"], axis=0),
        helper.make_node("Resize", ["input", "", "", "sizes"], ["output"], mode="nearest"),
    ]
    inits = [helper.make_tensor(n, TensorProto.INT64, [1], [v]) for n, v in (("zero", 0), ("one", 1), ("width", 6))]
    return _save_graph(path, nodes, inits, ["batch", 4], list(out_shape))


def _slice_onnx(path, inp_shape, axis=1, extra=1.0):
    """Keeps the first max(x) + extra entries along axis."""
    onnx = pytest.importorskip("onnx")
    TensorProto, helper = onnx.TensorProto, onnx.helper

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["max"], keepdims=0),
        helper.make_node("Add", ["max", "one"], ["end_f"]),
        helper.make_node("Cast", ["end_f"], ["end_s"], to=TensorProto.INT64),
        helper.make_node("Reshape", ["end_s", "shape1"], ["end"]),
        helper.make_node("Slice", ["input", "zero", "end", "axis"], ["output"]),
    ]
    inits = [helper.make_tensor("one", TensorProto.FLOAT, [], [extra])]
    inits += [helper.make_tensor(n, TensorProto.INT64, [1], [v]) for n, v in (("shape1", 1), ("zero", 0), ("axis", axis))]
    out_shape = ["n", inp_shape[1]] if axis == 0 else [inp_shape[0], "n"]
    return _save_graph(path, nodes, inits, inp_shape, out_shape)


@pytest.mark.parametrize("device", _DEVICES)
def test_symbolic_output_dims_are_sized_at_load(tmp_path, device):
    path = _resize_onnx(tmp_path / "resize.onnx")
    assert not isinstance(ort.InferenceSession(path, providers=["CPUExecutionProvider"]).get_outputs()[0].shape[1], int)

    engine = ONNXEngine(path, device=device)
    assert engine._output_buffers[0].shape[1:] == (6,)
    x = torch.randn(3, 4, device=device)
    torch.testing.assert_close(engine.infer(x)[0], x[:, [0, 0, 1, 2, 2, 3]])


@pytest.mark.parametrize(
    "out_shape", [("batch", "n"), ("rows", 6), ("batch", 6)], ids=["symbolic width", "only dim 0 symbolic", "named like the batch"]
)
def test_dynamic_batch_output_dim0_that_does_not_follow_the_batch_is_rejected(tmp_path, out_shape):
    path = _resize_onnx(tmp_path / "resize.onnx", rows_follow_batch=False, out_shape=out_shape)
    with pytest.raises(NotImplementedError, match=r"shape \(1, 6\) at batch 2"):
        ONNXEngine(path, device="cpu")


def test_dynamic_batch_output_dim0_named_apart_from_the_batch_loads_when_it_follows(tmp_path):
    engine = ONNXEngine(_resize_onnx(tmp_path / "resize.onnx", out_shape=("rows", 6)), device="cpu")
    assert engine.infer(torch.randn(3, 4))[0].shape == (3, 6)


def test_output_empty_on_zeros_is_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match="is empty"):
        ONNXEngine(_slice_onnx(tmp_path / "slice.onnx", [2, 4], axis=0, extra=0.0), device="cpu")


def test_static_batch_output_dims_are_sized_at_that_batch(tmp_path):
    engine = ONNXEngine(_slice_onnx(tmp_path / "slice.onnx", [2, 4]), device="cpu")
    assert tuple(engine._output_buffers[0].shape) == (2, 1)


@pytest.mark.parametrize("device", _DEVICES)
def test_static_batch_output_dim0_is_sized_from_the_run(tmp_path, device):
    """Zeros keep one row of the batch-2 input, so dim 0 is 1, not the batch."""
    engine = ONNXEngine(_slice_onnx(tmp_path / "slice.onnx", [2, 4], axis=0), device=device)
    assert tuple(engine._output_buffers[0].shape) == (1, 4)
    x = torch.zeros(2, 4, device=device)
    torch.testing.assert_close(engine.infer(x)[0], x[:1])


@_needs_cuda_ep
def test_data_dependent_output_dim_fails_when_it_changes_cuda(tmp_path):
    """Zeros at load size the output to 1 column; twos need 3."""
    engine = ONNXEngine(_slice_onnx(tmp_path / "slice.onnx", ["batch", 4]), device="cuda")
    assert engine.infer(torch.zeros(1, 4, device="cuda"))[0].shape == (1, 1)
    with pytest.raises(Exception, match="shape verification failed"):
        engine.infer(torch.full((1, 4), 2.0, device="cuda"))
