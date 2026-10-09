import numpy as np
import pytest
import torch
import torch.nn as nn

trt = pytest.importorskip("tensorrt")
ort = pytest.importorskip("onnxruntime")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from lmi_common.trt_engine import TRTEngine  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


def _export_onnx(path: str, dynamic: bool) -> None:
    model = _Tiny().eval()
    dummy = torch.randn(1, 3, 32, 32)
    kwargs = {}
    if dynamic:
        kwargs["dynamic_axes"] = {"input": {0: "batch"}, "output": {0: "batch"}}
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        **kwargs,
    )


def _build_engine(onnx_path: str, engine_path: str, *, min_b: int, opt_b: int, max_b: int, static: bool) -> None:
    """Build a TRT engine from an ONNX file. If static, no profile is added and the engine has
    a fixed batch dim equal to whatever the ONNX declares (1 here)."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256 MiB

    if not static:
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (min_b, 3, 32, 32), (opt_b, 3, 32, 32), (max_b, 3, 32, 32))
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None")
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))


@pytest.fixture(scope="module")
def dynamic_engine_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("trt")
    onnx_path = str(d / "dyn.onnx")
    engine_path = str(d / "dyn.engine")
    _export_onnx(onnx_path, dynamic=True)
    _build_engine(onnx_path, engine_path, min_b=1, opt_b=4, max_b=8, static=False)
    return onnx_path, engine_path


@pytest.fixture(scope="module")
def static_engine_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("trt_static")
    onnx_path = str(d / "static.onnx")
    engine_path = str(d / "static.engine")
    _export_onnx(onnx_path, dynamic=False)
    _build_engine(onnx_path, engine_path, min_b=0, opt_b=0, max_b=0, static=True)
    return onnx_path, engine_path


def _ort_reference(onnx_path: str, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


@pytest.mark.parametrize("actual_batch", [1, 3, 8])
def test_dynamic_batch_matches_reference(dynamic_engine_path, actual_batch):
    """Drives bug #1: TRTEngine.__init__ on a dynamic engine currently calls
    get_tensor_profile_shape on output bindings, which is invalid in TRT. This test
    will fail at TRTEngine(...) until the output-shape derivation is fixed."""
    onnx_path, engine_path = dynamic_engine_path
    engine = TRTEngine(engine_path, device="cuda")
    assert engine.is_dynamic
    assert engine.max_batch == 8

    x_cpu = torch.randn(actual_batch, 3, 32, 32, dtype=torch.float32)
    x_cuda = x_cpu.cuda()
    out = engine.infer(x_cuda)[0].clone().cpu().numpy()

    ref = _ort_reference(onnx_path, x_cpu.numpy())
    assert out.shape == ref.shape == (actual_batch, 4)
    np.testing.assert_allclose(out, ref, atol=1e-3)


def test_copy_true_returns_independent_tensors(dynamic_engine_path):
    """copy=True (default): two sequential infer() calls must produce tensors that do not
    alias each other. The first result must stay stable after the second call overwrites
    the engine's internal buffer."""
    _, engine_path = dynamic_engine_path
    engine = TRTEngine(engine_path, device="cuda")

    x1 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")
    x2 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")

    out1 = engine.infer(x1)[0]
    snapshot = out1.clone()
    out2 = engine.infer(x2)[0]

    assert out1.data_ptr() != out2.data_ptr()
    torch.testing.assert_close(out1, snapshot)  # out1 unchanged by the second infer()
    assert not torch.allclose(out1, out2)  # sanity: different inputs → different outputs


def test_copy_false_returns_aliased_views(dynamic_engine_path):
    """copy=False: two sequential infer() calls return views into the same internal buffer.
    The first result is overwritten by the second call. Documented behavior — this test
    pins it so future changes don't silently break the opt-out contract."""
    _, engine_path = dynamic_engine_path
    engine = TRTEngine(engine_path, device="cuda")

    x1 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")
    x2 = torch.randn(4, 3, 32, 32, dtype=torch.float32, device="cuda")

    out1 = engine.infer(x1, copy=False)[0]
    snapshot = out1.clone()
    out2 = engine.infer(x2, copy=False)[0]

    assert out1.data_ptr() == out2.data_ptr()
    torch.testing.assert_close(out1, out2)  # out1 now reflects x2, not x1
    assert not torch.allclose(out1, snapshot)


def test_static_engine_matches_reference(static_engine_path):
    """Sanity check that the static-engine path still works (this path doesn't touch
    get_tensor_profile_shape, so it should pass even before bug #1 is fixed)."""
    onnx_path, engine_path = static_engine_path
    engine = TRTEngine(engine_path, device="cuda")
    assert not engine.is_dynamic
    assert engine.max_batch == 1

    x_cpu = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    out = engine.infer(x_cpu.cuda())[0].clone().cpu().numpy()
    ref = _ort_reference(onnx_path, x_cpu.numpy())
    np.testing.assert_allclose(out, ref, atol=1e-3)


def test_input_still_being_written_by_torch(dynamic_engine_path):
    """TRT must not read an input before torch's queued kernels have written it."""
    onnx_path, engine_path = dynamic_engine_path
    engine = TRTEngine(engine_path, device="cuda")
    base_cpu = torch.randn(8, 3, 32, 32, dtype=torch.float32)
    base = base_cpu.cuda()
    busy = torch.randn(4096, 4096, device="cuda")

    for shift in range(1, 8):
        for _ in range(4):  # queue slow kernels ahead of the input's
            busy = (busy @ busy).clamp(-1, 1)
        x = base.roll(shift, dims=0)  # differs per pass so a stale buffer gives a wrong answer
        out = engine.infer(x)[0].cpu().numpy()
        ref = _ort_reference(onnx_path, base_cpu.roll(shift, dims=0).numpy())
        np.testing.assert_allclose(out, ref, atol=1e-3)


def test_infer_outside_default_stream_raises(dynamic_engine_path):
    """infer() relies on the default stream to order torch's writes before TRT's reads."""
    _, engine_path = dynamic_engine_path
    engine = TRTEngine(engine_path, device="cuda")
    x = torch.randn(1, 3, 32, 32, dtype=torch.float32, device="cuda")

    with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match="default CUDA stream"):
        engine.infer(x)


def test_unindexed_cuda_device_is_resolved(dynamic_engine_path):
    _, engine_path = dynamic_engine_path
    engine = TRTEngine(engine_path, device="cuda")
    assert engine.device == torch.device(f"cuda:{torch.cuda.current_device()}")


def test_batch_outside_profile_raises(tmp_path):
    """TensorRT rejects a shape outside the profile without raising; infer() must not run on the old shape."""
    onnx_path = str(tmp_path / "dyn.onnx")
    engine_path = str(tmp_path / "dyn_min2.engine")
    _export_onnx(onnx_path, dynamic=True)
    _build_engine(onnx_path, engine_path, min_b=2, opt_b=4, max_b=8, static=False)
    engine = TRTEngine(engine_path, device="cuda")

    x = torch.randn(1, 3, 32, 32, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="optimization profile"):
        engine.infer(x)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_engine_on_non_current_device(dynamic_engine_path):
    """The engine must load on the requested GPU even when another GPU is current."""
    onnx_path, engine_path = dynamic_engine_path
    torch.cuda.set_device(0)
    engine = TRTEngine(engine_path, device="cuda:1")

    x_cpu = torch.randn(3, 3, 32, 32, dtype=torch.float32)
    out = engine.infer(x_cpu.to("cuda:1"))[0].cpu().numpy()
    np.testing.assert_allclose(out, _ort_reference(onnx_path, x_cpu.numpy()), atol=1e-3)


def test_non_cuda_device_rejected(dynamic_engine_path):
    _, engine_path = dynamic_engine_path
    with pytest.raises(ValueError, match="CUDA device"):
        TRTEngine(engine_path, device="cpu")


class _NonZero(nn.Module):
    def forward(self, x):
        return torch.nonzero(x > 0).to(torch.int32)


def test_data_dependent_output_shape_rejected(tmp_path):
    onnx_path = str(tmp_path / "nonzero.onnx")
    engine_path = str(tmp_path / "nonzero.engine")
    torch.onnx.export(_NonZero(), torch.randn(1, 3, 32, 32), onnx_path, input_names=["input"], output_names=["output"], opset_version=17)
    _build_engine(onnx_path, engine_path, min_b=0, opt_b=0, max_b=0, static=True)
    with pytest.raises(NotImplementedError, match="data-dependent shape"):
        TRTEngine(engine_path, device="cuda")


class _FlattenBatch(nn.Module):
    def forward(self, x):
        return x.flatten(0, 1)


def test_output_dim0_not_equal_to_batch_rejected(tmp_path):
    """An output of B*C rows would be cut to B rows by the batch slice in infer()."""
    onnx_path = str(tmp_path / "flatten.onnx")
    engine_path = str(tmp_path / "flatten.engine")
    torch.onnx.export(
        _FlattenBatch(),
        torch.randn(1, 3, 32, 32),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "rows"}},
        opset_version=17,
    )
    _build_engine(onnx_path, engine_path, min_b=1, opt_b=4, max_b=8, static=False)
    with pytest.raises(NotImplementedError, match="dim 0 equals the batch"):
        TRTEngine(engine_path, device="cuda")
