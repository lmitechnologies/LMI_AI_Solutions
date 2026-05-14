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
