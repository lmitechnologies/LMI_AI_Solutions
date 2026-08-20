"""Which ONNX layouts the local TensorRT builder accepts.

A model past the 2 GB protobuf limit keeps its weights in a sidecar file, and that sidecar resolves only
relative to the model's own directory — so the builder must hand TensorRT a path, never a byte buffer.
"""

import pytest
import torch

onnx = pytest.importorskip("onnx")

from lmi_common.trt_convert import onnx_to_trt  # noqa: E402

try:
    import tensorrt  # noqa: F401

    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False

needs_engine = pytest.mark.skipif(
    not (_HAS_TRT and torch.cuda.is_available()),
    reason="engine build needs TensorRT and a CUDA torch",
)


@pytest.fixture
def external_data_onnx(tmp_path):
    """An ONNX whose weights live in a sidecar, the layout every >2 GB export uses."""
    inline = tmp_path / "inline.onnx"
    torch.onnx.export(
        torch.nn.Conv2d(3, 64, 7),
        torch.zeros(1, 3, 64, 64),
        str(inline),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    path = tmp_path / "external.onnx"
    onnx.save(
        onnx.load(str(inline)),
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="external.onnx.data",
        size_threshold=0,
    )
    assert (tmp_path / "external.onnx.data").exists(), "onnx did not externalize the weights"
    return path


@needs_engine
def test_external_weights_build(external_data_onnx, tmp_path):
    engine_path = tmp_path / "external.engine"
    onnx_to_trt(str(external_data_onnx), str(engine_path), fp16=False, workspace_gb=2)
    assert engine_path.stat().st_size > 0


@needs_engine
def test_a_lost_sidecar_still_raises(external_data_onnx, tmp_path):
    """Resolving the sidecar must not turn a missing one into a silent partial build."""
    (tmp_path / "external.onnx.data").unlink()
    with pytest.raises(RuntimeError, match="Failed to parse ONNX model"):
        onnx_to_trt(str(external_data_onnx), str(tmp_path / "nope.engine"), fp16=False, workspace_gb=2)
