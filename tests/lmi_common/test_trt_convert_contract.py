"""What survives the ONNX → TensorRT conversion of an ultralytics model.

Ultralytics records end2end/task/names/imgsz in the ONNX and reads them back off the plan file on load. Nothing here asserts that
file layout: the engine is loaded through ultralytics' own AutoBackend, so these fail if a future ultralytics changes either half
of the convention. Without the metadata a YOLO26 segmentation engine decodes down the wrong branch, since its (1, 300, 38) output
misses the shape check that saves the detect head.
"""

import inspect
import shutil
from pathlib import Path

import pytest
import torch

onnx = pytest.importorskip("onnx")

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.autobackend import AutoBackend  # noqa: E402

from lmi_common.trt_convert import _inspect_onnx, onnx_to_trt  # noqa: E402

ASSET_MODEL = "tests/assets/models/od/ultralytics/yolo11n.pt"
POSE_ASSET = "tests/assets/models/od/ultralytics/yolo11n-pose.pt"
IMGSZ = [640, 640]

try:
    import tensorrt  # noqa: F401

    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False

# The build needs TensorRT and a GPU; loading the result back needs a CUDA-enabled torch.
needs_engine = pytest.mark.skipif(
    not (_HAS_TRT and torch.cuda.is_available()),
    reason="engine round trip needs TensorRT and a CUDA torch",
)


def _claim(onnx_path: str):
    """What onnx_to_trt decides about an ONNX: its props if onnx2engine will build it, else None."""
    props, use_onnx2engine = _inspect_onnx(onnx_path)
    return props if use_onnx2engine else None


def _export(tmp_dir, asset: str = ASSET_MODEL, task: str = "detect", **kwargs) -> str:
    """Export an asset model to ONNX inside tmp_dir; ultralytics writes next to the weights, so copy them there first."""
    weights = tmp_dir / Path(asset).name
    shutil.copy(asset, weights)
    return YOLO(str(weights), task=task).export(format="onnx", imgsz=IMGSZ, simplify=True, **kwargs)


@pytest.fixture(scope="module")
def ultralytics_onnx(tmp_path_factory):
    """A real ultralytics ONNX export, the input the gadget hands the converter."""
    return _export(tmp_path_factory.mktemp("ul_onnx"), dynamic=False)


def test_metadata_detected_on_ultralytics_export(ultralytics_onnx):
    metadata = _claim(ultralytics_onnx)
    assert metadata is not None, "ultralytics stopped marking its exports with author=Ultralytics"
    assert metadata["task"] == "detect"
    assert "end2end" in metadata


def test_onnx2engine_call_surface_is_unchanged():
    """The kwargs onnx_to_trt passes onnx2engine, pinned without a GPU: otherwise a rename surfaces as a TypeError mid-build."""
    try:
        from ultralytics.utils.export import onnx2engine
    except ImportError as e:
        pytest.fail(f"onnx2engine moved; onnx_to_trt now silently falls back to the local builder: {e}")

    params = inspect.signature(onnx2engine).parameters
    assert "metadata" in params, "onnx2engine no longer takes metadata; nothing would embed the props map"
    assert "workspace" in params, "onnx2engine renamed its workspace argument"
    assert "quantize" in params or "half" in params, "onnx2engine changed its precision selector again"


def test_plain_onnx_is_not_claimed(tmp_path):
    """A non-ultralytics ONNX must keep using the builder that owns the batch profile."""
    path = tmp_path / "plain.onnx"
    torch.onnx.export(
        torch.nn.Conv2d(3, 8, 3).eval(),
        torch.zeros(1, 3, 32, 32),
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    assert _claim(str(path)) is None


def test_dynamic_shape_export_is_not_claimed(tmp_path_factory):
    """onnx2engine cannot express a batch-only profile, so dynamic models stay on the local builder."""
    exported = _export(tmp_path_factory.mktemp("ul_dyn"), dynamic=True)
    assert _claim(exported) is None


@needs_engine
def test_engine_carries_metadata_back_to_autobackend(ultralytics_onnx, tmp_path):
    """The round trip that matters: what AutoBackend reports off the built engine."""
    engine_path = tmp_path / "model.engine"
    onnx_to_trt(ultralytics_onnx, str(engine_path), fp16=False, workspace_gb=2)

    backend = AutoBackend(str(engine_path), torch.device("cuda:0"))
    assert backend.end2end is False, "yolo11 is not end-to-end; a True here means the flag is not being read"
    assert backend.task == "detect"
    assert list(backend.imgsz) == IMGSZ
    assert len(backend.names) == 80


@needs_engine
def test_engine_carries_kpt_shape(tmp_path_factory, tmp_path):
    """YoloPose reshapes the keypoint block with model.kpt_shape, which exists only in the metadata."""
    onnx_path = _export(tmp_path_factory.mktemp("ul_pose"), asset=POSE_ASSET, task="pose", dynamic=False)
    engine_path = tmp_path / "pose.engine"
    onnx_to_trt(onnx_path, str(engine_path), fp16=False, workspace_gb=2)

    backend = AutoBackend(str(engine_path), torch.device("cuda:0"))
    assert list(backend.kpt_shape) == [17, 3]
    assert backend.task == "pose"


@needs_engine
def test_our_metadata_survives_the_ultralytics_builder(ultralytics_onnx, tmp_path):
    """onnx2engine writes the whole props map, so our payload must come back off its engine too, not just ours."""
    from lmi_common.model_metadata import embed_onnx_metadata
    from lmi_common.trt_engine import TRTEngine

    annotated = tmp_path / "annotated.onnx"
    shutil.copy(ultralytics_onnx, annotated)
    embed_onnx_metadata(str(annotated), {"class_names": ["cat", "dog"]})

    engine_path = tmp_path / "annotated.engine"
    onnx_to_trt(str(annotated), str(engine_path), fp16=False, workspace_gb=2)

    assert TRTEngine(str(engine_path), device="cuda").metadata == {"class_names": ["cat", "dog"]}
    assert AutoBackend(str(engine_path), torch.device("cuda:0")).task == "detect", "ultralytics' own keys must survive too"
