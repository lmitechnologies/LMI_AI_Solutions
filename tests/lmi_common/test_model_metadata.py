"""Metadata embedded in a model file, so a deployed model needs no sidecar.

Covers both halves of the convention: an ONNX carries the payload in its metadata_props, an engine in a
length-tagged JSON header that TensorRT would otherwise choke on. Metadata is optional throughout —
files without any must still load, reporting {}.
"""

import json

import pytest
import torch

onnx = pytest.importorskip("onnx")

from lmi_common.model_metadata import (  # noqa: E402
    ONNX_METADATA_KEY,
    embed_onnx_metadata,
    engine_metadata_header,
    metadata_from_onnx_props,
    read_onnx_metadata,
    split_engine_metadata,
)

try:
    import tensorrt  # noqa: F401

    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False

needs_engine = pytest.mark.skipif(
    not (_HAS_TRT and torch.cuda.is_available()),
    reason="engine round trip needs TensorRT and a CUDA torch",
)

PAYLOAD = {"class_names": ["cat", "dog"]}


@pytest.fixture(scope="module")
def plain_onnx(tmp_path_factory):
    """A minimal ONNX with no metadata of its own."""
    path = tmp_path_factory.mktemp("meta") / "plain.onnx"
    torch.onnx.export(
        torch.nn.Conv2d(3, 8, 3).eval(),
        torch.zeros(1, 3, 32, 32),
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    return str(path)


def test_onnx_round_trip(plain_onnx, tmp_path):
    import shutil

    path = str(tmp_path / "embedded.onnx")
    shutil.copy(plain_onnx, path)
    embed_onnx_metadata(path, PAYLOAD)
    assert read_onnx_metadata(path) == PAYLOAD


def test_onnx_embed_is_idempotent(plain_onnx, tmp_path):
    """Re-exporting over an embedded file must overwrite the payload, not add a second property."""
    import shutil

    path = str(tmp_path / "twice.onnx")
    shutil.copy(plain_onnx, path)
    embed_onnx_metadata(path, PAYLOAD)
    embed_onnx_metadata(path, {"class_names": ["bird"]})
    assert read_onnx_metadata(path) == {"class_names": ["bird"]}
    assert [p.key for p in onnx.load(path).metadata_props].count(ONNX_METADATA_KEY) == 1


def test_onnx_without_metadata(plain_onnx):
    assert read_onnx_metadata(plain_onnx) == {}


@pytest.mark.parametrize("props", [None, {}, {"author": "someone else"}, {ONNX_METADATA_KEY: "not json"}, {ONNX_METADATA_KEY: "[1, 2]"}])
def test_props_without_usable_payload(props):
    assert metadata_from_onnx_props(props) == {}


def test_onnx_engine_reports_metadata(plain_onnx, tmp_path):
    """What the ONNX backend actually consumes: onnxruntime's own view of the embedded payload."""
    pytest.importorskip("onnxruntime")
    import shutil

    from lmi_common.onnx_engine import ONNXEngine

    path = str(tmp_path / "ort.onnx")
    shutil.copy(plain_onnx, path)
    embed_onnx_metadata(path, PAYLOAD)
    assert ONNXEngine(path, device="cpu").metadata == PAYLOAD
    assert ONNXEngine(plain_onnx, device="cpu").metadata == {}


def test_header_round_trip():
    plan = b"ftrt" + b"\x00" * 64
    metadata, stripped = split_engine_metadata(engine_metadata_header(PAYLOAD) + plan)
    assert metadata == PAYLOAD
    assert stripped == plan


def test_headerless_plan_is_untouched():
    plan = b"ftrt" + json.dumps(PAYLOAD).encode()  # magic first: the JSON-looking tail must not be read as a header
    assert split_engine_metadata(plan) == ({}, plan)


@pytest.mark.parametrize("raw", [b"", b"\x00\x00\x00\x00", b"\xff\xff\xff\xffnope", (10).to_bytes(4, "little") + b"not json.."])
def test_undecodable_header_leaves_bytes_alone(raw):
    assert split_engine_metadata(raw) == ({}, raw)


@needs_engine
def test_engine_carries_metadata_to_trt_engine(plain_onnx, tmp_path):
    """The round trip that matters: metadata embedded in an ONNX survives the build and the plan still deserializes."""
    import shutil

    from lmi_common.trt_convert import onnx_to_trt
    from lmi_common.trt_engine import TRTEngine

    onnx_path = str(tmp_path / "src.onnx")
    shutil.copy(plain_onnx, onnx_path)
    embed_onnx_metadata(onnx_path, PAYLOAD)

    engine_path = str(tmp_path / "src.engine")
    onnx_to_trt(onnx_path, engine_path, fp16=False, workspace_gb=2)
    assert TRTEngine(engine_path, device="cuda").metadata == PAYLOAD


@needs_engine
def test_engine_without_metadata_still_loads(plain_onnx, tmp_path):
    from lmi_common.trt_convert import onnx_to_trt
    from lmi_common.trt_engine import TRTEngine

    engine_path = str(tmp_path / "plain.engine")
    onnx_to_trt(plain_onnx, engine_path, fp16=False, workspace_gb=2)
    assert TRTEngine(engine_path, device="cuda").metadata == {}
