"""Metadata embedded in a model file, so a deployed model needs no sidecar.

Covers both halves of the convention: an ONNX carries the payload in its metadata_props, an engine in a
length-tagged JSON header that TensorRT would otherwise choke on. Both hold the same props map, so the
asserts below pair them off. Metadata is optional throughout — files without any must still load,
reporting {}.
"""

import json
import os
import shutil
import stat

import pytest
import torch

onnx = pytest.importorskip("onnx")

from lmi_common.model_metadata import (  # noqa: E402
    _MAX_HEADER_BYTES,
    METADATA_KEY,
    SCHEMA_VERSION,
    embed_onnx_metadata,
    encode_metadata,
    engine_props_header,
    metadata_from_props,
    read_onnx_props,
    split_engine_props,
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


@pytest.fixture
def embedded_onnx(plain_onnx, tmp_path):
    """A copy of the plain ONNX carrying PAYLOAD."""
    path = str(tmp_path / "embedded.onnx")
    shutil.copy(plain_onnx, path)
    embed_onnx_metadata(path, PAYLOAD)
    return path


def test_onnx_round_trip(embedded_onnx):
    assert metadata_from_props(read_onnx_props(embedded_onnx)) == PAYLOAD


def test_payload_travels_in_a_versioned_envelope(embedded_onnx):
    """The version is what lets a future reader recognize a format change instead of guessing."""
    assert json.loads(read_onnx_props(embedded_onnx)[METADATA_KEY]) == {"version": SCHEMA_VERSION, "payload": PAYLOAD}


def test_a_future_version_is_still_read(caplog):
    """A payload of optional keys survives a version bump, so warn rather than drop the model's class names."""
    raw = json.dumps({"version": SCHEMA_VERSION + 1, "payload": PAYLOAD})
    assert metadata_from_props({METADATA_KEY: raw}) == PAYLOAD
    assert "version" in caplog.text


def test_onnx_embed_is_idempotent(embedded_onnx):
    """Re-exporting over an embedded file must overwrite the payload, not add a second property."""
    embed_onnx_metadata(embedded_onnx, {"class_names": ["bird"]})
    assert metadata_from_props(read_onnx_props(embedded_onnx)) == {"class_names": ["bird"]}
    assert [p.key for p in onnx.load(embedded_onnx).metadata_props].count(METADATA_KEY) == 1


def test_embed_keeps_the_file_mode(plain_onnx, tmp_path):
    """The swap must not hand back the 0600 of a temp file; a deployed model is often group-readable."""
    path = str(tmp_path / "mode.onnx")
    shutil.copy(plain_onnx, path)
    os.chmod(path, 0o644)
    embed_onnx_metadata(path, PAYLOAD)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o644


def test_a_failed_save_leaves_the_export_intact(plain_onnx, tmp_path, monkeypatch):
    """Embedding is an in-place rewrite, so a save that dies partway must not destroy the export."""
    path = tmp_path / "kept.onnx"
    shutil.copy(plain_onnx, path)
    before = path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(onnx, "save", boom)
    with pytest.raises(OSError):
        embed_onnx_metadata(str(path), PAYLOAD)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_onnx_without_metadata(plain_onnx):
    assert read_onnx_props(plain_onnx) == {}


@pytest.mark.parametrize(
    "props",
    [
        None,
        {},
        {"author": "someone else"},
        {METADATA_KEY: "not json"},
        {METADATA_KEY: "[1, 2]"},
        {METADATA_KEY: json.dumps(PAYLOAD)},  # a bare payload, missing the envelope
        {METADATA_KEY: json.dumps({"version": SCHEMA_VERSION, "payload": [1, 2]})},
    ],
)
def test_props_without_usable_payload(props):
    assert metadata_from_props(props) == {}


def test_onnx_engine_reports_metadata(plain_onnx, embedded_onnx):
    """What the ONNX backend actually consumes: onnxruntime's own view of the embedded payload."""
    pytest.importorskip("onnxruntime")

    from lmi_common.onnx_engine import ONNXEngine

    assert ONNXEngine(embedded_onnx, device="cpu").metadata == PAYLOAD
    assert ONNXEngine(plain_onnx, device="cpu").metadata == {}


def test_header_round_trip():
    plan = b"ftrt" + b"\x00" * 64
    props, stripped = split_engine_props(engine_props_header({METADATA_KEY: encode_metadata(PAYLOAD)}) + plan)
    assert metadata_from_props(props) == PAYLOAD
    assert stripped == plan


def test_an_engine_header_holds_the_same_props_an_onnx_does(embedded_onnx):
    """One shape for both files, so a single reader decodes either and neither side needs its own rules."""
    props = read_onnx_props(embedded_onnx)
    assert split_engine_props(engine_props_header(props) + b"ftrt")[0] == props


def test_a_foreign_header_keeps_its_own_keys():
    """An ultralytics engine header must survive the read: unknown keys pass through, ours is absent."""
    foreign = {"author": "Ultralytics", "task": "detect"}
    props, _ = split_engine_props(engine_props_header(foreign) + b"ftrt")
    assert props == foreign
    assert metadata_from_props(props) == {}


def test_an_unreadable_header_is_refused_at_write_time():
    """Past the reader's cap the header reads back as plan bytes, and TensorRT then blames the plan for it."""
    with pytest.raises(ValueError, match="cap"):
        engine_props_header({METADATA_KEY: "x" * (_MAX_HEADER_BYTES + 1)})


def test_headerless_plan_is_untouched():
    plan = b"ftrt" + json.dumps(PAYLOAD).encode()  # magic first: the JSON-looking tail must not be read as a header
    assert split_engine_props(plan) == ({}, plan)


@pytest.mark.parametrize("raw", [b"", b"\x00\x00\x00\x00", b"\xff\xff\xff\xffnope", (10).to_bytes(4, "little") + b"not json.."])
def test_undecodable_header_leaves_bytes_alone(raw):
    assert split_engine_props(raw) == ({}, raw)


@needs_engine
def test_engine_carries_metadata_to_trt_engine(embedded_onnx, tmp_path):
    """The round trip that matters: metadata embedded in an ONNX survives the build and the plan still deserializes."""
    from lmi_common.trt_convert import onnx_to_trt
    from lmi_common.trt_engine import TRTEngine

    engine_path = str(tmp_path / "src.engine")
    onnx_to_trt(embedded_onnx, engine_path, fp16=False, workspace_gb=2)
    assert TRTEngine(engine_path, device="cuda").metadata == PAYLOAD


@needs_engine
def test_engine_without_metadata_still_loads(plain_onnx, tmp_path):
    from lmi_common.trt_convert import onnx_to_trt
    from lmi_common.trt_engine import TRTEngine

    engine_path = str(tmp_path / "plain.engine")
    onnx_to_trt(plain_onnx, engine_path, fp16=False, workspace_gb=2)
    assert TRTEngine(engine_path, device="cuda").metadata == {}
