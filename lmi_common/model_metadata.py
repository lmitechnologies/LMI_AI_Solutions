"""Embed and read model metadata (class names, ...) so a deployed model is a single file.

An ONNX file carries the payload in its ``metadata_props``. A serialized TensorRT plan has no metadata
slot, so the engine file is prefixed with a length-tagged JSON header holding that same props map — the
layout ultralytics uses, which keeps their engines readable here and ours readable there.

Two levels, kept distinct throughout: *props* is the annotation map as it sits on the wire, shared with
whatever else marked the file; *metadata* is our own payload, a versioned envelope under a single props key.
An ONNX and the engine built from it therefore carry byte-identical props, and one reader decodes both.
ONNX limits props values to strings; a header written elsewhere can hold any JSON, so readers accept that.

Metadata is always optional: readers return ``{}`` for a file that carries none, and writers skip an empty
payload, so plain ONNX files and headerless engines keep working unchanged.

Anything that re-saves an ONNX (onnx-graphsurgeon, onnxsim) drops metadata_props, so embed last.
"""

import json
import logging
import os
import stat
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

METADATA_KEY = "lmi_metadata"
SCHEMA_VERSION = 1

_PLAN_MAGIC = b"ftrt"  # first bytes of a headerless serialized TensorRT plan
_MAX_HEADER_BYTES = 1 << 20


def embed_onnx_metadata(onnx_path: str, metadata: Mapping[str, Any]) -> None:
    """Write ``metadata`` into an ONNX file's metadata_props, in place. Requires the ``onnx`` package."""
    import onnx

    # load_external_data=False leaves weights in their .data file: nothing to inline, nothing to orphan.
    model = onnx.load(onnx_path, load_external_data=False)
    prop = next((p for p in model.metadata_props if p.key == METADATA_KEY), None)
    if prop is None:
        prop = model.metadata_props.add()
        prop.key = METADATA_KEY
    prop.value = encode_metadata(metadata)
    _save_in_place(model, onnx_path)
    logger.info(f"Embedded metadata {sorted(metadata)} in {onnx_path}")


def _save_in_place(model, onnx_path: str) -> None:
    """Swap in a fully-written temp file, so a save that dies partway cannot destroy the export.

    The temp file sits in the model's own directory, where its relative external-data paths still resolve.
    """
    import onnx

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(onnx_path)), suffix=".onnx.tmp")
    os.close(fd)
    try:
        os.chmod(tmp, stat.S_IMODE(os.stat(onnx_path).st_mode))  # mkstemp is 0600; keep the export's own mode
        onnx.save(model, tmp)
        os.replace(tmp, onnx_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_onnx_props(onnx_path: str) -> Dict[str, str]:
    """Read an ONNX file's whole metadata_props map — ours and anyone else's."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    return {p.key: p.value for p in model.metadata_props}


def encode_metadata(metadata: Mapping[str, Any]) -> str:
    """Encode ``metadata`` as the versioned envelope that sits under ``METADATA_KEY`` in a props map."""
    return json.dumps({"version": SCHEMA_VERSION, "payload": dict(metadata)})


def metadata_from_props(props: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Decode our payload out of a props map: an ONNX's metadata_props, onnxruntime's
    ``custom_metadata_map``, or an engine header."""
    raw = (props or {}).get(METADATA_KEY)
    if not raw or not isinstance(raw, str):
        return {}
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Ignoring unreadable '{METADATA_KEY}' metadata")
        return {}
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        logger.warning(f"Ignoring '{METADATA_KEY}' metadata that is not a versioned envelope")
        return {}
    version = envelope.get("version")
    if version != SCHEMA_VERSION:
        # The payload is a bag of optional keys, so reading an unknown version beats refusing to load.
        logger.warning(f"'{METADATA_KEY}' metadata is version {version}, this build writes {SCHEMA_VERSION}")
    return envelope["payload"]


def engine_props_header(props: Mapping[str, Any]) -> bytes:
    """Serialize a props map as the length-tagged JSON header that prefixes an engine plan.

    Raises ValueError past the reader's cap: an oversized header is written fine but read back as plan
    bytes, and TensorRT then blames the plan for a header it never saw.
    """
    payload = json.dumps(dict(props)).encode("utf-8")
    if len(payload) > _MAX_HEADER_BYTES:
        raise ValueError(f"Engine metadata header is {len(payload)} bytes; readers cap it at {_MAX_HEADER_BYTES}")
    return len(payload).to_bytes(4, byteorder="little", signed=True) + payload


def split_engine_props(raw: bytes) -> Tuple[Dict[str, Any], bytes]:
    """Split engine-file bytes into (props map, serialized plan).

    A plan carrying no header is returned untouched with an empty map. TensorRT rejects a plan with
    leading bytes, so the header must be stripped before deserializing.
    """
    if raw[:4] == _PLAN_MAGIC:
        return {}, raw
    length = int.from_bytes(raw[:4], byteorder="little", signed=True)
    if not 0 < length <= min(_MAX_HEADER_BYTES, len(raw) - 4):
        return {}, raw
    try:
        props = json.loads(raw[4 : 4 + length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, raw
    if not isinstance(props, dict):
        return {}, raw
    return props, raw[4 + length :]
