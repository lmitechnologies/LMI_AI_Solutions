"""Embed and read model metadata (class names, ...) so a deployed model is a single file.

An ONNX file carries the payload in its ``metadata_props``. A serialized TensorRT plan has no
metadata slot, so the engine file is prefixed with a length-tagged JSON header — the same layout
ultralytics uses, which keeps their engines readable here.

Metadata is always optional: readers return ``{}`` for a file that carries none, and writers skip
an empty payload, so plain ONNX files and headerless engines keep working unchanged.
"""

import json
import logging
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

ONNX_METADATA_KEY = "lmi_metadata"

_PLAN_MAGIC = b"ftrt"  # first bytes of a headerless serialized TensorRT plan
_MAX_HEADER_BYTES = 1 << 20


def embed_onnx_metadata(onnx_path: str, metadata: Mapping[str, Any]) -> None:
    """Write ``metadata`` into an ONNX file's metadata_props, in place. Requires the ``onnx`` package."""
    import onnx

    model = onnx.load(onnx_path)
    prop = next((p for p in model.metadata_props if p.key == ONNX_METADATA_KEY), None)
    if prop is None:
        prop = model.metadata_props.add()
        prop.key = ONNX_METADATA_KEY
    prop.value = json.dumps(dict(metadata))
    onnx.save(model, onnx_path)
    logger.info(f"Embedded metadata {sorted(metadata)} in {onnx_path}")


def read_onnx_metadata(onnx_path: str) -> Dict[str, Any]:
    """Read embedded metadata from an ONNX file ({} when it carries none)."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    return metadata_from_onnx_props({p.key: p.value for p in model.metadata_props})


def metadata_from_onnx_props(props: Optional[Mapping[str, str]]) -> Dict[str, Any]:
    """Decode the payload out of an ONNX metadata map, e.g. onnxruntime's ``custom_metadata_map``."""
    raw = (props or {}).get(ONNX_METADATA_KEY)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Ignoring unreadable '{ONNX_METADATA_KEY}' ONNX metadata")
        return {}
    return value if isinstance(value, dict) else {}


def engine_metadata_header(metadata: Mapping[str, Any]) -> bytes:
    """Serialize ``metadata`` as the length-tagged JSON header that prefixes an engine plan."""
    payload = json.dumps(dict(metadata)).encode("utf-8")
    return len(payload).to_bytes(4, byteorder="little", signed=True) + payload


def split_engine_metadata(raw: bytes) -> Tuple[Dict[str, Any], bytes]:
    """Split engine-file bytes into (metadata, serialized plan).

    A plan carrying no header is returned untouched with empty metadata. TensorRT rejects a plan with
    leading bytes, so the header must be stripped before deserializing.
    """
    if raw[:4] == _PLAN_MAGIC:
        return {}, raw
    length = int.from_bytes(raw[:4], byteorder="little", signed=True)
    if not 0 < length <= min(_MAX_HEADER_BYTES, len(raw) - 4):
        return {}, raw
    try:
        value = json.loads(raw[4 : 4 + length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, raw
    if not isinstance(value, dict):
        return {}, raw
    return value, raw[4 + length :]
