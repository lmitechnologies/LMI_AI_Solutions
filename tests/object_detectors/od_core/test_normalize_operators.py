import pytest

from object_detectors.od_core.od_base import ODBase


def test_none_operators_returns_empty_per_image():
    assert ODBase._normalize_operators(None, 3) == [[], [], []]


def test_broadcast_metadata_length_one():
    """metadata length 1 broadcasts to every image in the batch."""
    history = [{"type": "resize", "metadata": [{"src_size": [1280, 720], "dst_size": [640, 640]}]}]
    result = ODBase._normalize_operators(history, 2)
    assert len(result) == 2
    for chain in result:
        assert chain[0]["type"] == "resize"
        assert chain[0]["metadata"] == [{"src_size": [1280, 720], "dst_size": [640, 640]}]


def test_per_image_metadata_sliced():
    """metadata length B yields each image's own slice."""
    history = [
        {
            "type": "resize",
            "metadata": [
                {"src_size": [1280, 720], "dst_size": [640, 640]},
                {"src_size": [1024, 768], "dst_size": [640, 640]},
            ],
        }
    ]
    result = ODBase._normalize_operators(history, 2)
    assert result[0][0]["metadata"] == [{"src_size": [1280, 720], "dst_size": [640, 640]}]
    assert result[1][0]["metadata"] == [{"src_size": [1024, 768], "dst_size": [640, 640]}]


def test_id_preserved_through_slice():
    history = [{"type": "crop", "id": "UUID_1", "metadata": [{"box": [0, 0, 10, 10], "orig_size": [100, 100]}]}]
    result = ODBase._normalize_operators(history, 1)
    assert result[0][0]["id"] == "UUID_1"


def test_metadata_length_mismatch_raises():
    history = [{"type": "resize", "metadata": [{"src_size": [1280, 720], "dst_size": [640, 640]}] * 3}]
    with pytest.raises(ValueError, match="not 1 .* or 2 "):
        ODBase._normalize_operators(history, 2)


def test_legacy_shape_rejected():
    """Legacy single-key dicts no longer match the unified shape."""
    legacy = [{"resize": [640, 640, 1280, 720]}]
    with pytest.raises(ValueError, match="unified preprocessing history"):
        ODBase._normalize_operators(legacy, 2)
