import pytest

from lmi_utils.preprocess_utils.ops import ResizeMeta
from object_detectors.od_core.od_base import ODBase


def test_none_operators_returns_empty_per_image():
    assert ODBase._normalize_operators(None, 3) == [[], [], []]


def test_broadcast_metadata_length_one():
    """Length-1 Meta broadcasts to every image in the batch."""
    history = [ResizeMeta(src_sizes=[[1280, 720]], dst_sizes=[[640, 640]], pads=[[0, 0, 0, 0]])]
    result = ODBase._normalize_operators(history, 2)
    assert len(result) == 2
    for chain in result:
        m = chain[0]
        assert isinstance(m, ResizeMeta)
        assert m.src_sizes == [[1280, 720]]
        assert m.dst_sizes == [[640, 640]]


def test_per_image_metadata_sliced():
    """Length-B Meta yields each image's own slice."""
    history = [
        ResizeMeta(
            src_sizes=[[1280, 720], [1024, 768]],
            dst_sizes=[[640, 640], [640, 640]],
            pads=[[0, 0, 0, 0], [0, 0, 0, 0]],
        )
    ]
    result = ODBase._normalize_operators(history, 2)
    assert result[0][0].src_sizes == [[1280, 720]]
    assert result[1][0].src_sizes == [[1024, 768]]


def test_metadata_length_mismatch_raises():
    history = [
        ResizeMeta(
            src_sizes=[[1280, 720]] * 3,
            dst_sizes=[[640, 640]] * 3,
            pads=[[0, 0, 0, 0]] * 3,
        )
    ]
    with pytest.raises(ValueError, match="not 1 .*or 2 "):
        ODBase._normalize_operators(history, 2)


def test_legacy_dict_shape_rejected():
    """Dict-based history is no longer accepted; must be typed Meta."""
    legacy = [{"type": "resize", "metadata": [{}]}]
    with pytest.raises(ValueError, match="typed Meta records"):
        ODBase._normalize_operators(legacy, 2)
