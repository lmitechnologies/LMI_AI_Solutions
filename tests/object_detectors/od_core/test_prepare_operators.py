import pytest

from lmi_utils.preprocess_utils.ops import ResizeMeta, TileMeta
from object_detectors.od_core.od_base import ODBase


def test_none_operators_returns_empty():
    assert ODBase._prepare_operators(None, 3) == []


def test_broadcast_metadata_length_one():
    """Length-1 Meta broadcasts to every image in the batch."""
    history = [ResizeMeta(src_sizes=[[1280, 720]], dst_sizes=[[640, 640]], pads=[[0, 0, 0, 0]])]
    result = ODBase._prepare_operators(history, 2)
    assert len(result) == 1
    m = result[0]
    assert isinstance(m, ResizeMeta)
    assert m.src_sizes == [[1280, 720], [1280, 720]]
    assert m.dst_sizes == [[640, 640], [640, 640]]
    assert m.pads == [[0, 0, 0, 0], [0, 0, 0, 0]]


def test_per_image_metadata_passed_through():
    """Length-B Meta is used as-is."""
    history = [
        ResizeMeta(
            src_sizes=[[1280, 720], [1024, 768]],
            dst_sizes=[[640, 640], [640, 640]],
            pads=[[0, 0, 0, 0], [0, 0, 0, 0]],
        )
    ]
    result = ODBase._prepare_operators(history, 2)
    assert result[0] is history[0]


def test_metadata_length_mismatch_raises():
    history = [
        ResizeMeta(
            src_sizes=[[1280, 720]] * 3,
            dst_sizes=[[640, 640]] * 3,
            pads=[[0, 0, 0, 0]] * 3,
        )
    ]
    with pytest.raises(ValueError, match="not 1 .*or 2 "):
        ODBase._prepare_operators(history, 2)


def test_legacy_dict_shape_rejected():
    """Dict-based history is no longer accepted; must be typed Meta."""
    legacy = [{"type": "resize", "metadata": [{}]}]
    with pytest.raises(ValueError, match="typed Meta records"):
        ODBase._prepare_operators(legacy, 2)


def _tile_meta(n_tiles):
    n = len(n_tiles)
    return TileMeta(
        tile_sizes=[[320, 320]] * n,
        strides=[[320, 320]] * n,
        im_sizes=[[640, 640]] * n,
        scale_sizes=[[640, 640]] * n,
        n_tiles=n_tiles,
        batch_sizes=[1] * n,
        num_channels=[3] * n,
        scale_modes=["padding"] * n,
        overlap_modes=["average"] * n,
    )


def test_tile_meta_is_sized_by_its_tiles_not_the_batch():
    """A tile entry describes source images, so its length stays at 1 even for a 4-tile batch."""
    history = [_tile_meta([[2, 2]])]
    result = ODBase._prepare_operators(history, 4)
    assert result[0] is history[0]


def test_tile_count_mismatch_raises():
    history = [_tile_meta([[2, 2]])]
    with pytest.raises(ValueError, match="describes 4 tiles, but 3 images"):
        ODBase._prepare_operators(history, 3)


def test_entries_after_a_tile_are_sized_to_the_tiles():
    """A resize applied per tile keeps its own length; the tile entry stays at one per source image."""
    tile = _tile_meta([[2, 2], [2, 2]])
    resize = ResizeMeta(
        src_sizes=[[320, 320]] * 8,
        dst_sizes=[[640, 640]] * 8,
        pads=[[0, 0, 0, 0]] * 8,
    )
    result = ODBase._prepare_operators([tile, resize], 8)
    assert result == [tile, resize]


def test_entries_before_a_tile_are_sized_to_the_source_images():
    """A resize applied to the source images is validated against 2, not the 8 tiles it became."""
    pre = ResizeMeta(src_sizes=[[1280, 720], [1024, 768]], dst_sizes=[[640, 640]] * 2, pads=[[0, 0, 0, 0]] * 2)
    tile = _tile_meta([[2, 2], [2, 2]])
    result = ODBase._prepare_operators([pre, tile], 8)
    assert result == [pre, tile]


def test_broadcast_before_a_tile_uses_the_source_image_count():
    """A length-1 record ahead of a tile step broadcasts to the source images, not to the tiles."""
    pre = ResizeMeta(src_sizes=[[1280, 720]], dst_sizes=[[640, 640]], pads=[[0, 0, 0, 0]])
    tile = _tile_meta([[2, 2], [2, 2]])
    result = ODBase._prepare_operators([pre, tile], 8)
    assert result[0].src_sizes == [[1280, 720], [1280, 720]]
