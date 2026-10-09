import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.ops import ResizeMeta, TileMeta
from lmi_utils.preprocess_utils.reconstructor import Reconstructor
from object_detectors.od_core.od_base import ODBase


def test_none_operators_returns_empty():
    assert ODBase._check_operators(None) == []
    assert ODBase._check_operators([]) == []


def test_typed_history_passed_through():
    history = [ResizeMeta(src_sizes=[[1280, 720], [1024, 768]], dst_sizes=[[640, 640]] * 2, pads=[[0, 0, 0, 0]] * 2)]
    assert ODBase._check_operators(history) is history


def test_legacy_dict_shape_rejected():
    """Dict-based history is no longer accepted; must be typed Meta."""
    legacy = [{"type": "resize", "metadata": [{}]}]
    with pytest.raises(ValueError, match="typed Meta records"):
        ODBase._check_operators(legacy)


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


def _results(n):
    return {"boxes": [torch.tensor([[10.0, 20.0, 30.0, 40.0]]) for _ in range(n)], "scores": [torch.ones(1) for _ in range(n)]}


def _maps(n, hw=(32, 32)):
    return [torch.rand(*hw) for _ in range(n)]


ONE_IMAGE_RECORDS = {
    "resize": steps.revert_resize(src_sizes=[[64, 64]], dst_sizes=[[32, 32]], pads=[[0, 0, 0, 0]]),
    "pad": steps.revert_pad(pads=[[1, 1, 1, 1]]),
    "flip": steps.revert_flip(lr=[True], ud=[False], sizes=[[32, 32]]),
    "cropbox": steps.revert_cropbox(boxes=[[4, 4, 36, 36]], orig_sizes=[[64, 64]]),
    "rotate": steps.revert_rotate(angles=[90.0], src_sizes=[[32, 32]], dst_sizes=[[32, 32]]),
}


@pytest.mark.parametrize("name", list(ONE_IMAGE_RECORDS))
def test_one_image_record_for_a_batch_raises(name):
    """A record for one image is not copied to a batch of 4: coords and maps both raise."""
    history = [ONE_IMAGE_RECORDS[name]]
    rec = Reconstructor()
    with pytest.raises(ValueError, match=name):
        rec.reconstruct_coordinates(_results(4), history)
    with pytest.raises(ValueError, match=name):
        rec.reconstruct_images(_maps(4), history)


@pytest.mark.parametrize("name", list(ONE_IMAGE_RECORDS))
def test_one_image_record_for_one_image_works(name):
    history = [ONE_IMAGE_RECORDS[name]]
    rec = Reconstructor()
    assert len(rec.reconstruct_coordinates(_results(1), history)["boxes"]) == 1
    assert len(rec.reconstruct_images(_maps(1), history)) == 1


def test_records_after_a_tile_are_sized_to_the_tiles():
    tile = _tile_meta([[2, 2], [2, 2]])
    per_tile = ResizeMeta(src_sizes=[[320, 320]] * 8, dst_sizes=[[640, 640]] * 8, pads=[[0, 0, 0, 0]] * 8)
    out = Reconstructor().reconstruct_coordinates(_results(8), [tile, per_tile])
    assert len(out["boxes"]) == 2

    one_value = ResizeMeta(src_sizes=[[320, 320]], dst_sizes=[[640, 640]], pads=[[0, 0, 0, 0]])
    with pytest.raises(ValueError, match="resize"):
        Reconstructor().reconstruct_coordinates(_results(8), [tile, one_value])


def test_records_before_a_tile_are_sized_to_the_source_images():
    tile = _tile_meta([[2, 2], [2, 2]])
    pre = ResizeMeta(src_sizes=[[1280, 720], [1024, 768]], dst_sizes=[[640, 640]] * 2, pads=[[0, 0, 0, 0]] * 2)
    assert len(Reconstructor().reconstruct_coordinates(_results(8), [pre, tile])["boxes"]) == 2

    one_value = ResizeMeta(src_sizes=[[1280, 720]], dst_sizes=[[640, 640]], pads=[[0, 0, 0, 0]])
    with pytest.raises(ValueError, match="resize"):
        Reconstructor().reconstruct_coordinates(_results(8), [one_value, tile])


def test_tile_count_mismatch_raises():
    with pytest.raises((ValueError, RuntimeError)):
        Reconstructor().reconstruct_coordinates(_results(3), [_tile_meta([[2, 2]])])


def test_numpy_results_raise_too():
    history = [ONE_IMAGE_RECORDS["resize"]]
    results = {"boxes": [np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)] * 4, "scores": [np.ones(1, dtype=np.float32)] * 4}
    with pytest.raises(ValueError, match="resize"):
        Reconstructor().reconstruct_coordinates(results, history)
