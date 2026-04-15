import numpy as np
import pytest
import torch

from lmi_utils.image_utils.types import _assert_consistent_types, normalize_image_batch

H, W, C = 64, 64, 3


def _np():
    return np.zeros((H, W, C), dtype=np.uint8)


def _tensor():
    return torch.zeros((H, W, C), dtype=torch.uint8)


class TestNormalizeImageBatch:
    def test_single_numpy(self):
        img = _np()
        out = normalize_image_batch(img)
        assert isinstance(out, list) and len(out) == 1
        assert out[0] is img

    def test_single_tensor(self):
        img = _tensor()
        out = normalize_image_batch(img)
        assert isinstance(out, list) and len(out) == 1
        assert out[0] is img

    def test_list_numpy(self):
        imgs = [_np(), _np()]
        out = normalize_image_batch(imgs)
        assert out is imgs

    def test_list_tensors(self):
        imgs = [_tensor(), _tensor()]
        out = normalize_image_batch(imgs)
        assert out is imgs

    def test_bhwc_numpy(self):
        batch = np.zeros((3, H, W, C), dtype=np.uint8)
        out = normalize_image_batch(batch)
        assert isinstance(out, list) and len(out) == 3
        assert all(item.ndim == 3 for item in out)

    def test_bhwc_tensor(self):
        batch = torch.zeros((3, H, W, C), dtype=torch.uint8)
        out = normalize_image_batch(batch)
        assert isinstance(out, list) and len(out) == 3
        assert all(item.ndim == 3 for item in out)

    def test_empty_list(self):
        assert normalize_image_batch([]) == []

    def test_mixed_list_raises(self):
        with pytest.raises(TypeError, match="mixed types"):
            normalize_image_batch([_np(), _tensor()])


class TestAssertConsistentTypes:
    def test_all_numpy(self):
        _assert_consistent_types([_np(), _np()])  # no raise

    def test_all_tensors(self):
        _assert_consistent_types([_tensor(), _tensor()])  # no raise

    def test_empty(self):
        _assert_consistent_types([])  # no raise

    def test_mixed_raises(self):
        with pytest.raises(TypeError):
            _assert_consistent_types([_np(), _tensor()])
