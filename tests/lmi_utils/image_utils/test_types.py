import numpy as np
import pytest
import torch

from lmi_utils.image_utils.types import _assert_consistent_types, assert_hwc, assert_image_like, assert_uint8, normalize_image_batch

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

    def test_invalid_single_raises(self):
        with pytest.raises(TypeError, match="np.ndarray or torch.Tensor"):
            normalize_image_batch("not_an_image")

    def test_invalid_in_list_raises(self):
        with pytest.raises(TypeError, match="np.ndarray or torch.Tensor"):
            normalize_image_batch([_np(), "not_an_image"])

    def test_non_uint8_numpy_raises(self):
        with pytest.raises(ValueError, match="uint8"):
            normalize_image_batch(np.zeros((H, W, C), dtype=np.float32))

    def test_non_uint8_tensor_raises(self):
        with pytest.raises(ValueError, match="uint8"):
            normalize_image_batch(torch.zeros((H, W, C), dtype=torch.float32))

    def test_non_uint8_in_list_raises(self):
        with pytest.raises(ValueError, match="uint8"):
            normalize_image_batch([np.zeros((H, W, C), dtype=np.float32)])

    def test_non_3d_single_raises(self):
        with pytest.raises(ValueError, match="3-dimensional"):
            normalize_image_batch(np.zeros((H, W), dtype=np.uint8))

    def test_non_3d_in_list_raises(self):
        with pytest.raises(ValueError, match="3-dimensional"):
            normalize_image_batch([np.zeros((H, W), dtype=np.uint8)])


class TestAssertHwc:
    def test_3d_passes(self):
        assert_hwc(_np())  # no raise

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="3-dimensional"):
            assert_hwc(np.zeros((H, W), dtype=np.uint8))

    def test_4d_raises(self):
        with pytest.raises(ValueError, match="3-dimensional"):
            assert_hwc(np.zeros((2, H, W, C), dtype=np.uint8))


class TestAssertUint8:
    def test_numpy_uint8_passes(self):
        assert_uint8(_np())  # no raise

    def test_tensor_uint8_passes(self):
        assert_uint8(_tensor())  # no raise

    def test_numpy_float_raises(self):
        with pytest.raises(ValueError, match="uint8"):
            assert_uint8(np.zeros((H, W, C), dtype=np.float32))

    def test_tensor_float_raises(self):
        with pytest.raises(ValueError, match="uint8"):
            assert_uint8(torch.zeros((H, W, C), dtype=torch.float32))


class TestAssertImageLike:
    def test_numpy_passes(self):
        assert_image_like(_np())  # no raise

    def test_tensor_passes(self):
        assert_image_like(_tensor())  # no raise

    def test_string_raises(self):
        with pytest.raises(TypeError, match="np.ndarray or torch.Tensor"):
            assert_image_like("not_an_image")

    def test_int_raises(self):
        with pytest.raises(TypeError, match="np.ndarray or torch.Tensor"):
            assert_image_like(42)

    def test_none_raises(self):
        with pytest.raises(TypeError, match="np.ndarray or torch.Tensor"):
            assert_image_like(None)


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
