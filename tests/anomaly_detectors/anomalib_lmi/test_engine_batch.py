import numpy as np
import pytest
import torch

from anomaly_detectors.anomalib_lmi.base import _AnomalibEngine


def _engine_model(tmp_path, max_batch):
    class _Engine:
        def __init__(self, model_path, device):
            self._input_names = ["input"]
            self._output_names = ["anomaly_map"]
            self._output_buffers = [torch.empty(1, 1, 8, 8)]
            self.input_shape = (3, 8, 8)
            self.fp16 = False
            self.is_dynamic = True
            self.max_batch = max_batch

    class _Model(_AnomalibEngine):
        _engine_cls = _Engine

    path = tmp_path / "model.onnx"
    path.touch()
    return _Model(str(path), device="cpu")


def _images(n):
    return [np.zeros((8, 8, 3), dtype=np.uint8)] * n


def test_unbounded_engine_takes_any_batch(tmp_path):
    model = _engine_model(tmp_path, max_batch=None)
    assert model.batch_size is None
    assert model.preprocess(_images(40)).shape == (40, 3, 8, 8)


def test_bounded_engine_rejects_a_larger_batch(tmp_path):
    model = _engine_model(tmp_path, max_batch=4)
    assert model.preprocess(_images(4)).shape == (4, 3, 8, 8)
    with pytest.raises(ValueError, match="exceeds"):
        model.preprocess(_images(5))
