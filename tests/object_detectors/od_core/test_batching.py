import numpy as np
import pytest
import torch

from object_detectors.od_core.od_base import ODBase
from object_detectors.od_core.results import Results


class _Recorder(ODBase):
    """Backend whose forward records each batch size; every image's score is its fill value, to check order."""

    RESIZE_PRESERVE_ASPECT = False

    def __init__(self, max_batch_size=None, fixed_batch_size=None):
        self.max_batch_size = max_batch_size
        self.fixed_batch_size = fixed_batch_size
        self.batches = []

    def warmup(self):
        pass

    def preprocess(self, images):
        return torch.stack([torch.as_tensor(im)[0, 0, 0].float() for im in images])

    def forward(self, batch):
        self.batches.append(len(batch))
        return batch

    def postprocess(self, outputs, **kwargs):
        return [Results(boxes=torch.zeros((1, 4)), scores=v.reshape(1), classes=np.array(["a"])) for v in outputs]


def _images(n):
    return [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]


def _scores(out):
    return [float(s[0]) for s in out["scores"]]


def test_dynamic_engine_chunks_to_its_max_without_padding():
    model = _Recorder(max_batch_size=2)
    out, _ = model.predict(_images(5), configs=0.5)
    assert model.batches == [2, 2, 1]
    assert _scores(out) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("requested, expected", [(4, [2, 2, 1]), (1, [1, 1, 1, 1, 1])])
def test_requested_batch_size_is_capped_at_the_max(requested, expected):
    model = _Recorder(max_batch_size=2)
    model.predict(_images(5), configs=0.5, batch_size=requested)
    assert model.batches == expected


def test_without_a_max_runs_all_at_once():
    model = _Recorder()
    model.predict(_images(5), configs=0.5)
    assert model.batches == [5]


def test_fixed_batch_pads_the_last_chunk():
    model = _Recorder(fixed_batch_size=2)
    out, _ = model.predict(_images(5), configs=0.5)
    assert model.batches == [2, 2, 2]
    assert _scores(out) == [0, 1, 2, 3, 4]
