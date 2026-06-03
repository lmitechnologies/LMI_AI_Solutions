import logging

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.preprocessor import Preprocessor

logger = logging.getLogger()


@pytest.mark.parametrize(
    "im, wh, par",
    [
        (torch.rand(100, 100, 3).numpy(), [151, 131], True),
        (torch.rand(100, 100, 3), [71, 75], False),
    ],
)
def test_preprocessor_resize(im, wh, par):
    preprocessor = Preprocessor()
    ims, _ = preprocessor.preprocess(im, [steps.resize(width=wh[0], height=wh[1], preserve_aspect=par)])
    im_out = ims[0]
    assert im_out.shape[1] == wh[0]
    assert im_out.shape[0] == wh[1]
    assert isinstance(im_out, type(im))


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_preprocessor_4d_batch_input(input_type):
    preprocessor = Preprocessor()
    B, H, W, C = 3, 100, 100, 3
    if input_type == "torch":
        batch = torch.randint(0, 256, (B, H, W, C), dtype=torch.uint8)
    else:
        batch = np.random.randint(0, 256, (B, H, W, C), dtype=np.uint8)

    ims, _ = preprocessor.preprocess(batch, [steps.resize(width=64, height=64)])

    assert len(ims) == B
    for im in ims:
        assert im.shape == (64, 64, C)
        assert isinstance(im, type(batch))
