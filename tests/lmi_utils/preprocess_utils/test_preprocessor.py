import logging

import pytest
import torch
from preprocess_utils.preprocessor import Preprocessor

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@pytest.mark.parametrize(
    "im, wh, par",
    [
        (torch.rand(100, 100, 3).numpy(), [151, 131], True),
        (torch.rand(100, 100, 3).numpy(), [71, 75], False),
    ],
)
def test_preprocessor_resize(im, wh, par):
    preprocessor = Preprocessor()
    processing_steps = [
        {
            "type": "resize",
            "configuration": {"width": wh[0], "height": wh[1], "preserve_aspect": par},
        }
    ]
    im_out, operators = preprocessor.preprocess(im, processing_steps)
    expected_w = wh[0] if wh[0] is not None else im.shape[1]
    expected_h = wh[1] if wh[1] is not None else im.shape[0]
    assert im_out.shape[1] == expected_w
    assert im_out.shape[0] == expected_h
