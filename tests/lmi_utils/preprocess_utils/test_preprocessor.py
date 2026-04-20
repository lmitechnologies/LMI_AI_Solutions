import logging

import numpy as np
import pytest
import torch

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
    processing_steps = [
        {
            "type": "resize",
            "configuration": {"width": wh[0], "height": wh[1], "preserve_aspect": par},
        }
    ]
    ims, operators = preprocessor.preprocess(im, processing_steps)
    im_out = ims[0]
    expected_w = wh[0] if wh[0] is not None else im.shape[1]
    expected_h = wh[1] if wh[1] is not None else im.shape[0]
    assert im_out.shape[1] == expected_w
    assert im_out.shape[0] == expected_h
    assert isinstance(im_out, type(im))


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_preprocessor_4d_batch_input(input_type):
    """Test that a 4D BHWC batch is split into individual images before processing."""
    preprocessor = Preprocessor()
    B, H, W, C = 3, 100, 100, 3
    if input_type == "torch":
        batch = torch.randint(0, 256, (B, H, W, C), dtype=torch.uint8)
    else:
        batch = np.random.randint(0, 256, (B, H, W, C), dtype=np.uint8)

    steps = [{"type": "resize", "configuration": {"width": 64, "height": 64}}]
    ims, history = preprocessor.preprocess(batch, steps)

    assert len(ims) == B
    for im in ims:
        assert im.shape == (64, 64, C)
        assert isinstance(im, type(batch))
