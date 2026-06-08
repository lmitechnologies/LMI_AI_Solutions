import numpy as np
import pytest
import torch

from lmi_utils.image_utils.img_resize import resize_and_pad
from lmi_utils.preprocess_utils.ops import ResizeMeta


@pytest.mark.parametrize(
    "input_shape, target_w, target_h",
    [
        ((100, 100, 3), 200, 200),
        ((100, 100, 3), 50, 50),
        ((100, 100, 3), 200, 50),
        ((50, 100, 3), 50, 50),
        ((10, 10, 3), 1000, 1000),
    ],
)
def test_resize_and_pad(input_shape, target_w, target_h):
    input_image = np.zeros(input_shape, dtype=np.uint8)
    h, w = input_shape[:2]

    output, ops = resize_and_pad(input_image, width=target_w, height=target_h, preserve_aspect=False, return_operators=True)

    assert output.shape == (target_h, target_w, 3)
    assert len(ops) == 1
    meta = ops[0]
    assert isinstance(meta, ResizeMeta)
    assert meta.src_sizes[0] == [w, h]
    assert meta.dst_sizes[0] == [target_w, target_h]
    assert meta.pads[0] == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "input_shape, target_w, target_h, expected_resize, expected_pad",
    [
        ((100, 100, 3), 200, 200, [200, 200], [0, 0]),
        ((100, 50, 3), 100, 100, [50, 100], [50, 0]),
        ((50, 100, 3), 100, 100, [100, 50], [0, 50]),
        ((100, 200, 3), 50, 50, [50, 25], [0, 25]),
    ],
)
def test_resize_and_pad_preserve_aspect(input_shape, target_w, target_h, expected_resize, expected_pad):
    input_image = np.zeros(input_shape, dtype=np.uint8)
    h, w = input_shape[:2]

    output, ops = resize_and_pad(input_image, width=target_w, height=target_h, preserve_aspect=True, return_operators=True)

    assert output.shape == (target_h, target_w, 3)
    assert len(ops) == 1
    meta = ops[0]
    assert isinstance(meta, ResizeMeta)
    assert meta.src_sizes[0] == [w, h]
    assert meta.dst_sizes[0] == [expected_resize[0], expected_resize[1]]

    exp_pad_w, exp_pad_h = expected_pad
    pad_l, pad_r, pad_t, pad_b = meta.pads[0]
    assert pad_l + pad_r == exp_pad_w
    assert pad_t + pad_b == exp_pad_h


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_resize_and_pad_polymorphism(input_type):
    target_w, target_h = 100, 100

    if input_type == "numpy":
        input_image = np.zeros((50, 50), dtype=np.uint8)
    else:
        input_image = torch.zeros((50, 50), dtype=torch.uint8)

    output, ops = resize_and_pad(input_image, width=target_w, height=target_h, preserve_aspect=True, return_operators=True)

    if input_type == "numpy":
        assert isinstance(output, np.ndarray), "Input was Numpy, expected Numpy output"
    else:
        assert isinstance(output, torch.Tensor), "Input was Tensor, expected Tensor output"

    assert output.shape == (100, 100)

    meta = ops[0]
    assert isinstance(meta, ResizeMeta)
    assert meta.src_sizes[0] == [50, 50]
    assert meta.dst_sizes[0] == [100, 100]
    # 50x50 → 100x100 preserve aspect: no padding (already square).
    assert meta.pads[0] == [0, 0, 0, 0]
