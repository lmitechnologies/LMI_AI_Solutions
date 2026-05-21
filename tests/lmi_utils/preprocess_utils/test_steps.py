"""Unit tests for typed step builders in lmi_utils.preprocess_utils.steps.

Lock in the exact dict shape each builder produces so the Python builder path
and the v3 manifest path can never drift, and so an end-to-end run through
Preprocessor accepts the builder output unchanged.
"""

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.preprocessor import Preprocessor


def test_resize_full_kwargs():
    assert steps.resize(width=640, height=480, preserve_aspect=True, mode="bilinear") == {
        "type": "resize",
        "configuration": {"preserve_aspect": True, "mode": "bilinear", "width": 640, "height": 480},
    }


def test_resize_partial_kwargs_strips_none():
    assert steps.resize(width=640) == {
        "type": "resize",
        "configuration": {"preserve_aspect": False, "mode": "bilinear", "width": 640},
    }


def test_resize_with_id():
    out = steps.resize(width=128, height=128, id="r1")
    assert out["id"] == "r1"
    assert out["type"] == "resize"


def test_crop():
    assert steps.crop(boxes=[[1, 2, 3, 4]]) == {
        "type": "crop",
        "configuration": {"boxes": [[1, 2, 3, 4]]},
    }


def test_crop_with_id():
    assert steps.crop(boxes=[[0, 0, 10, 10]], id="c1")["id"] == "c1"


def test_crop_to_label():
    assert steps.crop_to_label(label="BOTTLE", id="ctl") == {
        "type": "crop-to-label",
        "configuration": {"label": "BOTTLE"},
        "id": "ctl",
    }


def test_flip_defaults():
    assert steps.flip() == {
        "type": "flip",
        "configuration": {"lr": False, "ud": False},
    }


def test_flip_lr_ud():
    assert steps.flip(lr=True, ud=True) == {
        "type": "flip",
        "configuration": {"lr": True, "ud": True},
    }


def test_pad_with_wh():
    assert steps.pad(width=512, height=512) == {
        "type": "pad",
        "configuration": {"value": 0, "width": 512, "height": 512},
    }


def test_pad_with_explicit_pad():
    assert steps.pad(pad=[5, 5, 3, 3], value=128) == {
        "type": "pad",
        "configuration": {"value": 128, "pad": [5, 5, 3, 3]},
    }


def test_rotate():
    assert steps.rotate(angle=30) == {
        "type": "rotate",
        "configuration": {"angle": 30.0},
    }


def test_rotate_with_id():
    out = steps.rotate(angle=-45, id="r")
    assert out == {
        "type": "rotate",
        "configuration": {"angle": -45.0},
        "id": "r",
    }


def test_tile():
    assert steps.tile(tile_size=[256, 256], stride=[128, 128]) == {
        "type": "tile",
        "configuration": {
            "tile_size": [256, 256],
            "stride": [128, 128],
            "scale_mode": "padding",
            "overlap_mode": "average",
        },
    }


def test_tile_with_modes():
    out = steps.tile(tile_size=64, stride=32, scale_mode="interpolation", overlap_mode="average", id="t")
    assert out["configuration"]["scale_mode"] == "interpolation"
    assert out["id"] == "t"


def test_all_builders_keyword_only():
    # Positional args must fail — every builder uses `*,` to force keyword usage.
    with pytest.raises(TypeError):
        steps.resize(640, 480)  # type: ignore[misc]
    with pytest.raises(TypeError):
        steps.crop([[0, 0, 10, 10]])  # type: ignore[misc]
    with pytest.raises(TypeError):
        steps.flip(True)  # type: ignore[misc]


def test_id_omitted_when_none():
    # No `id` key when caller did not pass one — matches manifest dicts byte-for-byte.
    assert "id" not in steps.resize(width=64, height=64)
    assert "id" not in steps.crop(boxes=[[0, 0, 1, 1]])
    assert "id" not in steps.flip()


def test_builder_output_runs_through_preprocessor():
    """End-to-end: builders feed into Preprocessor.preprocess unmodified."""
    img = torch.rand(100, 80, 3)
    pipeline = [
        steps.resize(width=64, height=64, preserve_aspect=True),
        steps.flip(lr=True),
    ]
    imgs, history = Preprocessor().preprocess(img, pipeline)
    assert len(imgs) == 1
    assert imgs[0].shape[:2] == (64, 64)
    assert [h["type"] for h in history] == ["resize", "flip"]


def test_builder_matches_manifest_dict():
    """A builder's output must equal a hand-written manifest dict for the same config."""
    builder_dict = steps.resize(width=224, height=224, preserve_aspect=True)
    manifest_dict = {
        "type": "resize",
        "configuration": {"width": 224, "height": 224, "preserve_aspect": True, "mode": "bilinear"},
    }
    img = np.random.rand(80, 60, 3).astype(np.float32)
    a_imgs, a_hist = Preprocessor().preprocess(img, [builder_dict])
    b_imgs, b_hist = Preprocessor().preprocess(img, [manifest_dict])
    assert a_hist == b_hist
    assert np.allclose(a_imgs[0], b_imgs[0])
