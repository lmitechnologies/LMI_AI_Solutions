"""Tests for the typed steps namespace.

Verifies that each forward alias constructs the matching Config dataclass with
the expected field values, and that the resulting Configs pass through
Preprocessor.preprocess.
"""

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.ops import (
    CropConfig,
    FlipConfig,
    PadConfig,
    ResizeConfig,
    RotateConfig,
    TileConfig,
)
from lmi_utils.preprocess_utils.preprocessor import Preprocessor


def test_resize_full_kwargs():
    cfg = steps.resize(width=640, height=480, preserve_aspect=True, mode="bilinear")
    assert isinstance(cfg, ResizeConfig)
    assert cfg.width == 640 and cfg.height == 480
    assert cfg.preserve_aspect is True and cfg.mode == "bilinear"


def test_resize_partial_kwargs_defaults():
    cfg = steps.resize(width=640)
    assert cfg.width == 640
    assert cfg.height is None
    assert cfg.preserve_aspect is False
    assert cfg.mode == "bilinear"


def test_resize_with_id():
    assert steps.resize(width=128, height=128, id="r1").id == "r1"


def test_crop():
    cfg = steps.crop(boxes=[[1, 2, 3, 4]])
    assert isinstance(cfg, CropConfig)
    assert cfg.boxes == [[1, 2, 3, 4]]


def test_crop_with_id():
    assert steps.crop(boxes=[[0, 0, 10, 10]], id="c1").id == "c1"


def test_flip_defaults():
    cfg = steps.flip()
    assert isinstance(cfg, FlipConfig)
    assert cfg.lr is False and cfg.ud is False


def test_flip_lr_ud():
    cfg = steps.flip(lr=True, ud=True)
    assert cfg.lr is True and cfg.ud is True


def test_pad_with_wh():
    cfg = steps.pad(width=512, height=512)
    assert isinstance(cfg, PadConfig)
    assert cfg.width == 512 and cfg.height == 512 and cfg.pad is None and cfg.value == 0


def test_pad_with_explicit_pad():
    cfg = steps.pad(pad=[5, 5, 3, 3], value=128)
    assert cfg.pad == [5, 5, 3, 3]
    assert cfg.value == 128


def test_rotate():
    cfg = steps.rotate(angle=30)
    assert isinstance(cfg, RotateConfig)
    assert cfg.angle == 30


def test_rotate_with_id():
    cfg = steps.rotate(angle=-45, id="r")
    assert cfg.angle == -45 and cfg.id == "r"


def test_tile():
    cfg = steps.tile(tile_size=[256, 256], stride=[128, 128])
    assert isinstance(cfg, TileConfig)
    assert cfg.tile_size == [256, 256]
    assert cfg.stride == [128, 128]
    assert cfg.scale_mode == "padding"
    assert cfg.overlap_mode == "average"


def test_tile_with_modes():
    cfg = steps.tile(tile_size=64, stride=32, scale_mode="interpolation", overlap_mode="average", id="t")
    assert cfg.scale_mode == "interpolation" and cfg.id == "t"


def test_id_defaults_to_none():
    assert steps.resize(width=64, height=64).id is None
    assert steps.crop(boxes=[[0, 0, 1, 1]]).id is None
    assert steps.flip().id is None


def test_config_runs_through_preprocessor():
    img = torch.rand(100, 80, 3)
    pipeline = [
        steps.resize(width=64, height=64, preserve_aspect=True),
        steps.flip(lr=True),
    ]
    imgs, history = Preprocessor().preprocess(img, pipeline)
    assert len(imgs) == 1
    assert imgs[0].shape[:2] == (64, 64)
    assert len(history) == 2


def test_invalid_crop_config_raises():
    with pytest.raises(ValueError, match="non-empty list"):
        steps.crop(boxes=[])


def test_invalid_pad_length_raises():
    with pytest.raises(ValueError, match=r"\[L, R, T, B\]"):
        steps.pad(pad=[1, 2, 3])


def test_parse_steps_dict_to_config():
    """JSON manifests are bridged through parse_steps."""
    from lmi_utils.preprocess_utils import parse_steps

    manifest = [
        {"type": "resize", "configuration": {"width": 224, "height": 224, "preserve_aspect": True}},
    ]
    configs = parse_steps(manifest)
    assert isinstance(configs[0], ResizeConfig)
    assert configs[0].width == 224

    img = np.random.rand(80, 60, 3).astype(np.float32)
    a_imgs, _ = Preprocessor().preprocess(img, configs)
    b_imgs, _ = Preprocessor().preprocess(img, [steps.resize(width=224, height=224, preserve_aspect=True)])
    assert np.allclose(a_imgs[0], b_imgs[0])
