import logging
import os

import pytest
import torch
import torchvision

from lmi_utils.image_utils.tiler import ScaleMode, Tiler, upscale_image
from lmi_utils.system_utils import path_utils

logger = logging.getLogger()


PATH_IMG = "tests/assets/images/dota"


def load_imgs(im_dir, recursive=True):
    im_paths = path_utils.get_relative_paths(im_dir, recursive)
    im_paths.sort()
    imgs = []
    for p in im_paths:
        im = torchvision.io.read_image(os.path.join(im_dir, p)).unsqueeze(0)  # [b,c,h,w]
        imgs.append(im)
    return imgs


@pytest.mark.parametrize(
    ["im", "tile", "stride", "expected_tile_hw", "expected_resized_hw"],
    [
        (
            torch.rand(1, 3, 639, 640),
            [224, 224],
            [224, 224],
            [9, 3, 224, 224],
            [672, 672],
        ),
        (
            torch.rand(1, 3, 640, 640),
            [224, 224],
            [112, 112],
            [25, 3, 224, 224],
            [672, 672],
        ),
        (
            torch.rand(1, 3, 447, 449),
            [224, 224],
            [112, 112],
            [12, 3, 224, 224],
            [448, 560],
        ),
        (  # sides smaller than the tile by a whole number of strides
            torch.rand(1, 3, 112, 56),
            [224, 224],
            [112, 56],
            [1, 3, 224, 224],
            [224, 224],
        ),
        (
            torch.rand(1, 1, 110, 110),
            [224, 112],
            [112, 112],
            [1, 1, 224, 112],
            [224, 112],
        ),
    ],
)
def test_cases(im, tile, stride, expected_tile_hw, expected_resized_hw):
    t = Tiler(tile, stride)
    tiles1 = t.tile(im)
    assert list(tiles1.shape) == expected_tile_hw
    assert list(t.scale_size) == expected_resized_hw
    assert tiles1.dtype == im.dtype

    recon = t.untile(tiles1)
    assert torch.equal(im, recon)

    if torch.cuda.is_available():
        im = im.cuda()
        tiles = t.tile(im)
        assert tiles.dtype == im.dtype
        assert tiles.device == im.device

        recon = t.untile(tiles)
        assert torch.equal(im, recon)
        assert im.device == recon.device


@pytest.mark.parametrize(
    ["im", "tile", "stride"],
    [
        (torch.rand(9, 3, 448, 448), 224, 112),
        (torch.rand(8, 3, 512, 512), 256, 256),
    ],
)
def test_batch(im, tile, stride):
    mode = ScaleMode.INTERPOLATION
    t = Tiler(tile, stride)
    tiles = t.tile(im, mode)
    im2 = t.untile(tiles, mode)
    assert torch.equal(im, im2)


@pytest.mark.parametrize(
    ["im", "tile", "stride"],
    [
        (torch.rand(1, 3, 639, 640), [224, 224], [112, 112]),
        (torch.rand(1, 3, 640, 640), [224, 224], [224, 224]),
        (torch.rand(1, 1, 110, 110), [224, 112], [112, 112]),
    ],
)
def test_tile_boxes_cover_the_tiles_in_order(im, tile, stride):
    tiler = Tiler(tile, stride)
    tiles = tiler.tile(im)
    scaled = upscale_image(im, tiler.scale_size, ScaleMode.PADDING)[0]
    boxes = tiler.tile_boxes()

    assert len(boxes) == len(tiles)
    for t, (x0, y0, x1, y1) in zip(tiles, boxes.int().tolist()):
        assert torch.equal(t, scaled[:, y0:y1, x0:x1])


def test_tile_boxes_needs_a_tiled_or_restored_state():
    with pytest.raises(RuntimeError, match="state incomplete"):
        Tiler([224, 224], [112, 112]).tile_boxes()


@pytest.mark.parametrize("overlap_mode", ["linear", "cosine", "gaussian", "average", "max"])
def test_untile_with_a_one_pixel_overlap_blends_without_nan(overlap_mode):
    # a 1 px overlap leaves a blend region of 0 px, which used to make the blend mask 0/0
    t = Tiler([8, 8], [7, 7])
    im = torch.rand(1, 1, 15, 15)
    out = t.untile(t.tile(im), overlap_mode=overlap_mode)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("overlap_mode", ["linear", "cosine", "gaussian", "average", "max"])
@pytest.mark.parametrize(["tile", "stride"], [(256, 128), (256, 240), (64, 56), (32, 16), (8, 7), (32, 32)])
def test_untile_rebuilds_a_flat_image_at_any_overlap(tile, stride, overlap_mode):
    # regressions this covers: a gaussian of the distance from the tile centre underflowed to 0 over the whole
    # tile at small overlaps; linear and cosine tapered the image border to 0 with no second tile to make up
    # the weight; a 1 px overlap put both tiles on their own zero-weight edge at the seam
    t = Tiler([tile, tile], [stride, stride])
    im = torch.full((1, 1, tile * 2 + 3, tile * 2 + 5), 100.0)
    out = t.untile(t.tile(im), overlap_mode=overlap_mode)
    assert torch.allclose(out, im, atol=1e-3)


@pytest.mark.parametrize("overlap_mode", ["linear", "cosine", "gaussian"])
def test_blended_untile_round_trips_a_random_batch(overlap_mode):
    t = Tiler([32, 32], [24, 24])
    im = torch.rand(2, 3, 101, 77)
    out = t.untile(t.tile(im), overlap_mode=overlap_mode)
    assert torch.allclose(out, im, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8, torch.int16])
def test_interpolation_upscale_is_bilinear_not_nearest(dtype):
    # nearest duplicates whole source pixels, so a smooth ramp comes back with flat steps and the
    # detector sees an aliased image; it was the silent default before a mode was passed
    from lmi_utils.image_utils.tiler import upscale_image

    ramp = torch.linspace(0, 200, 16).view(1, 1, 1, 16).expand(1, 1, 16, 16).contiguous().to(dtype)
    out = upscale_image(ramp, (16, 37), ScaleMode.INTERPOLATION)

    assert out.dtype == dtype
    assert out.shape == (1, 1, 16, 37)
    row = out[0, 0, 0].float()
    assert len(row.unique()) > 16  # bilinear invents in-between values; nearest can only repeat the 16 it had


def test_interpolation_upscale_keeps_a_bool_image_binary():
    # torch resamples floats only, and a bool image is a mask: it must come back as a clean 0/1, not rounded ints
    from lmi_utils.image_utils.tiler import upscale_image

    mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    mask[0, 0, 2:6, 2:6] = True
    out = upscale_image(mask, (16, 16), ScaleMode.INTERPOLATION)

    assert out.dtype == torch.bool
    assert out[0, 0, 5:11, 5:11].all()
    assert not out[0, 0, 0, :].any()
