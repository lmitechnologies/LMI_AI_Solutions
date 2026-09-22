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


@pytest.mark.parametrize("overlap_mode", ["linear", "cosine", "gaussian", "average", "max"])
def test_feature_map_untile_blends_at_the_downscaled_size(overlap_mode):
    # untile a model's feature maps: tiles come back smaller than the image tiles, so positions, stride and
    # canvas rescale by tile_h/tile_size. The neighbour-aware blend must size its mask to the feature tile,
    # not the image tile, or the broadcast would mismatch and border tiles would be tapered to no weight.
    t = Tiler([8, 8], [4, 4])
    t.tile(torch.zeros(1, 1, 16, 16))  # sets the grid; 3x3 tiles over a 16x16 scaled image
    feat = torch.full((t.n_tiles[0] * t.n_tiles[1], 1, 4, 4), 7.0)  # 4x4 feature tiles at half scale
    out = t.untile(feat, overlap_mode=overlap_mode)
    assert out.shape == (1, 1, 8, 8)  # im_size halved with the feature maps
    assert not torch.isnan(out).any()
    assert torch.allclose(out, torch.full_like(out, 7.0), atol=1e-3)


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


def test_untile_non_integral_feature_scale_covers_output():
    image = torch.ones(1, 1, 640, 640)
    tiler = Tiler([224, 224], [112, 112])
    features = torch.nn.functional.interpolate(tiler.tile(image), size=(55, 55), mode="nearest")

    reconstructed = tiler.untile(features)

    assert reconstructed.shape == (1, 1, 157, 157)
    assert torch.isfinite(reconstructed).all()
    assert torch.equal(reconstructed, torch.ones_like(reconstructed))


def test_untile_returns_the_size_it_asked_for():
    # 752 px at 1/32 is 23.5 feature columns, and rounding asked for 24 from a 23 column canvas; the crop
    # in downscale_image then quietly handed back 23, so the returned shape did not match the computed one
    tiler = Tiler([32, 32], [16, 16])
    flat = torch.full((1, 1, 480, 752), 100.0)
    features = torch.nn.functional.interpolate(tiler.tile(flat), size=(1, 1), mode="nearest")

    out = tiler.untile(features)

    assert out.shape == (1, 1, 15, 23)  # floor(480/32), floor(752/32)
    assert torch.equal(out, torch.full_like(out, 100.0))


def test_untile_never_returns_an_empty_image():
    # 97 px at 1/224 floors to 0 rows, and an anomaly map with no rows is not something a caller can use
    tiler = Tiler([224, 224], [112, 112])
    features = torch.nn.functional.interpolate(tiler.tile(torch.ones(1, 1, 97, 131)), size=(1, 1), mode="nearest")

    out = tiler.untile(features)

    assert out.shape == (1, 1, 1, 1)


@pytest.mark.parametrize("feature_hw, expected", [((7, 7), (14, 14)), ((28, 7), (56, 14)), ((55, 55), (110, 110))])
def test_untile_scales_each_axis_on_its_own(feature_hw, expected):
    # a feature map need not keep the tile's aspect ratio; every axis of the output grid is derived separately
    tiler = Tiler([224, 112], [112, 56])
    image = torch.ones(1, 1, 448, 224)
    features = torch.nn.functional.interpolate(tiler.tile(image), size=feature_hw, mode="nearest")

    out = tiler.untile(features)

    assert out.shape == (1, 1, *expected)
    assert torch.equal(out, torch.ones_like(out))


def test_tiler_state_may_hold_tensors():
    # metadata round-trips through torch tensors, and untile rounds with them
    src = Tiler([32, 32], [16, 16])
    src.tile(torch.rand(1, 1, 64, 64))
    as_tensors = {k: ([torch.tensor(x) for x in v] if isinstance(v, (list, tuple)) else v) for k, v in src.to_dict().items()}

    tiler = Tiler.from_dict(as_tensors)

    assert tiler.untile(torch.rand(9, 1, 32, 32)).shape == (1, 1, 64, 64)
    assert tiler.tile_boxes().shape == (9, 4)


def test_untile_survives_tracing():
    # anomalib traces the model to export it, and under tracing a derived tensor's shape is 0-dim tensors,
    # so every dimension untile reads has to be pulled back to an int before any arithmetic

    class Tiled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tiler = Tiler([64, 64], [32, 32])

        def forward(self, x):
            features = torch.nn.functional.avg_pool2d(self.tiler.tile(x), 4)
            return self.tiler.untile(features)

    image = torch.rand(1, 3, 224, 224)
    assert Tiled()(image).shape == (1, 3, 56, 56)
    torch.jit.trace(Tiled(), image, check_trace=False)


def test_expected_scale_rejects_tiles_that_are_not_image_tiles():
    # callers that untile image tiles pass expected_scale=1, so a wrong tile size raises instead of
    # being read as a feature map and rebuilt at that scale
    tiler = Tiler([32, 32], [16, 16])
    tiler.tile(torch.rand(1, 1, 64, 64))

    with pytest.raises(ValueError, match=r"Expected tile size \[32, 32\]"):
        tiler.untile(torch.rand(9, 1, 16, 16), expected_scale=1)

    assert tiler.untile(torch.rand(9, 1, 32, 32), expected_scale=1).shape == (1, 1, 64, 64)


def test_expected_scale_admits_a_known_feature_scale():
    tiler = Tiler([32, 32], [16, 16])
    tiler.tile(torch.rand(1, 1, 64, 64))

    assert tiler.untile(torch.rand(9, 1, 8, 8), expected_scale=0.25).shape == (1, 1, 16, 16)
    with pytest.raises(ValueError, match="at scale 0.25"):
        tiler.untile(torch.rand(9, 1, 4, 4), expected_scale=0.25)


def test_untile_rounds_integer_seams_instead_of_truncating():
    # the blend canvas is float, and a plain .to(uint8) always cut downward, biasing every seam
    t = Tiler([32, 32], [16, 16])
    tiles = t.tile(torch.zeros(1, 1, 64, 64, dtype=torch.uint8)).clone()
    tiles[0], tiles[1] = 201, 202  # the seam they share averages 201.5

    out = t.untile(tiles, overlap_mode="average")

    assert out.dtype == torch.uint8
    assert out[0, 0, 0, 16:32].unique().tolist() == [202]


def test_untile_thresholds_a_bool_mask_at_half():
    # .to(bool) made any non-zero True, so a seam only one tile claimed still came back set
    t = Tiler([32, 32], [16, 16])
    clean = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    clean[0, 0, 8:40, 8:40] = True
    assert torch.equal(t.untile(t.tile(clean)), clean)

    tiles = t.tile(torch.zeros(1, 1, 64, 64, dtype=torch.bool)).clone()
    tiles[0] = True  # one of the two tiles covering the seam

    assert t.untile(tiles, overlap_mode="average")[0, 0, 0, 16:32].unique().tolist() == [False]


def test_untile_max_mode_keeps_negative_values():
    # the canvas was zeroed, so torch.maximum floored every negative value at 0
    t = Tiler([64, 64], [32, 32])
    im = torch.full((1, 1, 128, 128), -5.0)

    out = t.untile(t.tile(im), overlap_mode="max")

    assert torch.equal(out, im)


@pytest.mark.parametrize(["im", "tile", "stride"], [(torch.rand(1, 3, 400, 400), 224, 112), (torch.rand(1, 1, 300, 300), 128, 64)])
def test_interpolation_round_trip_resamples_both_ways(im, tile, stride):
    # upscale went bilinear while downscale kept torch's nearest default, so the reverse dropped whole columns;
    # the existing interpolation tests all use sizes that divide evenly, where neither direction resamples at all
    t = Tiler([tile, tile], [stride, stride])
    assert tuple(t.tile(im, ScaleMode.INTERPOLATION).shape[2:]) == (tile, tile)
    assert tuple(t.scale_size) != tuple(im.shape[2:])  # this case really does resize

    out = t.untile(t.tile(im, ScaleMode.INTERPOLATION), ScaleMode.INTERPOLATION)

    assert out.shape == im.shape
    smooth = torch.linspace(0, 1, im.shape[-1]).expand(im.shape[0], im.shape[1], im.shape[-2], im.shape[-1]).contiguous()
    back = t.untile(t.tile(smooth, ScaleMode.INTERPOLATION), ScaleMode.INTERPOLATION)
    assert torch.allclose(back, smooth, atol=5e-4)  # a nearest reverse lands around 3e-3 on this ramp


def test_tile_boxes_rejects_metadata_that_contradicts_the_grid():
    src = Tiler([32, 32], [16, 16])
    src.tile(torch.rand(1, 1, 64, 64))
    meta = src.to_dict()
    meta["n_tiles"] = [99, 99]

    with pytest.raises(ValueError, match="does not match"):
        Tiler.from_dict(meta).tile_boxes()
