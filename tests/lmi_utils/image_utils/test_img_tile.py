import json
import logging
import os
import subprocess
import tempfile

import pytest
import torch
import torchvision

from lmi_utils.image_utils.img_tile import ScaleMode, to_images, to_tiles
from lmi_utils.system_utils import path_utils

logger = logging.getLogger(__name__)

PATH_IMG = "tests/assets/images/dota8"


def load_imgs(im_dir, recursive=True):
    im_paths = path_utils.get_relative_paths(im_dir, recursive)
    im_paths.sort()
    imgs = []
    for p in im_paths:
        im = torchvision.io.read_image(os.path.join(im_dir, p)).unsqueeze(0)  # [b,c,h,w]
        imgs.append(im)
    return imgs


@pytest.mark.parametrize(
    "tile_hw, stride_hw, expected_tile_hw",
    [(224, 112, [224, 224]), (256, 256, [256, 256]), ([224, 256], [56, 64], [224, 256])],
)
def test_cases(tile_hw, stride_hw, expected_tile_hw):
    imgs = load_imgs(PATH_IMG)
    with tempfile.TemporaryDirectory() as tmpdir:
        to_tiles(PATH_IMG, tmpdir, tile_hw, stride_hw, recursive=True)
        tiles = load_imgs(tmpdir)
        for tile in tiles:
            assert list(tile.shape[-2:]) == expected_tile_hw

        with tempfile.TemporaryDirectory() as tmpdir2:
            to_images(tmpdir, tmpdir2)
            imgs_recon = load_imgs(tmpdir2)
            for im1, im2 in zip(imgs, imgs_recon):
                assert torch.equal(im1, im2)


@pytest.mark.parametrize("tile, stride", [(224, 112), (256, 256), ([224, 256], [56, 64])])
def test_interpolation(tile, stride):
    with tempfile.TemporaryDirectory() as tmp1:
        with tempfile.TemporaryDirectory() as tmp2:
            to_tiles(PATH_IMG, tmp1, tile, stride, mode=ScaleMode.INTERPOLATION)
            to_images(tmp1, tmp2)  # the metadata carries the mode

            imgs = load_imgs(tmp1)
            for im in imgs:
                if isinstance(tile, int):
                    tile = [tile] * 2
                assert list(im.shape[-2:]) == tile


def test_cmds():
    imgs = load_imgs(PATH_IMG)
    my_env = os.environ.copy()
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = f"python -m lmi_utils.image_utils.img_tile --option tile -i {str(PATH_IMG)} -o {str(tmpdir)} --tile 224 224 --stride 112 112"
        out = subprocess.run(cmd, check=True, shell=True, env=my_env, capture_output=True, text=True)
        logger.info(out.stdout)
        logger.info(out.stderr)

        with tempfile.TemporaryDirectory() as tmpdir2:
            cmd = f"python -m lmi_utils.image_utils.img_tile --option untile -i {str(tmpdir)} -o {str(tmpdir2)}"
            out = subprocess.run(cmd, check=True, shell=True, env=my_env, capture_output=True, text=True)
            logger.info(out.stdout)
            logger.info(out.stderr)

            imgs2 = load_imgs(tmpdir2)
            for im1, im2 in zip(imgs, imgs2):
                assert torch.equal(im1, im2)


def test_untile_falls_back_to_the_given_mode_for_metadata_without_one(tmp_path):
    # metadata written before scale_mode was recorded; --resize is the only record of interpolation
    tiles_dir, current, legacy = tmp_path / "tiles", tmp_path / "current", tmp_path / "legacy"
    to_tiles(PATH_IMG, tiles_dir, 224, 112, mode=ScaleMode.INTERPOLATION)
    to_images(tiles_dir, current)
    for p in tiles_dir.glob("*metadata.json"):
        meta = json.loads(p.read_text())
        del meta["scale_mode"]
        p.write_text(json.dumps(meta))

    to_images(tiles_dir, legacy, mode=ScaleMode.INTERPOLATION)

    for im1, im2 in zip(load_imgs(current), load_imgs(legacy), strict=True):
        assert torch.equal(im1, im2)
