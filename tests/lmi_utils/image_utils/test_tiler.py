import pytest
import os
import logging
import torch
import torchvision

from image_utils.tiler import Tiler, ScaleMode
from system_utils import path_utils

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

PATH_IMG = 'tests/assets/images/dota'


def load_imgs(im_dir, recursive=True):
    im_paths = path_utils.get_relative_paths(im_dir,recursive)
    im_paths.sort()
    imgs = []
    for p in im_paths:
        im = torchvision.io.read_image(os.path.join(im_dir,p)).unsqueeze(0) # [b,c,h,w]
        imgs.append(im)
    return imgs


@pytest.mark.parametrize(
    ['im','tile','stride','expected_tile_hw','expected_resized_hw'],
    [
        (torch.rand(1,3,639,640),[224,224],[224,224],[9,3,224,224],[672,672]),
        (torch.rand(1,3,640,640),[224,224],[112,112],[25,3,224,224],[672,672]),
        (torch.rand(1,3,447,449),[224,224],[112,112],[12,3,224,224],[448,560]),
        (torch.rand(1,1,110,110),[224,112],[112,112],[1,1,224,112],[224,112]),
    ]
)
def test_cases(im,tile,stride,expected_tile_hw,expected_resized_hw):
    t = Tiler(tile,stride)
    tiles1 = t.tile(im)
    assert list(tiles1.shape) == expected_tile_hw
    assert list(t.scale_size) == expected_resized_hw
    assert tiles1.dtype==im.dtype
    
    recon = t.untile(tiles1)
    assert torch.equal(im,recon)
    
    if torch.cuda.is_available():
        im = im.cuda()
        tiles = t.tile(im)
        assert tiles.dtype==im.dtype
        assert tiles.device==im.device
        
        recon = t.untile(tiles)
        assert torch.equal(im,recon)
        assert im.device==recon.device
        

@pytest.mark.parametrize(
    ['im','tile','stride'],
    [
        (torch.rand(9,3,448,448), 224, 112),
        (torch.rand(8,3,512,512), 256, 256),
    ]
)        
def test_batch(im, tile, stride):
    mode = ScaleMode.INTERPOLATION
    t = Tiler(tile, stride)
    tiles = t.tile(im, mode)
    im2 = t.untile(tiles, mode)
    assert torch.equal(im,im2)
    