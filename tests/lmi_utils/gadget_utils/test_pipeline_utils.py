import pytest
import torch
import numpy as np
import logging
import sys
import os
import cv2

# add path to the repo
PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))
sys.path.append(os.path.join(ROOT, 'lmi_utils'))

import gadget_utils.pipeline_utils as pipeline_utils


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Test_resize_image:
    @pytest.mark.parametrize(
        "im, resize_args, expected_shape, expected_type",
        [
            (torch.rand(150, 100, 3).numpy(), {"H": 300}, (300, 200, 3), np.ndarray),  # Test resize height
            (torch.rand(100, 150, 3).numpy(), {"W": 30}, (20, 30, 3), np.ndarray),    # Test resize width
            (torch.rand(150, 100, 3).numpy(), {"W": 200, "H": 200}, (200, 200, 3), np.ndarray),  # Test warp
            (torch.rand(150, 100).numpy(), {"W": 50}, (75, 50), np.ndarray),  # Test gray image (2D)
            (torch.rand(150, 100, 1).numpy(), {"W": 50}, (75, 50, 1), np.ndarray),  # Test gray image (3D)
        ]
    )
    def test_cases(self,im, resize_args, expected_shape, expected_type):
        im2 = pipeline_utils.resize_image(im, **resize_args)
        assert im2.shape == expected_shape
        assert type(im2) == expected_type
        
    
    def test_tensors(self):
        im  = torch.rand(100, 150, 3)
        im2 = pipeline_utils.resize_image(im, H=10)
        assert im2.shape == (10, 15, 3)
        assert type(im2) == torch.Tensor
        
        if torch.cuda.is_available():
            im  = torch.rand(100, 100, 3).cuda()
            im2 = pipeline_utils.resize_image(im, W=150)
            assert im2.shape == (150, 150, 3)
            assert type(im2) == torch.Tensor
            assert im2.is_cuda

            
class Test_fit_im_to_size:
    def np_func(self, im,W=None,H=None):
        BLACK = (0,0,0)
        h_im,w_im=im.shape[:2]
        if W is None:
            W=w_im
        if H is None:
            H=h_im
        # pad or crop width
        if W >= w_im:
            pad_L=(W-w_im)//2
            pad_R=W-w_im-pad_L
            im=cv2.copyMakeBorder(im,0,0,pad_L,pad_R,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad_L = (w_im-W)//2
            pad_R = w_im-W-pad_L
            im = im[:,pad_L:-pad_R]
            pad_L *= -1
            pad_R *= -1
        # pad or crop height
        if H >= h_im:
            pad_T=(H-h_im)//2
            pad_B=H-h_im-pad_T
            im=cv2.copyMakeBorder(im,pad_T,pad_B,0,0,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad_T = (h_im-H)//2
            pad_B = h_im-H-pad_T
            im = im[pad_T:-pad_B,:]
            pad_T *= -1
            pad_B *= -1
        return im, pad_L, pad_R, pad_T, pad_B
    
    @pytest.mark.parametrize(
        "im, wh, expected_pad, expected_shape",
        [
            (torch.rand(100, 100, 3).numpy(), [121, None], [10, 11, 0, 0], (100, 121, 3)),
            (torch.rand(100, 100, 3).numpy(), [None, 131], [0, 0, 15, 16], (131, 100, 3)),
            (torch.rand(100, 100, 3).numpy(), [151, 131], [25, 26, 15, 16], (131, 151, 3)),
            (torch.rand(100, 100, 3).numpy(), [71, 75], [-14, -15, -12, -13], (75, 71, 3)),
            (torch.rand(100, 100, 3).numpy(), [None, 89], [0, 0, -5, -6], (89, 100, 3)),
            (torch.rand(100, 100, 3).numpy(), [89, None], [-5, -6, 0, 0], (100, 89, 3)),
            (torch.rand(100, 100).numpy(), [89, None], [-5, -6, 0, 0], (100, 89)),
            (torch.rand(100, 100, 1).numpy(), [89, None], [-5, -6, 0, 0], (100, 89, 1)),
        ]
    )
    def test_cases(self, im, wh, expected_pad, expected_shape):
        W,H = wh
        im2, l, r, t, b = pipeline_utils.fit_im_to_size(im, W=W, H=H)
        im3, l3, r3, t3, b3 = self.np_func(im, W, H)
        assert np.array_equal([l,r,t,b], expected_pad)
        assert im2.shape == expected_shape
        assert isinstance(im2, np.ndarray)
        assert np.array_equal(np.squeeze(im2), im3)
        assert l == l3 and r == r3 and t == t3 and b == b3
        
    def test_tensors(self):
        # cpu
        im = torch.rand(100, 100, 3)
        im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, W=89)
        assert l==-5 and r==-6 and t==0 and b==0
        assert im2.shape == (100, 89, 3)
        assert type(im2) == torch.Tensor
        
        if torch.cuda.is_available():
            im = torch.rand(100, 100, 3).cuda()
            im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, W=89)
            assert l==-5 and r==-6 and t==0 and b==0
            assert im2.shape == (100, 89, 3)
            assert type(im2) == torch.Tensor
            assert im2.is_cuda
            
        

class Test_revert_to_origin:        
    @pytest.mark.parametrize(
        "pts, operations, expected",
        [
            (
                [[10, 20], [30, 40], [50, 60], [70, 80]],
                [{'resize': [100, 100, 200, 300]}, {'pad': [8, 9, 10, 11]}, {'stretch': [1.5, 2]}],
                np.array([[-2.66, 0.0], [24.0, 30.0], [50.66, 60.0], [77.34, 90.0]])
            ),
            (
                np.array([[15, 25], [35, 45], [55, 65], [75, 85]]),
                [{'resize': [200, 300, 100, 100]}, {'pad': [8, 9, 10, 11]}, {'stretch': [1.5, 2]}],
                np.array([[1.0, 0.83], [7.67, 4.17], [14.34, 7.5], [21.0, 10.83]])
            ),
            (
                [[15, 25, 35, 45], [55, 65, 75, 85]],
                [{'resize': [200, 300, 100, 100]}, {'pad': [8, 9, 10, 11]}, {'stretch': [1.5, 2]}],
                np.array([[1.0, 0.83, 7.67, 4.17], [14.34, 7.5, 21.0, 10.83]])
            ),
        ]
    )
    def test_cases(self, pts, operations, expected):
        pts2 = pipeline_utils.revert_to_origin(pts, operations)
        assert np.array_equal(pts2, expected.round().clip(0))
        
    def test_types(self):
        pts = np.array([[15, 25, 35, 45], [55, 65, 75, 85]])
        operations = [{'resize':[200,300,100,100]},{'pad':[8,9,10,11]},{'stretch':[1.5,2]}]
        pts2 = pipeline_utils.revert_to_origin(pts, operations)
        assert type(pts2) == np.ndarray
        
        pts2 = pipeline_utils.revert_to_origin(pts.tolist(), operations)
        assert type(pts2) == list
        
        pts2 = pipeline_utils.revert_to_origin(torch.tensor(pts), operations)
        assert type(pts2) == torch.Tensor
        
        pts2 = pipeline_utils.revert_to_origin(torch.tensor(pts).cuda(), operations)
        assert type(pts2) == torch.Tensor
        assert pts2.is_cuda
        
        
class Test_profile_to_3d:
    def np_func(self, profile, resolution, offset):
        if profile.dtype != np.int16:
            raise Exception(f'profile.dtype should be int16, got {profile.dtype}')
        TWO_TO_FIFTEEN = 2 ** 15
        h,w = profile.shape[:2]
        x1,y1 = 0,0
        x2,y2 = w,h
        mask = profile != -TWO_TO_FIFTEEN
        xx,yy = np.meshgrid(np.arange(x1,x2), np.arange(y1,y2))
        X = offset[0] + xx * resolution[0]
        Y = offset[1] + yy * resolution[1]
        Z = offset[2] + profile*resolution[2]
        return X,Y,Z,mask
    
    @pytest.mark.parametrize(
        "profile, resolution, offset",
        [
            (torch.randint(-32768, 32767, (100,100), dtype=torch.int16), [0.7, 0.96, 0.9], [0.1, 0.1, 0.1]),
            (torch.randint(-32768, 32767, (90,100), dtype=torch.int16), [1, 1, 1], [0, 0, 0]),
        ]
    )
    def test_cases(self, profile, resolution, offset):
        x1,y1,z1,m1 = pipeline_utils.profile_to_3d(profile.numpy(), resolution, offset)
        x2,y2,z2,m2 = pipeline_utils.profile_to_3d(profile, resolution, offset)
        x2,y2,z2,m2 = x2.numpy(), y2.numpy(), z2.numpy(), m2.numpy()
        assert np.allclose(x1, x2)
        assert np.allclose(y1, y2)
        assert np.allclose(z1, z2)
        assert np.allclose(m1, m2)
        x3,y3,z3,m3 = self.np_func(profile.numpy(), resolution, offset)
        assert np.array_equal(x1, x3)
        assert np.array_equal(y1, y3)
        assert np.array_equal(z1, z3)
        assert np.array_equal(m1, m3)
        
        
    @pytest.mark.parametrize(
        "profile, resolution, offset",
        [
            (torch.randint(-32768, 32767, (100,100), dtype=torch.int16), [0.7, 0.96, 0.9], [0.1, 0.1, 0.1]),
            (torch.randint(-32768, 32767, (90,100), dtype=torch.int16), [1, 1, 1], [0, 0, 0]),
        ]
    )
    def test_types(self, profile, resolution, offset):
        if torch.cuda.is_available():
            profile = profile.cuda()
            x,y,z,m = pipeline_utils.profile_to_3d(profile, resolution, offset)
            assert x.is_cuda
            assert y.is_cuda
            assert z.is_cuda
            assert m.is_cuda
        
        with pytest.raises(Exception) as info:
            x,y,z,m = pipeline_utils.profile_to_3d(profile.to(torch.uint16), resolution, offset)
        logger.debug(info.value)
        
        
class Test_uint16_to_int16:
    def np_func(self, profile):
        if profile.dtype != np.uint16:
            raise Exception(f'dtype should be uint16, got {profile.dtype}')
        TWO_TO_FIFTEEN = 2 ** 15
        return profile.view(np.int16) + np.int16(-TWO_TO_FIFTEEN)
    
    
    @pytest.mark.parametrize(
        "profile",
        [
            (torch.randint(0, 65535, (100,100), dtype=torch.uint16)),
            (torch.randint(0, 65535, (220,210), dtype=torch.uint16)),
        ]
    )
    def test_cases(self, profile):
        p1 = pipeline_utils.uint16_to_int16(profile)
        p2 = self.np_func(profile.numpy())
        p1 = p1.numpy()
        assert np.array_equal(p1, p2)
        
        if torch.cuda.is_available():
            profile = profile.cuda()
            p3 = pipeline_utils.uint16_to_int16(profile)
            assert p3.is_cuda
            assert np.array_equal(p3.cpu().numpy(), p2)
        
        with pytest.raises(Exception) as info:
            pipeline_utils.uint16_to_int16(profile.to(torch.int16))
        logger.debug(info.value)
        
        
class Test_pts_to_3d:
    def np_func(self, pts, profile, resolution, offset):
        if profile.dtype != np.int16:
            raise Exception(f'profile.dtype should be int16, got {profile.dtype}')
        xyz = []
        for pt in pts:
            if len(pt)!=2:
                raise Exception(f'pts should be a list of (x,y) points, got {pt}')
            x,y = map(int,pt)
            nx = offset[0] + x * resolution[0]
            ny = offset[1] + y * resolution[1]
            nz = offset[2] + profile[y][x]*resolution[2]
            xyz += [[nx,ny,nz]]
        return np.array(xyz)
    
    @pytest.mark.parametrize(
        "pts, profile, resolution, offset",
        [
            (np.array([[10, 20], [30, 40], [50, 60], [70, 80]]), torch.randint(-32768, 32767, (100,100), dtype=torch.int16).numpy(), [0.7, 0.96, 0.9], [0.1, 0.1, 0.1]),
            (torch.tensor([[15, 25], [35, 45], [55, 65], [75, 85]]), torch.randint(-32768, 32767, (90,100), dtype=torch.int16), [1, 1, 1], [0, 0, 0]),
        ]
    )
    def test_cases(self, pts, profile, resolution, offset):
        xyz1 = pipeline_utils.pts_to_3d(pts, profile, resolution, offset)
        if isinstance(pts, torch.Tensor):
            pts = pts.numpy()
            profile = profile.numpy()
        xyz2 = self.np_func(pts, profile, resolution, offset)
        assert np.array_equal(xyz1, xyz2)
        
        if torch.cuda.is_available():
            if not isinstance(pts, torch.Tensor):
                pts = torch.from_numpy(pts).cuda()
                profile = torch.from_numpy(profile).cuda()
            xyz3 = pipeline_utils.pts_to_3d(pts, profile, resolution, offset)
            assert xyz3.is_cuda
            assert np.array_equal(xyz3.cpu().numpy(), xyz2)
            
    def test_types(self):
        pts = np.array([[15, 25], [35, 45], [55, 65], [75, 85]])
        profile = torch.randint(-32768, 32767, (90,100), dtype=torch.int16)
        resolution = [1, 1, 1]
        offset = [0, 0, 0]
        with pytest.raises(Exception) as info:
            pipeline_utils.pts_to_3d(pts, profile.to(torch.uint16), resolution, offset)
        logger.debug(info.value)
        
        pts2 = np.expand_dims(pts, axis=0)
        with pytest.raises(Exception) as info:
            pipeline_utils.pts_to_3d(pts2, profile.numpy(), resolution, offset)
        logger.debug(info.value)
            
        