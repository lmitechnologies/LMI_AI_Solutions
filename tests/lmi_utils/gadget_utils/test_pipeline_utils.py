import pytest
import torch
import numpy as np
import logging
import sys
import os

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
        assert np.array_equal([l,r,t,b], expected_pad)
        assert im2.shape == expected_shape
        assert isinstance(im2, np.ndarray)
        
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
    
    def old_func(self, profile, resolution, offset):
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
            (torch.randint(-32768, 32767, (100,100), dtype=torch.int16), [0.7, 0.96, 0.9], [0.1, 0.1, 0.1]),  # Test 2D
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
        x3,y3,z3,m3 = self.old_func(profile.numpy(), resolution, offset)
        assert np.array_equal(x1, x3)
        assert np.array_equal(y1, y3)
        assert np.array_equal(z1, z3)
        assert np.array_equal(m1, m3)
        