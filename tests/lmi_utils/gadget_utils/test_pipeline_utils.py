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
        "im, resize_args, expected_shape",
        [
            (torch.randint(0,256,(150, 100, 3),dtype=torch.float32).numpy(), {"H": 300}, (300, 200, 3)),  # Test resize height
            (torch.randint(0,256,(100, 150, 3),dtype=torch.uint8).numpy(), {"W": 30}, (20, 30, 3)),    # Test resize width
            (torch.randint(0,256,(150, 100, 3),dtype=torch.int16).numpy(), {"W": 200, "H": 200}, (200, 200, 3)),  # Test warp
            (torch.rand(150, 100).numpy(), {"W": 50}, (75, 50)),  # Test gray image (2D)
            (torch.rand(150, 100, 1).numpy(), {"W": 50}, (75, 50, 1)),  # Test gray image (3D)
        ]
    )
    def test_cases(self,im, resize_args, expected_shape):
        im2 = pipeline_utils.resize_image(im, **resize_args)
        assert im2.shape == expected_shape
        assert im2.dtype == im.dtype
        
        im2 = pipeline_utils.resize_image(torch.from_numpy(im), **resize_args)
        assert im2.shape == expected_shape
        assert type(im2) == torch.Tensor
        
        if torch.cuda.is_available():
            tmp = torch.from_numpy(im).cuda()
            im2 = pipeline_utils.resize_image(tmp, **resize_args)
            assert im2.shape == expected_shape
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
        "im, wh",
        [
            (torch.rand(100, 100, 3).numpy(), [121, None]),
            (torch.rand(100, 100, 3).numpy(), [None, 131]),
            (torch.rand(100, 100, 3).numpy(), [151, 131]),
            (torch.rand(100, 100, 3).numpy(), [71, 75]),
            (torch.rand(100, 100, 3).numpy(), [None, 89]),
            (torch.rand(100, 100, 3).numpy(), [89, None]),
            (torch.rand(100, 100).numpy(), [89, None]),
            (torch.rand(100, 100, 1).numpy(), [89, None]),
        ]
    )
    def test_cases(self, im, wh):
        W,H = wh
        im1, l1, r1, t1, b1 = self.np_func(im, W, H)
        
        im2, l2, r2, t2, b2 = pipeline_utils.fit_im_to_size(im, W=W, H=H)
        assert isinstance(im2, np.ndarray)
        assert np.array_equal(im1, np.squeeze(im2))
        assert l1 == l2 and r1 == r2 and t1 == t2 and b1 == b2
        
        im2, l2, r2, t2, b2 = pipeline_utils.fit_im_to_size(torch.from_numpy(im), W=W, H=H)
        assert isinstance(im2, torch.Tensor)
        assert np.array_equal(im1, np.squeeze(im2.numpy()))
        assert l1 == l2 and r1 == r2 and t1 == t2 and b1 == b2
        
        if torch.cuda.is_available():
            im = torch.from_numpy(im).cuda()
            im2, l2, r2, t2, b2 = pipeline_utils.fit_im_to_size(im, W=W, H=H)
            assert im2.is_cuda
            assert np.array_equal(im1, np.squeeze(im2.cpu().numpy()))
            assert l1 == l2 and r1 == r2 and t1 == t2 and b1 == b2
            
        
class Test_revert_to_origin:
    def np_func(self, pts:np.ndarray, operations:list, verbose=False):
        def revert(x,y, operations):
            nx,ny = x,y
            for operator in reversed(operations):
                if 'resize' in operator:
                    tw,th,orig_w,orig_h = operator['resize']
                    r = [tw/orig_w,th/orig_h]
                    nx,ny = nx/r[0], ny/r[1]
                if 'pad' in operator:
                    pad_L,pad_R,pad_T,pad_B = operator['pad']
                    nx,ny = nx-pad_L,ny-pad_T
                if 'stretch' in operator:
                    s = operator['stretch']
                    nx,ny = nx/s[0], ny/s[1]
                if verbose:
                    logger.info(f'after {operator}, pt: {x:.2f},{y:.2f} -> {nx:.2f},{ny:.2f}')
            nx = round(nx)
            ny = round(ny)
            return [max(nx,0),max(ny,0)]

        pts2 = []
        if isinstance(pts, list):
            pts = np.array(pts)
        for pt in pts:
            if len(pt)==0:
                continue
            if len(pt)==2:
                x,y = pt
                pts2.append(revert(x,y,operations))
            elif len(pt)==4:
                x1,y1,x2,y2 = pt
                pts2.append(revert(x1,y1,operations)+revert(x2,y2,operations))
            else:
                raise Exception(f'does not support pts neither Nx2 nor Nx4. Got shape: {pt.shape} with val: {pt}')
        return pts2
    
    @pytest.mark.parametrize(
        "pts, operations",
        [
            (
                [[10.1, 20.0], [30.2, 40.4], [50.5, 60.4], [70.2, 80.1]],
                [{'resize': [100, 100, 200, 300]}, {'pad': [8, 9, 10, 11]}, {'stretch': [1.5, 2]}],
            ),
            (
                np.array([[15.3, 25.8], [35.7, 45], [55.6, 65], [75, 85.3]]),
                [{'resize': [200, 300, 100, 100]}, {'pad': [-8, -9, 10, 11]}, {'stretch': [1.3, 1.5]}],
            ),
            (
                [[15, 25, 35, 45], [55, 65, 75, 85]],
                [{'resize': [200, 300, 100, 100]}, {'pad': [8, 9, 10, 11]}, {'stretch': [1.5, 2]}],
            ),
        ]
    )
    def test_cases(self, pts, operations):
        pts1 = self.np_func(pts, operations)
        
        pts2 = pipeline_utils.revert_to_origin(pts, operations)
        assert np.array_equal(pts1, pts2)
        
        if not isinstance(pts, np.ndarray):
            pts = np.array(pts)
        pts2 = pipeline_utils.revert_to_origin(torch.from_numpy(pts), operations)
        assert np.array_equal(pts1, pts2.numpy())
        
        if torch.cuda.is_available():
            pts = torch.tensor(pts).cuda()
            pts2 = pipeline_utils.revert_to_origin(pts, operations)
            assert pts2.is_cuda
            assert np.array_equal(pts1, pts2.cpu().numpy())
        
        
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
            (torch.randint(-32768, 32767, (100,100), dtype=torch.int16).numpy(), [0.7, 0.96, 0.9], [0.1, 0.1, 0.1]),
            (torch.randint(-32768, 32767, (90,100), dtype=torch.int16).numpy(), [1, 1, 1], [0, 0, 0]),
        ]
    )
    def test_cases(self, profile, resolution, offset):
        x1,y1,z1,m1 = self.np_func(profile, resolution, offset)
        x2,y2,z2,m2 = pipeline_utils.profile_to_3d(profile, resolution, offset)
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
        assert np.array_equal(z1, z2)
        assert np.array_equal(m1, m2)
        
        x3,y3,z3,m3 = pipeline_utils.profile_to_3d(torch.from_numpy(profile), resolution, offset)
        assert np.array_equal(x1, x3.numpy())
        assert np.array_equal(y1, y3.numpy())
        assert np.array_equal(z1, z3.numpy())
        assert np.array_equal(m1, m3.numpy())
        
        profile = torch.from_numpy(profile)
        if torch.cuda.is_available():
            x4,y4,z4,m4 = pipeline_utils.profile_to_3d(profile.cuda(), resolution, offset)
            assert x4.is_cuda and y4.is_cuda and z4.is_cuda and m4.is_cuda
            assert np.array_equal(x1, x4.cpu().numpy())
            assert np.array_equal(y1, y4.cpu().numpy())
            assert np.array_equal(z1, z4.cpu().numpy())
            assert np.array_equal(m1, m4.cpu().numpy())
            
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
            (torch.randint(0, 65535, (500,500), dtype=torch.uint16).numpy()),
            (torch.randint(0, 65535, (520,530), dtype=torch.uint16).numpy()),
        ]
    )
    def test_cases(self, profile):
        p1 = self.np_func(profile)
        p2 = pipeline_utils.uint16_to_int16(profile)
        assert np.array_equal(p1, p2)
        
        p2 = pipeline_utils.uint16_to_int16(torch.from_numpy(profile))
        assert np.array_equal(p1, p2.numpy())
        
        profile = torch.from_numpy(profile)
        if torch.cuda.is_available():
            p2 = pipeline_utils.uint16_to_int16(profile.cuda())
            assert p2.is_cuda
            assert np.array_equal(p1, p2.cpu().numpy())
        
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
            (np.array([[10.0, 20.3], [30.4, 40.5], [50.1, 60], [70, 80]]), torch.randint(-32768, 32767, (100,100), dtype=torch.int16).numpy(), [0.7, 0.96, 0.9], [0.1, 0.1, 0.1]),
            (np.array([[15, 25], [35, 45], [55, 65], [75, 85]]), torch.randint(-32768, 32767, (90,100), dtype=torch.int16).numpy(), [1, 1, 1], [0, 0, 0]),
        ]
    )
    def test_cases(self, pts, profile, resolution, offset):
        xyz1 = self.np_func(pts, profile, resolution, offset)
        
        xyz2 = pipeline_utils.pts_to_3d(pts, profile, resolution, offset)
        assert np.array_equal(xyz1, xyz2)
        
        xyz2 = pipeline_utils.pts_to_3d(torch.from_numpy(pts), torch.from_numpy(profile), resolution, offset)
        assert np.array_equal(xyz1, xyz2.numpy())
        
        if torch.cuda.is_available():
            if not isinstance(pts, torch.Tensor):
                pts = torch.from_numpy(pts).cuda()
                profile = torch.from_numpy(profile).cuda()
            xyz3 = pipeline_utils.pts_to_3d(pts, profile, resolution, offset)
            assert xyz3.is_cuda
            assert np.array_equal(xyz1,xyz3.cpu().numpy())
            
    def test_error_handle(self):
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
            

class Test_apply_operations:
    def np_func(self, pts:np.ndarray, operations:list):
        def apply(x,y, operations):
            nx,ny = x,y
            for operator in operations:
                if 'resize' in operator:
                    tw,th,orig_w,orig_h = operator['resize']
                    r = [tw/orig_w,th/orig_h]
                    nx,ny = nx*r[0], ny*r[1]
                if 'pad' in operator:
                    pad_L,pad_R,pad_T,pad_B = operator['pad']
                    nx,ny = nx+pad_L,ny+pad_T
                if 'stretch' in operator:
                    s = operator['stretch']
                    nx,ny = nx*s[0], ny*s[1]
                if 'flip' in operator:
                    lr,ud,im_w,im_h = operator['flip']
                    if lr:
                        nx = im_w-nx
                    if ud:
                        ny = im_h-ny
            nx,ny = round(nx),round(ny)
            return [max(nx,0),max(ny,0)]

        pts2 = []
        if isinstance(pts, list):
            pts = np.array(pts)
        for pt in pts:
            if len(pt)==0:
                continue
            if len(pt)==2:
                x,y = pt
                pts2.append(apply(x,y,operations))
            elif len(pt)==4:
                x1,y1,x2,y2 = pt
                pts2.append(apply(x1,y1,operations)+apply(x2,y2,operations))
            else:
                raise Exception(f'does not support pts neither Nx2 nor Nx4. Got shape: {pt.shape} with val: {pt}')
        return pts2
    
    
    @pytest.mark.parametrize(
        "pts, operations",
        [
            ([[10, 20], [30, 40], [50, 60], [70, 80]], [{'resize': [100, 100, 200, 300]}, {'pad': [8, 9, 10, 11]}, {'stretch': [1.5, 2]}, {'flip': [True, False, 100, 100]}]),
            (np.array([[15.0, 25.2], [35.1, 45.0], [55.0, 65], [75.5, 85]]), [{'resize': [200, 300, 100, 100]}, {'pad': [6, 7, 9, 10]}, {'stretch': [1.0, 2]}]),
            ([[15, 25, 35, 45], [55, 65, 75, 85]], [{'resize': [200, 300, 100, 100]}, {'pad': [11,9,-2,-3]}, {'flip': [False, True, 200, 300]}]),
        ]
    )
    def test_cases(self, pts, operations):
        pts1 = self.np_func(pts, operations)
        pts2 = pipeline_utils.apply_operations(pts, operations)
        assert np.array_equal(pts1, pts2)
        
        if not isinstance(pts, np.ndarray):
            pts = np.array(pts)
        pts2 = pipeline_utils.apply_operations(torch.from_numpy(pts), operations)
        assert np.array_equal(pts1, pts2.numpy())
        
        if torch.cuda.is_available():
            pts = torch.tensor(pts).cuda()
            pts2 = pipeline_utils.apply_operations(pts, operations)
            assert pts2.is_cuda
            assert np.array_equal(pts1, pts2.cpu().numpy())
            
            
class Test_revert_mask_to_origin:
    @pytest.mark.parametrize(
        "mask, operations, expected_shape",
        [
            (np.random.randint(0, 255, (100,100,3)), [{'pad': [8, 9, 10, 11]}, {'flip': [True, False, 100, 100]}], (79, 83, 3)),
            (np.random.randint(0, 255, (90,100)), [{'flip': [True, False, 100, 90]}], (90, 100)),
        ]
    )
    def test_cases(self, mask, operations, expected_shape):
        mask2 = pipeline_utils.revert_mask_to_origin(mask, operations)
        assert mask2.shape == expected_shape
        
        if 'flip' in operations[0]:
            lr,up,im_w,im_h = operations[0]['flip']
            mask3 = mask.copy()
            if lr:
                mask3 = np.flip(mask3, axis=1)
            if up:
                mask3 = np.flip(mask3, axis=0)
            assert np.array_equal(mask2, mask3)
        