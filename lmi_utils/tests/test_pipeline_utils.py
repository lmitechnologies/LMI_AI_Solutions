import pytest
import gadget_utils.pipeline_utils as pipeline_utils
import torch
import numpy as np
import cv2
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Test_resize_image:
    def test_resize_height(self):
        im = torch.rand(150, 100, 3).numpy()
        im2 = pipeline_utils.resize_image(im, H=300)
        assert im2.shape == (300, 200, 3)
        
    def test_resize_width(self):
        im = torch.rand(100, 150, 3).numpy()
        im2 = pipeline_utils.resize_image(im, W=30)
        assert im2.shape == (20, 30, 3)
        
    def test_warp(self):
        im = torch.rand(150, 100, 3).numpy()
        im2 = pipeline_utils.resize_image(im, 200, 200)
        assert im2.shape == (200, 200, 3)
        
    def test_gray_image(self):
        im = torch.rand(150, 100).numpy()
        im2 = pipeline_utils.resize_image(im, W=50)
        assert im2.shape == (75, 50, 1)
        assert type(im2) == np.ndarray
    
    def test_tensor_cpu(self):
        im  = torch.rand(100, 150, 3)
        im2 = pipeline_utils.resize_image(im, H=10)
        assert im2.shape == (10, 15, 3)
        assert type(im2) == torch.Tensor
        
    def test_tensor_gpu(self):
        if torch.cuda.is_available():
            im  = torch.rand(100, 100, 3).cuda()
            im2 = pipeline_utils.resize_image(im, 150)
            assert im2.shape == (150, 150, 3)
            assert type(im2) == torch.Tensor
            assert im2.is_cuda

            
            
class Test_fit_im_to_size:

    def test_pad_w(self):
        im = torch.rand(100, 100, 3).numpy()
        im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, W=121)
        assert l==10 and r==11 and t==0 and b==0
        assert im2.shape == (100, 121, 3)
    
    def test_pad_h(self):
        im = torch.rand(100, 100, 3).numpy()
        im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, H=131)
        assert l==0 and r==0 and t==15 and b==16
        assert im2.shape == (131, 100, 3)
        
    def test_pad_hw(self):
        im = torch.rand(100, 100, 3).numpy()
        im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, W=151, H=131)
        assert l==25 and r==26 and t==15 and b==16
        assert im2.shape == (131, 151, 3)
        
    def test_crop_hw(self):
        im = torch.rand(100, 100, 3).numpy()
        im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, W=71, H=75)
        assert l==-14 and r==-15 and t==-12 and b==-13
        assert im2.shape == (75, 71, 3)
        
    def test_crop_h(self):
        im = torch.rand(100, 100, 3).numpy()
        im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, H=89)
        assert l==0 and r==0 and t==-5 and b==-6
        assert im2.shape == (89, 100, 3)
        
    def test_crop_w(self):
        im = torch.rand(100, 100, 3).numpy()
        im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, W=89)
        assert l==-5 and r==-6 and t==0 and b==0
        assert im2.shape == (100, 89, 3)
        
    def test_gray_image(self):
        im = torch.rand(100, 100).numpy()
        im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, W=89)
        assert l==-5 and r==-6 and t==0 and b==0
        assert im2.shape == (100, 89)
        assert type(im2) == np.ndarray
        
    def test_tensor_cpu(self):
        im = torch.rand(100, 100, 3)
        im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, W=89)
        assert l==-5 and r==-6 and t==0 and b==0
        assert im2.shape == (100, 89, 3)
        assert type(im2) == torch.Tensor
        
    def test_tensor_gpu(self):
        if torch.cuda.is_available():
            im = torch.rand(100, 100, 3).cuda()
            im2,l,r,t,b = pipeline_utils.fit_im_to_size(im, W=89)
            assert l==-5 and r==-6 and t==0 and b==0
            assert im2.shape == (100, 89, 3)
            assert type(im2) == torch.Tensor
            assert im2.is_cuda