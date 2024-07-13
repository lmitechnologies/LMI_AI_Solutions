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
sys.path.append(os.path.join(ROOT, 'object_detectors'))


import gadget_utils.pipeline_utils as pipeline_utils
from yolov8_lmi.model import Yolov8


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


IMG_DIR = 'tests/assets/images'
MODEL_DET = 'tests/assets/models/yolov8n.pt'
MODEL_SEG = 'tests/assets/models/yolov8n-seg.pt'

@pytest.fixture
def model_det():
    return Yolov8(MODEL_DET)

@pytest.fixture
def model_seg():
    return Yolov8(MODEL_SEG)

@pytest.fixture
def test_inputs():
    paths = [os.path.join(IMG_DIR, img) for img in os.listdir(IMG_DIR)]
    images = []
    resized_images = []
    ops = []
    for p in paths:
        im = cv2.imread(p)
        h,w = im.shape[:2]
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im2 = cv2.resize(rgb, (640, 640))
        images.append(im)
        resized_images.append(im2)
        ops.append([{'resize': (640,640,w,h)}])
    return images, resized_images, ops


class Test_Yolov8:
    def test_warmup(self, model_det):
        model_det.warmup()
            
    def test_predict(self, model_det, test_inputs):
        for img,reszied,op in zip(*test_inputs):
            out,time_info = model_det.predict(reszied,configs=0.5,operators=op,return_tensor=False)
            assert len(out['boxes'])>0
            for sc in out['scores']:
                assert sc>=0.5
            im_out = model_det.annotate_image(out, img)
                
            if torch.cuda.is_available():
                out,time_info = model_det.predict(reszied,configs=0.5,operators=op,return_tensor=True)
                for b,sc in zip(out['boxes'], out['scores']):
                    assert b.is_cuda
                    assert sc.is_cuda
                img = torch.from_numpy(img).cuda()
                im_out = model_det.annotate_image(out, img)
                
                
class Test_Yolov8_Seg:
    def test_warmup(self, model_seg):
        model_seg.warmup()
            
    def test_predict(self, model_seg, test_inputs):
        for img,resized,op in zip(*test_inputs):
            out,time_info = model_seg.predict(resized,configs=0.5,operators=op,return_tensor=False)
            assert len(out['masks'])>0
            for sc in out['scores']:
                assert sc>=0.5
            im_out = model_seg.annotate_image(out, img)
                
            if torch.cuda.is_available():
                out,time_info = model_seg.predict(resized,configs=0.5,operators=op,return_tensor=True)
                for m,b,sc in zip(out['masks'], out['boxes'], out['scores']):
                    assert m.is_cuda
                    assert b.is_cuda
                    assert sc.is_cuda
                img = torch.from_numpy(img).cuda()
                im_out = model_seg.annotate_image(out, img)
