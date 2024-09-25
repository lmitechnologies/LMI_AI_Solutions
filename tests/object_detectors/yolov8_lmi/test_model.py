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
from yolov8_lmi.model import Yolov8, Yolov8Obb, Yolov8Pose


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


COCO_DIR = 'tests/assets/images/coco'
DOTA_DIR = 'tests/assets/images/dota'
MODEL_DET = 'tests/assets/models/od/yolov8n.pt'
MODEL_SEG = 'tests/assets/models/od/yolov8n-seg.pt'
MODEL_OBB = 'tests/assets/models/od/yolov8n-obb.pt'
MODEL_POSE = 'tests/assets/models/od/yolov8n-pose.pt'
OUT_DIR = 'tests/assets/validation'


@pytest.fixture
def model_det():
    return Yolov8(MODEL_DET)

@pytest.fixture
def model_seg():
    return Yolov8(MODEL_SEG)

@pytest.fixture
def model_obb():
    return Yolov8Obb(MODEL_OBB)

@pytest.fixture
def model_pose():
    return Yolov8Pose(MODEL_POSE)

def load_image(path):
    im = cv2.imread(path)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb

@pytest.fixture
def imgs_coco():
    im_dim = 640
    paths = [os.path.join(COCO_DIR, img) for img in os.listdir(COCO_DIR)]
    images = []
    resized_images = []
    ops = []
    for p in paths:
        rgb = load_image(p)
        h,w = rgb.shape[:2]
        im2 = cv2.resize(rgb, (im_dim, im_dim))
        images.append(rgb)
        resized_images.append(im2)
        ops.append([{'resize': (im_dim,im_dim,w,h)}])
    return images, resized_images, ops

@pytest.fixture
def imgs_dota():
    im_dim = 1024
    paths = [os.path.join(DOTA_DIR, img) for img in os.listdir(DOTA_DIR)]
    images = []
    resized_images = []
    ops = []
    for p in paths:
        rgb = load_image(p)
        h,w = rgb.shape[:2]
        im2 = cv2.resize(rgb, (im_dim, im_dim))
        images.append(rgb)
        resized_images.append(im2)
        ops.append([{'resize': (im_dim,im_dim,w,h)}])
    return images, resized_images, ops


class Test_Yolov8:
    def test_warmup(self, model_det):
        model_det.warmup()
            
    def test_predict(self, model_det, imgs_coco):
        i = 0
        for img,resized,op in zip(*imgs_coco):
            out,time_info = model_det.predict(resized,configs=0.5,operators=op)
            assert len(out['boxes'])>0
            for sc in out['scores']:
                assert sc>=0.5
            im_out = model_det.annotate_image(out, img)
                
            if torch.cuda.is_available():
                resized = torch.from_numpy(resized).cuda()
                out,time_info = model_det.predict(resized,configs=0.5,operators=op)
                for b,sc in zip(out['boxes'], out['scores']):
                    assert b.is_cuda and sc.is_cuda
                img = torch.from_numpy(img).cuda()
                im_out = model_det.annotate_image(out, img)
                os.makedirs(OUT_DIR, exist_ok=True)
                im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(OUT_DIR, f'det-{i}.png'), im_out)
            i += 1
                
                
class Test_Yolov8_Seg:
    def test_warmup(self, model_seg):
        model_seg.warmup()
            
    def test_predict(self, model_seg, imgs_coco):
        i = 0
        for img,resized,op in zip(*imgs_coco):
            out,time_info = model_seg.predict(resized,configs=0.5,operators=op)
            assert len(out['masks'])>0 and len(out['segments'])>0
            for sc in out['scores']:
                assert sc>=0.5
            im_out = model_seg.annotate_image(out, img)
                
            if torch.cuda.is_available():
                resized = torch.from_numpy(resized).cuda()
                out,time_info = model_seg.predict(resized,configs=0.5,operators=op)
                for seg,m,b,sc in zip(out['segments'], out['masks'], out['boxes'], out['scores']):
                    assert seg.is_cuda and m.is_cuda and b.is_cuda and sc.is_cuda
                img = torch.from_numpy(img).cuda()
                im_out = model_seg.annotate_image(out, img)
                im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                os.makedirs(OUT_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(OUT_DIR, f'seg-{i}.png'), im_out)
            i += 1


class Test_Yolov8_obb:
    def test_warmup(self, model_obb):
        model_obb.warmup()
        
    def test_predict(self, model_obb, imgs_dota):
        i = 0
        for img,resized,op in zip(*imgs_dota):
            out,time_info = model_obb.predict(resized,configs=0.5,operators=op)
            assert len(out['boxes'])>0
            for sc in out['scores']:
                assert sc>=0.5
            im_out = model_obb.annotate_image(out, img)
                
            if torch.cuda.is_available():
                resized = torch.from_numpy(resized).cuda()
                out,time_info = model_obb.predict(resized,configs=0.5,operators=op)
                for b,sc in zip(out['boxes'], out['scores']):
                    assert b.is_cuda and sc.is_cuda
                img = torch.from_numpy(img).cuda()
                im_out = model_obb.annotate_image(out, img)
                im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                os.makedirs(OUT_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(OUT_DIR, f'obb-{i}.png'), im_out)
            i += 1
            

class Test_Yolov8_pose:
    def test_warmup(self, model_pose):
        model_pose.warmup()
        
    def test_predict(self, model_pose, imgs_coco):
        i = 0
        for img,resized,op in zip(*imgs_coco):
            out,time_info = model_pose.predict(resized,configs=0.5,operators=op)
            assert len(out['boxes'])>0
            for sc in out['scores']:
                assert sc>=0.5
            im_out = model_pose.annotate_image(out, img)
                
            if torch.cuda.is_available():
                resized = torch.from_numpy(resized).cuda()
                out,time_info = model_pose.predict(resized,configs=0.5,operators=op)
                for b,sc,kp in zip(out['boxes'], out['scores'], out['points']):
                    assert b.is_cuda and sc.is_cuda and kp.is_cuda
                img = torch.from_numpy(img).cuda()
                im_out = model_pose.annotate_image(out, img)
                im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                os.makedirs(OUT_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(OUT_DIR, f'pose-{i}.png'), im_out)
            i += 1