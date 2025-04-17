import pytest
import torch
import numpy as np
import logging
import sys
import os
import cv2

# add path to the repo
PATH = os.path.abspath(__file__)
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))))
# sys.path.append(os.path.join(ROOT, 'lmi_utils'))
# sys.path.append(os.path.join(ROOT, 'object_detectors'))

@pytest.fixture()
def add_root_path(request):
    if request.config.getoption("--test-package") is False:
        sys.path.append(os.path.join(ROOT, 'lmi_utils'))
        sys.path.append(os.path.join(ROOT, 'object_detectors'))
        logger.info(f"Added {ROOT} to sys.path")
    else:
        logger.info("Skipping adding root path to sys.path")

import gadget_utils.pipeline_utils as pipeline_utils
from ultralytics_lmi.yolo.model import Yolo, YoloObb, YoloPose
from od_core.object_detector import ObjectDetector


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


COCO_DIR = 'tests/assets/images/coco'
DOTA8_DIR = 'tests/assets/images/dota8'
DOTA_DIR = 'tests/assets/images/dota'
OUT_DIR = 'tests/assets/validation'

OD_DET_MODELS = [
    "tests/assets/models/od/yolo11n.pt",
    "tests/assets/models/od/yolov8n.pt"
]

OD_SEG_MODELS = [
    "tests/assets/models/od/yolo11n-seg.pt",
    "tests/assets/models/od/yolov8n-seg.pt"
]


OD_OBB_DOTA_8 = [
    "tests/assets/models/od/yolo11n-obb.pt",
]

OD_OBB_DOTA = [
    "tests/assets/models/od/yolov8n-obb.pt",
]

OD_POSE_MODELS = [
    "tests/assets/models/od/yolo11n-pose.pt",
    "tests/assets/models/od/yolov8n-pose.pt"
]

@pytest.fixture
def model_det():
    return [
        Yolo(model) for model in OD_DET_MODELS
    ]

@pytest.fixture
def model_seg():
    return [
        Yolo(model) for model in OD_SEG_MODELS
    ]

@pytest.fixture
def model_obb_dota8():
    return [
        YoloObb(model) for model in OD_OBB_DOTA_8
    ]
    
@pytest.fixture
def model_obb_dota():
    return [
        YoloObb(model) for model in OD_OBB_DOTA
    ]

@pytest.fixture
def model_pose():
    return [
        YoloPose(model) for model in OD_POSE_MODELS
    ]
    
@pytest.fixture
def model_det_api():
    return [
        ObjectDetector(metadata=dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='od', framework='ultralytics', model_path=model)) for model in OD_DET_MODELS
    ]

@pytest.fixture
def model_seg_api():
    return [
        ObjectDetector(metadata=dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='seg', framework='ultralytics'), model_path=model) for model in OD_SEG_MODELS
    ]

@pytest.fixture
def model_obb_dota8_api():
    return [
        ObjectDetector(metadata=dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='obb', framework='ultralytics'), model_path=model) for model in OD_OBB_DOTA_8
    ]
    
@pytest.fixture
def model_obb_dota_api():
    return [
        ObjectDetector(metadata=dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='obb', framework='ultralytics'), model_path=model) for model in OD_OBB_DOTA
    ]


@pytest.fixture
def model_pose_api():
    return [
        ObjectDetector(dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='pose', framework='ultralytics'), model_path=model) for model in OD_POSE_MODELS
    ]

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
        if 'png' not in p and 'jpg' not in p:
            continue
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
        if 'png' not in p and 'jpg' not in p:
            continue
        rgb = load_image(p)
        h,w = rgb.shape[:2]
        im2 = cv2.resize(rgb, (im_dim, im_dim))
        images.append(rgb)
        resized_images.append(im2)
        ops.append([{'resize': (im_dim,im_dim,w,h)}])
    return images, resized_images, ops

@pytest.fixture
def imgs_dota8():
    im_dim = 1024
    paths = [os.path.join(DOTA8_DIR, img) for img in os.listdir(DOTA8_DIR)]
    images = []
    resized_images = []
    ops = []
    for p in paths:
        if 'png' not in p and 'jpg' not in p:
            continue
        rgb = load_image(p)
        h,w = rgb.shape[:2]
        im2 = cv2.resize(rgb, (im_dim, im_dim))
        images.append(rgb)
        resized_images.append(im2)
        ops.append([{'resize': (im_dim,im_dim,w,h)}])
    return images, resized_images, ops


class Test_Yolo_Det:
    def test_warmup(self, model_det,add_root_path):
        for model in model_det:
            model.warmup()
            
    def test_predict(self, model_det, imgs_coco, add_root_path):
        i = 0
        for model in model_det:
            for img,resized,op in zip(*imgs_coco):
                out,time_info = model.predict(resized,configs=0.5,operators=op)
                assert len(out['boxes'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for b,sc in zip(out['boxes'], out['scores']):
                        assert b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(OUT_DIR, f'det-{i}.png'), im_out)
                i += 1

class Test_Yolo_Det_API:
    def test_warmup(self, model_det_api, add_root_path):
        for model in model_det_api:
            model.warmup()
            
    def test_predict(self, model_det_api, imgs_coco, add_root_path):
        i = 0
        for model in model_det_api:
            for img,resized,op in zip(*imgs_coco):
                out,time_info = model.predict(resized,configs=0.5,operators=op)
                assert len(out['boxes'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for b,sc in zip(out['boxes'], out['scores']):
                        assert b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(OUT_DIR, f'det-{i}.png'), im_out)
                i += 1
                
class Test_Yolo_Seg:
    def test_warmup(self, model_seg, add_root_path):
        for model in model_seg:
            model.warmup()
            
    def test_predict(self, model_seg, imgs_coco, add_root_path):
        i = 0
        for model in model_seg:
            for img,resized,op in zip(*imgs_coco):
                out,time_info = model.predict(resized,configs=0.5,operators=op)
                assert len(out['masks'])>0 and len(out['segments'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for seg,m,b,sc in zip(out['segments'], out['masks'], out['boxes'], out['scores']):
                        assert seg.is_cuda and m.is_cuda and b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f'seg-{i}.png'), im_out)
                i += 1

class Test_Yolo_Seg_API:
    def test_warmup(self, model_seg_api, add_root_path):
        for model in model_seg_api:
            model.warmup()
            
    def test_predict(self, model_seg_api, imgs_coco, add_root_path):
        i = 0
        for model in model_seg_api:
            for img,resized,op in zip(*imgs_coco):
                out,time_info = model.predict(resized,configs=0.5,operators=op)
                assert len(out['masks'])>0 and len(out['segments'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for seg,m,b,sc in zip(out['segments'], out['masks'], out['boxes'], out['scores']):
                        assert seg.is_cuda and m.is_cuda and b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f'seg-{i}.png'), im_out)
                i += 1

class Test_Yolo_Obb:
    def test_warmup_dota8(self, model_obb_dota8, add_root_path):
        for model in model_obb_dota8:
            model.warmup()
            
    def test_warmup_dota(self, model_obb_dota, add_root_path):
        for model in model_obb_dota:
            model.warmup()
        
    def test_predict_dota8(self, model_obb_dota8, imgs_dota8, add_root_path):
        i = 0
        for model in model_obb_dota8:
            for img,resized,op in zip(*imgs_dota8):
                out,time_info = model.predict(resized,configs=0.5,operators=op) 
                assert len(out['boxes'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for b,sc in zip(out['boxes'], out['scores']):
                        assert b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f'obb-{i}.png'), im_out)
                i += 1
    
    def test_predict_dota(self, model_obb_dota, imgs_dota, add_root_path):
        i = 0
        for model in model_obb_dota:
            for img,resized,op in zip(*imgs_dota):
                out,time_info = model.predict(resized,configs=0.5,operators=op)
                assert len(out['boxes'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for b,sc in zip(out['boxes'], out['scores']):
                        assert b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f'obb-{i}.png'), im_out)
                i += 1

class Test_Yolo_Obb_API:
    def test_warmup_dota8(self, model_obb_dota8_api, add_root_path):
        for model in model_obb_dota8_api:
            model.warmup()
            
    def test_warmup_dota(self, model_obb_dota_api, add_root_path):
        for model in model_obb_dota_api:
            model.warmup()
        
    def test_predict_dota8(self, model_obb_dota8_api, imgs_dota8, add_root_path):
        i = 0
        for model in model_obb_dota8_api:
            for img,resized,op in zip(*imgs_dota8):
                out,time_info = model.predict(resized,configs=0.5,operators=op) 
                assert len(out['boxes'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for b,sc in zip(out['boxes'], out['scores']):
                        assert b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f'obb-{i}.png'), im_out)
                i += 1
    
    def test_predict_dota(self, model_obb_dota_api, imgs_dota, add_root_path):
        i = 0
        for model in model_obb_dota_api:
            for img,resized,op in zip(*imgs_dota):
                out,time_info = model.predict(resized,configs=0.5,operators=op)
                assert len(out['boxes'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for b,sc in zip(out['boxes'], out['scores']):
                        assert b.is_cuda and sc.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f'obb-{i}.png'), im_out)
                i += 1  

class Test_Yolo_Pose:
    def test_warmup(self, model_pose, add_root_path):
        for model in model_pose:
            model.warmup()
        
    def test_predict(self, model_pose, imgs_coco, add_root_path):
        i = 0
        for model in model_pose:
            for img,resized,op in zip(*imgs_coco):
                out,time_info = model.predict(resized,configs=0.5,operators=op)
                assert len(out['boxes'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for b,sc,kp in zip(out['boxes'], out['scores'], out['points']):
                        assert b.is_cuda and sc.is_cuda and kp.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f'pose-{i}.png'), im_out)
                i += 1
                

class Test_Yolo_Pose_API:
    def test_warmup(self, model_pose_api, add_root_path):
        for model in model_pose_api:
            model.warmup()
        
    def test_predict(self, model_pose_api, imgs_coco, add_root_path):
        i = 0
        for model in model_pose_api:
            for img,resized,op in zip(*imgs_coco):
                out,time_info = model.predict(resized,configs=0.5,operators=op)
                assert len(out['boxes'])>0
                for sc in out['scores']:
                    assert sc>=0.5
                im_out = model.annotate_image(out, img)
                    
                if torch.cuda.is_available():
                    resized = torch.from_numpy(resized).cuda()
                    out,time_info = model.predict(resized,configs=0.5,operators=op)
                    for b,sc,kp in zip(out['boxes'], out['scores'], out['points']):
                        assert b.is_cuda and sc.is_cuda and kp.is_cuda
                    img = torch.from_numpy(img).cuda()
                    im_out = model.annotate_image(out, img)
                    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(OUT_DIR, f'pose-{i}.png'), im_out)
                i += 1