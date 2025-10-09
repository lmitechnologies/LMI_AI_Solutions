import pytest
import torch
import numpy as np
import logging
import os
import cv2
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

import gadget_utils.pipeline_utils as pipeline_utils
from ultralytics_lmi.yolo.model import Yolo, YoloSeg, YoloObb, YoloPose
from od_core.object_detector import ObjectDetector


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


COCO_DIR = 'tests/assets/images/coco'
DOTA8_DIR = 'tests/assets/images/dota8'
DOTA_DIR = 'tests/assets/images/dota'
OUT_DIR = 'tests/outputs/od/ultralytics/yolo'

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
        YoloSeg(model) for model in OD_SEG_MODELS
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
        ObjectDetector(metadata=dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='od', framework='ultralytics', model_path=model, image_size=[640, 640])) for model in OD_DET_MODELS
    ]

@pytest.fixture
def model_seg_api():
    return [
        ObjectDetector(metadata=dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='seg', framework='ultralytics', model_path=model, image_size=[640, 640])) for model in OD_SEG_MODELS
    ]

@pytest.fixture
def model_obb_dota8_api():
    return [
        ObjectDetector(metadata=dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='obb', framework='ultralytics', model_path=model, image_size=[640, 640])) for model in OD_OBB_DOTA_8
    ]
    
@pytest.fixture
def model_obb_dota_api():
    return [
        ObjectDetector(metadata=dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='obb', framework='ultralytics', model_path=model, image_size=[640, 640])) for model in OD_OBB_DOTA
    ]


@pytest.fixture
def model_pose_api():
    return [
        ObjectDetector(metadata=dict(version='v1', model_name='yolov8' if 'yolov8n' in model else 'yolov11', task='pose', framework='ultralytics', model_path=model, image_size=[640, 640])) for model in OD_POSE_MODELS
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
    def test_compare_with_ultralytics(self, imgs_coco):
        for model_path in OD_DET_MODELS:
            ults_model = YOLO(model_path)
            our_model = Yolo(model_path)
            for img,resized,op in zip(*imgs_coco):
                resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                results = ults_model(resized_bgr, conf=0.5, iou=0.4, max_det=300, device='cuda' if torch.cuda.is_available() else 'cpu')
                ults_out = results[0].cpu().numpy()

                out,time_info = our_model.predict(resized, configs=0.5, iou=0.4, max_det=300)
                for sc1,sc2 in zip(out['scores'], ults_out.boxes.conf):
                    assert sc1==sc2
                for b1,b2 in zip(out['boxes'], ults_out.boxes.xyxy):
                    assert np.array_equal(b1, b2)
    
    def test_warmup(self, model_det):
        for model in model_det:
            model.warmup()
            
    def test_warmup_api(self, model_det_api):
        self.test_warmup(model_det_api)
        
    def test_predict_empty(self, model_det):
        operators = [{'resize': (640,640,1920,1080)}]
        for model in model_det:
            out,time_info = model.predict(np.zeros((640,640,3), dtype=np.uint8),configs=0.5,operators=operators)
            assert len(out['boxes'])==0 and len(out['scores'])==0
            
    def test_predict(self, model_det, imgs_coco):
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
                
    def test_predict_api(self, model_det_api, imgs_coco):
        self.test_predict(model_det_api, imgs_coco)
        
        
class Test_Yolo_Seg:
    def test_compare_with_ultralytics(self, imgs_coco):
        for model_path in OD_SEG_MODELS:
            ults_model = YOLO(model_path)
            our_model = Yolo(model_path)
            for img,resized,op in zip(*imgs_coco):
                resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                results = ults_model(resized_bgr, conf=0.5, iou=0.4, max_det=300, device='cuda' if torch.cuda.is_available() else 'cpu')
                ults_out = results[0].cpu().numpy()

                out,time_info = our_model.predict(resized, configs=0.5, iou=0.4, max_det=300, retina_masks=True)
                for sc1,sc2 in zip(out['scores'], ults_out.boxes.conf):
                    assert sc1==sc2
                for b1,b2 in zip(out['boxes'], ults_out.boxes.xyxy):
                    assert np.array_equal(b1, b2)
                for m1,m2 in zip(out['masks'], ults_out.masks.data):
                    assert np.array_equal(m1, m2)
                for s1,s2 in zip(out['segments'], results[0].masks.xy):
                    assert np.array_equal(s1, s2)
    
    def test_warmup(self, model_seg):
        for model in model_seg:
            model.warmup()
            
    def test_warmup_api(self, model_seg_api):
        self.test_warmup(model_seg_api)
        
    def test_predict_empty(self, model_seg):
        operators = [{'resize': (640,640,1920,1080)}]
        for model in model_seg:
            out,time_info = model.predict(np.zeros((640,640,3), dtype=np.uint8),configs=0.5,operators=operators)
            assert len(out['boxes'])==0 and len(out['masks'])==0 and len(out['segments'])==0 and len(out['scores'])==0
            
    def test_predict(self, model_seg, imgs_coco):
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
                
    def test_predict_api(self, model_seg_api, imgs_coco):
        self.test_predict(model_seg_api, imgs_coco)


class Test_Yolo_Obb:
    def compare_with_ultralytics(self, imgs, model_paths):
        for model_path in model_paths:
            ults_model = YOLO(model_path)
            our_model = YoloObb(model_path)
            for img,resized,op in zip(*imgs):
                resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                results = ults_model(resized_bgr, conf=0.5, iou=0.4, max_det=300, device='cuda' if torch.cuda.is_available() else 'cpu')
                ults_out = results[0].cpu().numpy()

                out,time_info = our_model.predict(resized, configs=0.5, iou=0.4, max_det=300)
                for sc1,sc2 in zip(out['scores'], ults_out.obb.conf):
                    assert sc1==sc2
                for b1,b2 in zip(out['boxes'], ults_out.obb.xyxyxyxy):
                    assert np.array_equal(b1, b2)
        
    def test_compare_with_ultralytics_dota8(self, imgs_dota8):
        self.compare_with_ultralytics(imgs_dota8, OD_OBB_DOTA_8)
        
    def test_compare_with_ultralytics_dota(self, imgs_dota):
        self.compare_with_ultralytics(imgs_dota, OD_OBB_DOTA)
    
    def test_warmup_dota8(self, model_obb_dota8):
        for model in model_obb_dota8:
            model.warmup()
            
    def test_warmup_dota8_api(self, model_obb_dota8_api):
        self.test_warmup_dota8(model_obb_dota8_api)
            
    def test_warmup_dota(self, model_obb_dota):
        for model in model_obb_dota:
            model.warmup()
            
    def test_warmup_dota_api(self, model_obb_dota_api):
        self.test_warmup_dota(model_obb_dota_api)
        
    def test_predict_empty(self, model_obb_dota8, model_obb_dota):
        operators = [{'resize': (640,640,1920,1080)}]
        for model in model_obb_dota8+model_obb_dota:
            out,time_info = model.predict(np.zeros((640,640,3), dtype=np.uint8),configs=0.5,operators=operators)
            assert len(out['boxes'])==0 and len(out['scores'])==0
        
    def test_predict_dota8(self, model_obb_dota8, imgs_dota8):
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
                
    def test_predict_dota8_api(self, model_obb_dota8_api, imgs_dota8):
        self.test_predict_dota8(model_obb_dota8_api, imgs_dota8)
    
    def test_predict_dota(self, model_obb_dota, imgs_dota):
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
                
    def test_predict_dota_api(self, model_obb_dota_api, imgs_dota):
        self.test_predict_dota(model_obb_dota_api, imgs_dota)
        

class Test_Yolo_Pose:
    def test_compare_with_ultralytics(self, imgs_coco):
        for model_path in OD_POSE_MODELS:
            ults_model = YOLO(model_path)
            our_model = YoloPose(model_path)
            for img,resized,op in zip(*imgs_coco):
                resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                results = ults_model(resized_bgr, conf=0.5, iou=0.4, max_det=300, device='cuda' if torch.cuda.is_available() else 'cpu')
                ults_out = results[0].cpu().numpy()

                out,time_info = our_model.predict(resized, configs=0.5, iou=0.4, max_det=300)
                for sc1,sc2 in zip(out['scores'], ults_out.boxes.conf):
                    assert np.array_equal(sc1, sc2)
                for b1,b2 in zip(out['boxes'], ults_out.boxes.xyxy):
                    assert np.array_equal(b1, b2)
                for k1,k2 in zip(out['points'], ults_out.keypoints.data):
                    assert np.array_equal(k1, k2)
    
    def test_warmup(self, model_pose):
        for model in model_pose:
            model.warmup()
            
    def test_warmup_api(self, model_pose_api):
        self.test_warmup(model_pose_api)
        
    def test_predict_empty(self, model_pose):
        operators = [{'resize': (640,640,1920,1080)}]
        for model in model_pose:
            out,time_info = model.predict(np.zeros((640,640,3), dtype=np.uint8),configs=0.5,operators=operators)
            assert len(out['boxes'])==0 and len(out['points'])==0 and len(out['scores'])==0
        
    def test_predict(self, model_pose, imgs_coco):
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
                
    def test_predict_api(self, model_pose_api, imgs_coco):
        self.test_predict(model_pose_api, imgs_coco)
        