import os
import pytest
import json 
import cv2
import torch
from torch import Tensor, nn
from detectron2 import model_zoo
from detectron2.utils.testing import (
    get_sample_coco_image,
)
import logging
import numpy as np


PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))

@pytest.fixture()
def add_root_path(request):
    if request.config.getoption("--test-package") is False:
        sys.path.append(os.path.join(ROOT, 'lmi_utils'))
        sys.path.append(os.path.join(ROOT, 'object_detectors'))
        logger.info(f"Added {ROOT} to sys.path")
    else:
        logger.info("Skipping adding root path to sys.path")

from detectron2_lmi.model import Detectron2Model
from od_core.object_detector import ObjectDetector

with open('tests/assets/coco_class_names.txt','r') as f:
    classnames = f.readlines()

# create a class map "0":"class1", "1":"class2", ...
class_map = {str(i): classnames[i].strip() for i in range(len(classnames))}    

with open('tests/assets/models/od/detectron2/class_map.json','w') as f:
    json.dump(class_map, f)

MASKRCNN_MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
COCO_CLASSMAP = 'tests/assets/models/od/detectron2/class_map.json'
MODEL_PATH = 'tests/assets/models/od/detectron2/model.pt'
SAMPLE_IMAGE = 'tests/assets/images/detectron2/sample_image.jpg'
OUT_DIR = 'tests/outputs/od/detectron2'

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture(scope='module')
def og_model():
    model = model_zoo.get(MASKRCNN_MODEL_CONFIG, trained=True)
    model.eval()
    return model


@pytest.fixture(scope='module')
def detectron2_model():
    model = Detectron2Model(MODEL_PATH, class_map=class_map)
    return model


@pytest.fixture(scope='module')
def detectron2_model_api():
    model = ObjectDetector(
        metadata=dict(version='v0', model_name='mask_rcnn', task='seg', framework='detectron2'), 
        model_path=MODEL_PATH, 
        class_map=class_map
    )
    return model



class TestDetectron2ModelPT:

    def test_model(self, og_model, detectron2_model):
        img = get_sample_coco_image()
        inputs = [{"image": img}]
        with torch.no_grad():
            orginal_preds = og_model.inference(inputs, do_postprocess=False)[0]
            
        confs = {
           v:0.00 for k,v in class_map.items()
        }
        image = cv2.imread(SAMPLE_IMAGE)
        preds, _ = model.predict(image, confs=confs, process_masks=False, iou=0.0)
        assert orginal_preds.pred_boxes.tensor.shape == preds.get('boxes').shape
        assert orginal_preds.pred_classes.shape == preds.get('classes').shape
        assert orginal_preds.scores.shape == preds.get('scores').shape
        assert orginal_preds.pred_masks.shape == preds.get('masks').shape
        
        # check if the scores are all close
        assert np.allclose(orginal_preds.scores.cpu().numpy(), preds.get('scores'))
        assert np.allclose(orginal_preds.pred_masks.cpu().numpy(), preds.get('masks'))
        
    def test_annotations(self, detectron2_model):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        image = cv2.imread(SAMPLE_IMAGE)
        outputs, _ = model.predict(image, confs=confs, return_segments=True, process_masks=True, iou=0.0)
        
        assert len(outputs['boxes']) == len(outputs['classes']) 
        assert len(outputs['scores']) == len(outputs['classes'])
        assert len(outputs['masks']) == len(outputs['classes'])
        assert len(outputs['segments']) == len(outputs['classes'])
        
        annotated_image = detectron2_model.annotate_image(
           outputs, image, show_segments=True
        )
        os.makedirs(OUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
        
    def test_operators(self, detectron2_model):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        image = cv2.imread(SAMPLE_IMAGE)
        image = cv2.resize(image, (512, 512))
        operators = [{'resize': [1024,1024,512,512]}]
        outputs , _ = model.predict(image, confs=confs, return_segments=False, process_masks=True, operators=operators, iou=0.0)
        assert 'segments' not in outputs
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks'])
        assert outputs['masks'].shape[1] == 512
        assert outputs['masks'].shape[2] == 512
        
        annotated_image = model.annotate_image(
           outputs, image
        )
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
    
    def test_operators_no_masks(self, detectron2_model):
        confs = {
           v:1.0 for k,v in class_map.items()
        }
        image = cv2.imread(SAMPLE_IMAGE)
        image = cv2.resize(image, (512, 512))
        operators = [{'resize': [1024,1024,512,512]}]
        outputs, _ = model.predict(image, confs=confs, return_segments=True, process_masks=True, operators=operators)
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        assert len(outputs['boxes']) == 0
        
class TestDetectron2ModelPT_API:

    def test_model(self, og_model, detectron2_model_api):
        img = get_sample_coco_image()
        inputs = [{"image": img}]
        with torch.no_grad():
            orginal_preds = og_model.inference(inputs, do_postprocess=False)[0]
            
        confs = {
           v:0.00 for k,v in class_map.items()
        }
        model = ObjectDetector(metadata=dict(version='v0', model_name='mask_rcnn', task='seg', framework='detectron2', class_map=class_map), model_path=MODEL_PATH)
        image = cv2.imread(SAMPLE_IMAGE)
        preds, _ = model.predict(image, confs=confs, process_masks=False, iou=0.0)
        assert orginal_preds.pred_boxes.tensor.shape == preds.get('boxes').shape
        assert orginal_preds.pred_classes.shape == preds.get('classes').shape
        assert orginal_preds.scores.shape == preds.get('scores').shape
        assert orginal_preds.pred_masks.shape == preds.get('masks').shape
        
        # check if the scores are all close
        assert np.allclose(orginal_preds.scores.cpu().numpy(), preds.get('scores'))
        assert np.allclose(orginal_preds.pred_masks.cpu().numpy(), preds.get('masks'))
        
    def test_annotations(self, detectron2_model_api):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        model = ObjectDetector(metadata=dict(version='v0', model_name='mask_rcnn', task='seg', framework='detectron2', class_map=class_map), model_path=MODEL_PATH)
        image = cv2.imread(SAMPLE_IMAGE)
        outputs, _ = model.predict(image, confs=confs, return_segments=True, process_masks=True, iou=0.0)
        
        assert len(outputs['boxes']) == len(outputs['classes']) 
        assert len(outputs['scores']) == len(outputs['classes'])
        assert len(outputs['masks']) == len(outputs['classes'])
        assert len(outputs['segments']) == len(outputs['classes'])
        
        annotated_image = detectron2_model_api.annotate_image(
           outputs, image, show_segments=True
        )
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
        
    def test_operators(self, detectron2_model_api):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        model = ObjectDetector(metadata=dict(version='v0', model_name='mask_rcnn', task='seg', framework='detectron2', class_map=class_map), model_path=MODEL_PATH)
        image = cv2.imread(SAMPLE_IMAGE)
        image = cv2.resize(image, (512, 512))
        operators = [{'resize': [1024,1024,512,512]}]
        outputs , _ = model.predict(image, confs=confs, return_segments=True, process_masks=True, operators=operators, iou=0.0)
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        assert outputs['masks'].shape[1] == 512
        assert outputs['masks'].shape[2] == 512
        
        annotated_image = detectron2_model_api.annotate_image(
           outputs, image, show_segments=True
        )
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
    
    def test_operators_no_masks(self, detectron2_model_api):
        confs = {
           v:1.0 for k,v in class_map.items()
        }
        model = ObjectDetector(metadata=dict(version='v0', model_name='mask_rcnn', task='seg', framework='detectron2', class_map=class_map), model_path=MODEL_PATH)
        image = cv2.imread(SAMPLE_IMAGE)
        image = cv2.resize(image, (512, 512))
        operators = [{'resize': [1024,1024,512,512]}]
        outputs, _ = model.predict(image, confs=confs, return_segments=True, process_masks=True, operators=operators)
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        assert len(outputs['boxes']) == 0