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
OUT_DIR = 'tests/assets/validation'

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestDetectron2ModelPT:

    def test_model(self):
        og_model = model_zoo.get(MASKRCNN_MODEL_CONFIG, trained=True)
        og_model.eval()
        img = get_sample_coco_image()
        inputs = [{"image": img}]
        with torch.no_grad():
            orginal_preds = og_model.inference(inputs, do_postprocess=False)[0]
            
        confs = {
           v:0.00 for k,v in class_map.items()
        }
        model = Detectron2Model(MODEL_PATH, class_map=class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        preds = model.predict(image, confs=confs, process_masks=False)
        assert orginal_preds.pred_boxes.tensor.shape == preds.get('boxes')[0].shape
        assert orginal_preds.pred_classes.shape == preds.get('classes')[0].shape
        assert orginal_preds.scores.shape == preds.get('scores')[0].shape
        assert orginal_preds.pred_masks.shape == preds.get('masks')[0].shape
        
        # check if the scores are all close
        assert np.allclose(orginal_preds.scores.cpu().numpy(), preds.get('scores'))
        assert np.allclose(orginal_preds.pred_masks.cpu().numpy(), preds.get('masks'))
        
    def test_annotations(self):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        model = Detectron2Model(MODEL_PATH, class_map=class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        outputs = model.predict(image, confs=confs, return_segments=True, process_masks=True)
        outputs['boxes'] = outputs['boxes'][0]
        outputs['classes'] = outputs['classes'][0]
        outputs['scores'] = outputs['scores'][0]
        outputs['masks'] = outputs['masks'][0]
        outputs['segments'] = outputs['segments'][0]
        
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        
        annotated_image = model.annotate_image(
           outputs, image, show_segments=True
        )
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
        
    def test_operators(self):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        model = Detectron2Model(MODEL_PATH, class_map=class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        image = cv2.resize(image, (512, 512))
        operators = [{'resize': [1024,1024,512,512]}]
        outputs = model.predict(image, confs=confs, return_segments=True, process_masks=True, operators=operators)
        outputs['boxes'] = outputs['boxes'][0]
        outputs['classes'] = outputs['classes'][0]
        outputs['scores'] = outputs['scores'][0]
        outputs['masks'] = outputs['masks'][0]
        outputs['segments'] = outputs['segments'][0]
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        assert outputs['masks'].shape[1] == 512
        assert outputs['masks'].shape[2] == 512
        
        annotated_image = model.annotate_image(
           outputs, image, show_segments=True
        )
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
    
    def test_operators_no_masks(self):
        confs = {
           v:1.0 for k,v in class_map.items()
        }
        model = Detectron2Model(MODEL_PATH, class_map=class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        image = cv2.resize(image, (512, 512))
        operators = [{'resize': [1024,1024,512,512]}]
        outputs = model.predict(image, confs=confs, return_segments=True, process_masks=True, operators=operators)
        outputs['boxes'] = outputs['boxes'][0]
        outputs['classes'] = outputs['classes'][0]
        outputs['scores'] = outputs['scores'][0]
        outputs['masks'] = outputs['masks'][0]
        outputs['segments'] = outputs['segments'][0]
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        assert len(outputs['boxes']) == 0
        
class TestDetectron2ModelPT_API:

    def test_model(self):
        og_model = model_zoo.get(MASKRCNN_MODEL_CONFIG, trained=True)
        og_model.eval()
        img = get_sample_coco_image()
        inputs = [{"image": img}]
        with torch.no_grad():
            orginal_preds = og_model.inference(inputs, do_postprocess=False)[0]
            
        confs = {
           v:0.00 for k,v in class_map.items()
        }
        model = ObjectDetector(metadata=dict(version='v0', model_name='mask_rcnn', task='seg', framework='detectron2'), model_path=MODEL_PATH, class_map=class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        preds = model.predict(image, confs=confs, process_masks=False)
        assert orginal_preds.pred_boxes.tensor.shape == preds.get('boxes')[0].shape
        assert orginal_preds.pred_classes.shape == preds.get('classes')[0].shape
        assert orginal_preds.scores.shape == preds.get('scores')[0].shape
        assert orginal_preds.pred_masks.shape == preds.get('masks')[0].shape
        
        # check if the scores are all close
        assert np.allclose(orginal_preds.scores.cpu().numpy(), preds.get('scores'))
        assert np.allclose(orginal_preds.pred_masks.cpu().numpy(), preds.get('masks'))
        
    def test_annotations(self):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        model = ObjectDetector(metadata=dict(version='v0', model_name='mask_rcnn', task='seg', framework='detectron2'), model_path=MODEL_PATH, class_map=class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        outputs = model.predict(image, confs=confs, return_segments=True, process_masks=True)
        outputs['boxes'] = outputs['boxes'][0]
        outputs['classes'] = outputs['classes'][0]
        outputs['scores'] = outputs['scores'][0]
        outputs['masks'] = outputs['masks'][0]
        outputs['segments'] = outputs['segments'][0]
        
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        
        annotated_image = model.annotate_image(
           outputs, image, show_segments=True
        )
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
        
    def test_operators(self):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        model = ObjectDetector(metadata=dict(version='v0', model_name='mask_rcnn', task='seg', framework='detectron2'), model_path=MODEL_PATH, class_map=class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        image = cv2.resize(image, (512, 512))
        operators = [{'resize': [1024,1024,512,512]}]
        outputs = model.predict(image, confs=confs, return_segments=True, process_masks=True, operators=operators)
        outputs['boxes'] = outputs['boxes'][0]
        outputs['classes'] = outputs['classes'][0]
        outputs['scores'] = outputs['scores'][0]
        outputs['masks'] = outputs['masks'][0]
        outputs['segments'] = outputs['segments'][0]
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        assert outputs['masks'].shape[1] == 512
        assert outputs['masks'].shape[2] == 512
        
        annotated_image = model.annotate_image(
           outputs, image, show_segments=True
        )
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
    
    def test_operators_no_masks(self):
        confs = {
           v:1.0 for k,v in class_map.items()
        }
        model = ObjectDetector(metadata=dict(version='v0', model_name='mask_rcnn', task='seg', framework='detectron2'), model_path=MODEL_PATH, class_map=class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        image = cv2.resize(image, (512, 512))
        operators = [{'resize': [1024,1024,512,512]}]
        outputs = model.predict(image, confs=confs, return_segments=True, process_masks=True, operators=operators)
        outputs['boxes'] = outputs['boxes'][0]
        outputs['classes'] = outputs['classes'][0]
        outputs['scores'] = outputs['scores'][0]
        outputs['masks'] = outputs['masks'][0]
        outputs['segments'] = outputs['segments'][0]
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        assert len(outputs['boxes']) == 0