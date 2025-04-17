import pytest
import logging
from collections.abc import Sequence
import sys
import os
import tempfile
import glob
import cv2
import numpy as np
import torch
import subprocess
import time
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer
from anomalib.data.utils import read_image

# add path to the repo
PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))
@pytest.fixture()
def add_root_path(request):
    if request.config.getoption("--test-package") is False:
        sys.path.append(os.path.join(ROOT, 'lmi_utils'))
        sys.path.append(os.path.join(ROOT, 'anomaly_detectors'))
        logger.info(f"Added {ROOT} to sys.path")
    else:
        logger.info("Skipping adding root path to sys.path")


from anomalib_lmi.anomaly_model2 import AnomalyModel2
from ad_core.anomaly_detector import AnomalyDetector
from gadget_utils import pipeline_utils


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DATA_PATH = 'tests/assets/images/nvtec-ad'
MODEL_PATH = 'tests/assets/models/ad/model_v1.pt'
OUTPUT_PATH = 'tests/assets/validation/ad_v1'


@pytest.fixture
def test_data():
    paths = glob.glob(os.path.join(DATA_PATH, '*.png'))
    out = []
    names = []
    for p in paths:
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out.append(rgb)
        names.append(os.path.basename(p))
    return out,names


def test_compare_results_with_anomalib(add_root_path):
    """
    compare prediction results between current implementation and anomalib
    """
    model1 = TorchInferencer(MODEL_PATH)
    model2 = AnomalyModel2(MODEL_PATH)
    paths = glob.glob(os.path.join(DATA_PATH, '*.png'))
    for p in paths:
        # using anomalib code
        tensor = read_image(p,as_tensor=True)
        pred = model1.forward(model1.pre_process(tensor))
        if isinstance(pred, dict):
            pred = pred['anomaly_map']
        elif isinstance(pred, Sequence):
            pred = pred[1]
        elif isinstance(pred, torch.Tensor):
            pass
        else:
            raise Exception(f'Not supported output: {type(pred)}')
        pred = pred.cpu().numpy().squeeze()
        
        # using AIS code
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pred2 = model2.predict(rgb)
        
        assert np.array_equal(pred, pred2)

def test_compare_results_with_anomalib_api(add_root_path):
    """
    compare prediction results between current implementation and anomalib
    """
    model1 = TorchInferencer(MODEL_PATH)
    model2 = AnomalyDetector(dict(framework='anomalib1', model_name='padim', task='seg', version='v1', model_path=MODEL_PATH))
    paths = glob.glob(os.path.join(DATA_PATH, '*.png'))
    for p in paths:
        # using anomalib code
        tensor = read_image(p,as_tensor=True)
        pred = model1.forward(model1.pre_process(tensor))
        if isinstance(pred, dict):
            pred = pred['anomaly_map']
        elif isinstance(pred, Sequence):
            pred = pred[1]
        elif isinstance(pred, torch.Tensor):
            pass
        else:
            raise Exception(f'Not supported output: {type(pred)}')
        pred = pred.cpu().numpy().squeeze()
        
        # using AIS code
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pred2 = model2.predict(rgb)
        
        assert np.array_equal(pred, pred2)

        
def test_warmup(add_root_path):
    ad = AnomalyModel2(MODEL_PATH,224,112)
    ad.warmup()
    ad.warmup([672,640])
    
    ad = AnomalyModel2(MODEL_PATH)
    ad.warmup()
    ad.warmup([256,224])

def test_warmup_api(add_root_path):
    ad = AnomalyDetector(dict(framework='anomalib1', model_name='padim', task='seg', version='v1', model_path=MODEL_PATH),224,112)
    ad.warmup()
    ad.warmup([672,640])
    
    ad = AnomalyDetector(dict(framework='anomalib1', model_name='padim', task='seg', version='v1', model_path=MODEL_PATH),224,112)
    ad.warmup()
    ad.warmup([256,224])
    

def test_model(add_root_path):
    ad = AnomalyModel2(MODEL_PATH,224,224,'resize')
    ad.test(DATA_PATH, os.path.join(OUTPUT_PATH,'tile-resize'))
    
    ad = AnomalyModel2(MODEL_PATH,224,224)
    ad.test(DATA_PATH, os.path.join(OUTPUT_PATH,'tile-pad'))
    
    ad = AnomalyModel2(MODEL_PATH)
    ad.test(DATA_PATH, OUTPUT_PATH)

def test_model_api(add_root_path):
    ad = AnomalyDetector(dict(framework='anomalib1', model_name='padim', version='v1', model_path=MODEL_PATH),224,224,'resize')
    ad.test(DATA_PATH, OUTPUT_PATH)
    
    ad = AnomalyDetector(dict(framework='anomalib1', model_name='padim', version='v1', model_path=MODEL_PATH))
    ad.test(DATA_PATH, OUTPUT_PATH)
    
def test_annotate(test_data, add_root_path):
    def old_func(img, ad_scores, ad_threshold, ad_max):
        # Resize AD score to match input image
        h_img,w_img=img.shape[:2]
        ad_scores=pipeline_utils.resize_image(ad_scores,H=h_img,W=w_img)
        # Set all low score pixels to threshold to improve heat map precision
        indices=np.where(ad_scores<ad_threshold)
        ad_scores[indices]=ad_threshold
        # Set upper limit on anomaly score.
        ad_scores[ad_scores>ad_max]=ad_max
        # Generate heat map
        ad_norm=(ad_scores-ad_threshold)/(ad_max-ad_threshold)
        ad_gray=(ad_norm*255).astype(np.uint8)
        ad_bgr = cv2.applyColorMap(np.expand_dims(ad_gray,-1), cv2.COLORMAP_TURBO)
        residual_rgb = cv2.cvtColor(ad_bgr, cv2.COLOR_BGR2RGB)
        # Overlay anomaly heat map with input image
        annot = cv2.addWeighted(img.astype(np.uint8), 0.6, residual_rgb, 0.4, 0)
        indices=np.where(ad_gray==0)
        # replace all below-threshold pixels with input image indicating no anomaly
        annot[indices]=img[indices]
        return annot
    
    
    ad = AnomalyModel2(MODEL_PATH)
    for _ in range(10):
        ad.warmup()
    
    out_path = os.path.join(OUTPUT_PATH,'annotate')
    os.makedirs(out_path,exist_ok=1)
    
    imgs,names = test_data
    for im,name in zip(imgs,names):
        pred = ad.predict(im)
        mean,max = pred.mean(),pred.max()
        
        t0 = time.time()
        out1 = old_func(im,pred,mean,max)
        t1 = time.time() - t0
        
        out2 = ad.annotate(im,pred,mean,max)
        bgr = cv2.cvtColor(out2,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_path,name),bgr)
        
        if torch.cuda.is_available():
            im = torch.from_numpy(im).cuda()
            pred = torch.from_numpy(pred).cuda()
            
            t0 = time.time()
            out3 = ad.annotate(im,pred,mean,max)
            t2 = time.time() - t0
            logger.info(f'improved proc time from {t1:.4f} to {t2:.4f}')
            
            assert np.array_equal(out1,out2)
            assert np.array_equal(out2,out3)
    
    
def test_cmds(add_root_path):
    """test model inference and model to tensorrt conversion
    """
    with tempfile.TemporaryDirectory() as t:
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = f'$PYTHONPATH:{ROOT}/lmi_utils:{ROOT}/anomaly_detectors'
        cmd = f'python -m anomalib_lmi.anomaly_model2 test -i {MODEL_PATH} -d {DATA_PATH} -o {str(t)} -g -p --tile 224 224 --stride 224 224 --resize'
        logger.info(f'running cmd: {cmd}')
        result = subprocess.run(cmd,shell=True,env=my_env,capture_output=True,text=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
        
        l1 = glob.glob(os.path.join(DATA_PATH, '*.png'))
        l2 = glob.glob(os.path.join(t, '*_annot.png'))
        assert len(l1) == len(l2)
        
        t2 = os.path.join(t,'recon')
        cmd = f'python -m anomalib_lmi.anomaly_model2 convert -i {MODEL_PATH} -o {t2} --hw 1120 1120 --tile 224 224 --stride 224 224'
        logger.info(f'running cmd: {cmd}')
        result = subprocess.run(cmd,shell=True,env=my_env,capture_output=True,text=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
        
        assert os.path.isfile(os.path.join(t2, 'model.engine'))
        