import logging
from collections.abc import Sequence
import sys
import os
import tempfile
import glob
import cv2
import numpy as np
import torch
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer
from anomalib.data.utils import read_image

# add path to the repo
PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))
sys.path.append(os.path.join(ROOT, 'lmi_utils'))
sys.path.append(os.path.join(ROOT, 'anomaly_detectors'))


import gadget_utils.pipeline_utils as pipeline_utils
from anomalib_lmi.anomaly_model2 import AnomalyModel2


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DATA_PATH = 'tests/assets/images/nvtec-ad'
MODEL_PATH = 'tests/assets/models/ad/model_v1.pt'
OUTPUT_PATH = 'tests/assets/validation/ad_v1'



def test_compare_results_with_anomalib():
    """
    compare prediction results between current implementation and anomalib
    """
    model1 = TorchInferencer(MODEL_PATH)
    model2 = AnomalyModel2(MODEL_PATH)
    paths = glob.glob(os.path.join(DATA_PATH, '*.png'))
    for p in paths:
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
        pred = pred.cpu().numpy()
        
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pred2 = model2.predict(rgb)
        
        assert np.allclose(pred,pred2,rtol=0)
        
        
def test_model():
    AnomalyModel2.test(
        MODEL_PATH, DATA_PATH, OUTPUT_PATH, generate_stats=True,annotate_inputs=True,
    )
    
    
def test_convert():
    with tempfile.TemporaryDirectory() as t:
        AnomalyModel2.convert(MODEL_PATH,t,fp16=True)
        AnomalyModel2.test(
            os.path.join(t,'model.engine'), DATA_PATH, OUTPUT_PATH, generate_stats=True,annotate_inputs=True,
        )
    