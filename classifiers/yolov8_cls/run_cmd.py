import subprocess
from datetime import date
import logging
import yaml
import os

from yolov8_lmi.run_cmd import check_path_exist, sanity_check, get_model_path, add_configs

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# mounted locations in the docker container
HYP_YAML = '/app/config/hyp.yaml'

# default configs
DATASET_PATH = '/app/dataset'
TRAIN_FOLDER = '/app/training'
VAL_FOLDER = '/app/validation'
PREDICT_FOLDER = '/app/prediction'
MODEL_PATH = '/app/trained-inference-models'
MODEL_NAMES = ['best.engine','best.pt']
SOURCE_PATH = '/app/data'



if __name__=='__main__':
    # check if files exist
    check_path_exist(HYP_YAML, True)
        
    # load hyp yaml file
    with open(HYP_YAML) as f:
        hyp = yaml.safe_load(f)
       
    # use today's date as the default output folder name
    defaults = {'name':date.today().strftime("%Y-%m-%d")}
    
    # add other default configs
    check_keys = {} # map < key : True if is_file else False >
    path_wts = get_model_path(MODEL_PATH, hyp['mode'])
    if hyp['mode'] == 'train':
        tmp = {'data':DATASET_PATH, 'project':TRAIN_FOLDER}
        check_keys['data'] = False
    elif hyp['mode'] == 'export':
        tmp = {'model':path_wts}
        check_keys['model'] = True
    elif hyp['mode'] == 'predict':
        tmp = {'model':path_wts, 'source':SOURCE_PATH, 'project':PREDICT_FOLDER}
        check_keys['source'] = False
        check_keys['model'] = True
    elif hyp['mode'] == 'val':
        tmp = {'data':DATASET_PATH, 'model':path_wts, 'project':VAL_FOLDER}
        check_keys['data'] = False
        check_keys['model'] = True
    else:
        raise Exception(f"Not support the mode: {hyp['mode']}. All supported modes are: train, val, predict, export.")
    defaults.update(tmp)
    add_configs(hyp, defaults)
    
    # error checking
    sanity_check(hyp, check_keys)
    
    # get final command
    final_cmd = ['yolo'] + [f'{k}={v}' for k, v in hyp.items()]
    logger.info(f'cmd: {final_cmd}')
    
    # run final command
    subprocess.run(final_cmd, check=True)
    
