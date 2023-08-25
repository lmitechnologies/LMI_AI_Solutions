import argparse
import subprocess
from datetime import date
import logging
import yaml
import os

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# mounted locations in the docker container
HYP_YAML = '/app/config/hyp.yaml'

# default configs
DATA_YAML = '/app/config/dataset.yaml'
TRAIN_FOLDER = '/app/training'
VAL_FOLDER = '/app/validation'
MODEL_PATH = '/app/trained-inference-models'
MODEL_NAMES = ['best.engine','best.pt']
SOURCE_PATH = '/app/data'


def check_file_exist(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found')
    
def sanity_check(final_configs:dict, check_keys:list):
    """check if the value to the check_keys exists. If not, throw exception.

    Args:
        final_configs (dict): the input configs
        check_keys (list): keys to be checked
    """
    for k in check_keys:
        check_file_exist(final_configs[k])
    
def get_model_path(path, mode):
    # if export mode, use 'best.pt'. 
    # otherwise:
    #   use 'best.engine' if it exists. use 'best.pt' if does not exist
    names = MODEL_NAMES[1:] if mode=='export' else MODEL_NAMES
    for fname in names:
        p = os.path.join(path, fname)
        if os.path.isfile(p):
            logger.info(f'Use the model weights: {p}')
            return p
    return

def add_configs(final_configs:dict, configs:dict):
    """add to configs only if the configs do NOT exist. Modify the final_configs in-place.

    Args:
        final_configs (dict): the output configs
        configs (dict): the configs to be added
    """
    for k,v in configs.items():
        if k not in final_configs:
            logger.info(f'Not found the config: {k}. Use the default: {v}')
            final_configs[k] = v



if __name__=='__main__':
    # check if files exist
    check_file_exist(HYP_YAML)
        
    # load hyp yaml file
    with open(HYP_YAML) as f:
        hyp = yaml.safe_load(f)
       
    # use today's date as the default output folder name
    defaults = {'name':date.today().strftime("%Y-%m-%d")}
    
    # add other default configs
    check_keys = []
    if hyp['mode'] == 'train':
        tmp = {'data':DATA_YAML, 'project':TRAIN_FOLDER}
        check_keys += ['data']
    elif hyp['mode'] in ['predict','export']:
        path = get_model_path(MODEL_PATH, hyp['mode']) # get the default model path
        tmp = {'model':path, 'source':SOURCE_PATH, 'project':VAL_FOLDER}
        check_keys += ['model', 'source'] if hyp['mode']=='predict' else ['model']
    else:
        raise Exception(f"Not support the mode: {hyp['mode']}. All supported modes are: train, predict, export")
    defaults.update(tmp)
    add_configs(hyp, defaults)
    
    # error checking
    sanity_check(hyp, check_keys)
    
    # get final command
    final_cmd = ['yolo'] + [f'{k}={v}' for k, v in hyp.items()]
    logger.info(f'cmd: {final_cmd}')
    
    # run final command
    subprocess.run(final_cmd, check=True)
    