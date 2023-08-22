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
DATA_YAML = '/app/config/dataset.yaml'

# default configs
TRAIN_FOLDER = '/app/training'
VAL_FOLDER = '/app/validation'
MODEL_PATH = '/app/trained-inference-models'
MODEL_NAMES = ['best.engine','best.pt']
SOURCE_PATH = '/app/data'


def check_file_exist(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found')
    
def check_folder_exist(path):
    if not os.path.isdir(path):
        raise Exception(f'path not exist: {path}')
    
def get_model_path(path):
    for fname in MODEL_NAMES:
        p = os.path.join(path, fname)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'Not found "best.pt" or "best.engine" in {path}')

def add_cmd(final_cmds:list, cmd:str):
    idx = cmd.find('=')
    key = cmd[:idx+1]
    for c in final_cmds:
        if c.find(key)!=-1:
            logger.info(f'Found {key} in both hyp.yaml and default configs. Overwrite the default config')
            return
    final_cmds.append(cmd)

def add_cmds(final_cmds:list, cmds:list):
    for cmd in cmds:
        add_cmd(final_cmds, cmd)



if __name__=='__main__':
    # check if files exist
    check_file_exist(HYP_YAML)
        
    # load hyp yaml file
    with open(HYP_YAML) as f:
        hyp = yaml.safe_load(f)
    # convert to list of paried strings, such as ['batch=64', 'epochs=100']
    hyp_cmd = [f'{k}={v}' for k, v in hyp.items()]
    
    # check if dataset yaml file exists in the train mode
    is_train = hyp['mode']=='train'
    is_predict = hyp['mode']=='predict'
    if is_train:
        check_file_exist(DATA_YAML)
    
    # get the final cmd
    today = date.today().strftime("%Y-%m-%d")   # use today's date as the output folder name
    cmd = ['yolo', f'name={today}'] + hyp_cmd
    
    # add default cmds if NOT exist in hyp.yaml 
    if is_train:
        tmp_cmd = [f'data={DATA_YAML}', f'project={TRAIN_FOLDER}']
        add_cmds(cmd,tmp_cmd)
    else:
        check_folder_exist(MODEL_PATH)
        path = get_model_path(MODEL_PATH)
        tmp_cmd = [f'model={path}', f'source={SOURCE_PATH}', f'project={VAL_FOLDER}']
        add_cmds(cmd,tmp_cmd)
    
    logger.info(f'cmd: {cmd}')
    
    # run command
    subprocess.run(cmd, check=True)
    