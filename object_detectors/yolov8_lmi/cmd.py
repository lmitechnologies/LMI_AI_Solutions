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
TRAIN_FOLDER = '/app/training'
# for predict and export
VAL_FOLDER = '/app/validation'
MODEL_PATH = '/app/trained-inference-models/best.pt'    
SOURCE_PATH = '/app/data'


def check_file_exist(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found')


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
    cmd = ['yolo', f'name={today}']
    if is_train:
        cmd += [f'data={DATA_YAML}', f'project={TRAIN_FOLDER}']
    else:
        check_file_exist(MODEL_PATH)
        cmd += [f'model={MODEL_PATH}', f'source={SOURCE_PATH}', f'project={VAL_FOLDER}']
    cmd += hyp_cmd
    
    logger.info(f'cmd: {cmd}')
    
    # run command
    subprocess.run(cmd, check=True)
    