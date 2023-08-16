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


if __name__=='__main__':
    # check if files exist
    if not os.path.isfile(HYP_YAML):
        raise FileNotFoundError(f'{HYP_YAML} not found')
    if not os.path.isfile(DATA_YAML):
        raise FileNotFoundError(f'{DATA_YAML} not found')
    
    # load hyp yaml file
    with open(HYP_YAML) as f:
        hyp = yaml.safe_load(f)
    # convert to list of paried strings, such as ['batch=64', 'epochs=100']
    cmd2 = [f'{k}={v}' for k, v in hyp.items()]
    
    # get the final training cmd
    today = date.today().strftime("%Y-%m-%d")   # use today's date as the training folder name
    cmd = ['yolo', f'data={DATA_YAML}', f'project={TRAIN_FOLDER}', f'name={today}']
    cmd += cmd2
    logger.info(f'cmd: {cmd}')
    
    # run training
    subprocess.run(cmd, check=True)
    