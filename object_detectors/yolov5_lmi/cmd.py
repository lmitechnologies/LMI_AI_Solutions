import subprocess
from datetime import date
import logging
import yaml
import os
import subprocess

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


YOLO_ROOT = '/repos/LMI_AI_Solutions/object_detectors/submodules/yolov5'
YOLO_SEG_ROOT = YOLO_ROOT + '/segment'
REPLACE_KEYS = {'model':'weights','batch':'batch-size','exist_ok':'exist-ok'}
HYP_KEYS = ['degrees','translate','scale','shear','perspective','flipud','fliplr','mosaic','mixup','copy_paste']
REMOVE_KEYS = ['mode','task'] + HYP_KEYS
NO_VAL_KEYS = ['rect','exist-ok']

# mounted locations in the docker container
HYP_YAML = '/app/config/hyp.yaml'
AUG_YAML = '/app/temp/augment.yaml'

# default configs
DATA_YAML = '/app/config/dataset.yaml'
TRAIN_FOLDER = '/app/training'
VAL_FOLDER = '/app/validation'
MODEL_PATH = '/app/trained-inference-models'
MODEL_NAMES = ['best.engine','best.pt']
SOURCE_PATH = '/app/data'


def check_path_exist(path, is_file:bool):
    """fail the program if the path does not exist

    Args:
        path (str): the input path
        is_file (bool): True if it's a file, False otherwise
    """
    if is_file and not os.path.isfile(path):
        raise Exception(f'Not found file: {path}')
    if not is_file and not os.path.isdir(path):
        raise Exception(f'Not found path: {path}')
    
    
def sanity_check(final_configs:dict, check_keys:dict):
    """check if the value to the check_keys exists. If not, throw exception.

    Args:
        final_configs (dict): the input configs
        check_keys (dict): < key_to_be_checked : True if is_file else False >
    """
    for k,v in check_keys.items():
        check_path_exist(final_configs[k],v)
    
    
def get_model_path(path, mode):
    # if export mode, use 'best.pt'. 
    # otherwise:
    #   use 'best.engine' if it exists. otherwise use 'best.pt' 
    names = MODEL_NAMES[1:] if mode=='export' else MODEL_NAMES
    for fname in names:
        p = os.path.join(path, fname)
        if os.path.isfile(p):
            logger.info(f'Use the model weights: {p}')
            return p
    raise Exception(f'No found weights {MODEL_NAMES} in: {path}')


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
    # check if paths exist
    check_path_exist(HYP_YAML, True)
    check_path_exist(YOLO_ROOT, False)
        
    # load hyp yaml file
    with open(HYP_YAML) as f:
        hyp = yaml.safe_load(f)
        
    if hyp['task'] not in ['detect','segment']:
        raise Exception(f"Not support the task: {hyp['task']}. All supported tasks are: detect, segment.")
       
    # use today's date as the default output folder name
    defaults = {'name':date.today().strftime("%Y-%m-%d")}
    
    # add other default configs
    check_keys = {} # <key:True/False>. True: check if the key as a file exists. False: check if the key as a folder exists
    if hyp['mode'] == 'train':
        tmp = {'data':DATA_YAML, 'project':TRAIN_FOLDER}
        check_keys['data'] = True
    elif hyp['mode'] in ['predict','export']:
        file_path = os.path.join(YOLO_ROOT, f'{hyp["mode"]}.py')
        path = get_model_path(MODEL_PATH, hyp['mode']) # get the default model path
        tmp = {'model':path, 'source':SOURCE_PATH, 'project':VAL_FOLDER}
        check_keys['model']=True
        if hyp['mode']=='predict':
            check_keys['source']=False 
    else:
        raise Exception(f"Not support the mode: {hyp['mode']}. All supported modes are: train, predict, export")
    defaults.update(tmp)
    add_configs(hyp, defaults)
    
    # error checking
    sanity_check(hyp, check_keys)
    
    # replace keys from yolov8 to yolov5
    for k,v in REPLACE_KEYS.items():
        if k in hyp:
            hyp[v] = hyp.pop(k)
    
    # get the target file to run
    if hyp['mode'] == 'train':
        target_file = os.path.join(YOLO_ROOT if hyp['task']=='detect' else YOLO_SEG_ROOT,'train.py')
    elif hyp['mode'] == 'export':
        target_file = os.path.join(YOLO_ROOT, 'export.py')
    elif hyp['mode'] == 'predict':
        if hyp['task'] == 'segment':
            target_file = os.path.join(YOLO_SEG_ROOT, 'predict.py')
        else:
            target_file = os.path.join(YOLO_ROOT, 'detect.py')
            
    # create a new hyp.yaml file for HYP_KEYS
    os.makedirs(os.path.dirname(AUG_YAML), exist_ok=True)
    with open(AUG_YAML, 'w') as f:
        hyp2 = {k:v for k,v in hyp.items() if k in HYP_KEYS}
        yaml.dump(hyp2, f)
        logger.info(f'created {AUG_YAML}: {hyp2}')
    hyp['hyp'] = AUG_YAML
            
    # remove keys
    for k in REMOVE_KEYS:
        if k in hyp:
            hyp.pop(k)
            
    # modify keys with no values
    for k in NO_VAL_KEYS:
        if k in hyp:
            if hyp[k]==True:
                hyp[k] = ''
            else:
                hyp.pop(k)
            
    # get final command
    l = []
    for k,v in hyp.items():
        l.append(f'--{k}')
        l.append(f'{v}')
    final_cmd = ['python3', target_file] + l

    logger.info(f'cmd: {final_cmd}')
    
    # run final command
    subprocess.run(final_cmd, check=True)
    
