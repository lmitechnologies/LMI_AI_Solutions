import subprocess
from datetime import date
import logging
import yaml
import os
from urllib.request import urlretrieve

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

YOLO_REPO_HYP_URL = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/hyps/hyp.scratch-low.yaml'
YOLO_ROOT = '/repos/LMI_AI_Solutions/object_detectors/submodules/yolov5'
YOLO_SEG_ROOT = YOLO_ROOT + '/segment'
REPLACE_KEYS = {'model':'weights','batch':'batch-size','exist_ok':'exist-ok','conf':'conf-thres','iou':'iou-thres',
                'show':'view-img','save_txt':'save-txt','save_conf':'save-conf','save_crop':'save-crop',
                'vid_stride':'vid-stride','line_width':'line-thickness','agnostic_nms':'agnostic-nms',
                }
NEG_KEYS = {'show_labels':'hide-labels','show_conf':'hide-conf'}
DEFAULT_KEYS = {'conf-thres':0.25,'line-thickness':2}
HYP_KEYS = ['lr0','lrf','momentum','weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','box',
            'cls','cls_pw','obj','obj_pw','iou_t','anchor_t','fl_gamma','hsv_h','hsv_s','hsv_v',
            'degrees','translate','scale','shear','perspective','flipud','fliplr','mosaic','mixup','copy_paste']
REMOVE_KEYS = ['mode','task','retina_masks'] + HYP_KEYS
NO_VAL_KEYS = ['rect','resume','nosave','noval','exist-ok','view-img','save-txt','save-csv','save-conf','save-crop',
               'agnostic-nms','augment','visualize','update','hide-labels','hide-conf','half','dnn']

# mounted locations in the docker container
HYP_YAML = '/app/config/hyp.yaml'
AUG_YAML = '/app/temp/augment.yaml'
REPO_YAML = '/app/config/repo_hyp.yaml'

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
    # download repo's yaml file
    urlretrieve(YOLO_REPO_HYP_URL, REPO_YAML)
    
    # check if paths exist
    check_path_exist(HYP_YAML, True)
    check_path_exist(REPO_YAML, True)
    check_path_exist(YOLO_ROOT, False)
    
    # load repo hyp yaml file
    with open(REPO_YAML) as f:
        hyp = yaml.safe_load(f)
        
    # load hyp yaml file
    with open(HYP_YAML) as f:
        temp_hyp = yaml.safe_load(f)
        
    # update hyp with temp_hyp
    hyp.update(temp_hyp)
        
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
        if 'imgsz' in hyp:
            hyp['imgsz'] = hyp['imgsz'].split(',')
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
            
    # negate keys
    for k,v in NEG_KEYS.items():
        if k in hyp:
            hyp[v] = not hyp.pop(k)
            
    # assign default vals to keys if they are None
    for k,v in DEFAULT_KEYS.items():
        if k in hyp and hyp[k] is None:
            hyp[k] = v
    
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
        yaml.dump(hyp2, f, sort_keys=False)
        logger.info(f'created hyp.yaml: {hyp2}')
    hyp['hyp'] = AUG_YAML
            
    # remove keys
    for k in REMOVE_KEYS:
        if k in hyp:
            hyp.pop(k)
            
    l = []
    # modify keys with no values
    for k in NO_VAL_KEYS:
        if k in hyp:
            if hyp[k]==True:
                l.append(f'--{k}')
            hyp.pop(k)
            
    # special cases
    if 'classes' in hyp:
        if hyp['classes'] is not None:
            hyp['classes'] = hyp['classes'].split(',')
        else:
            hyp.pop('classes')
            
    # get final command
    for k,v in hyp.items():
        l.append(f'--{k}')
        if isinstance(v, list) or isinstance(v, tuple):
            for v2 in v:
                l.append(v2)
        else:
            l.append(f'{v}')
    final_cmd = ['python3', target_file] + l

    logger.info(f'cmd: {final_cmd}')
    
    # run final command
    subprocess.run(final_cmd, check=True)
