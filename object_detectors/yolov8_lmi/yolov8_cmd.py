import subprocess
from datetime import date
import logging
import yaml
import os

logging.basicConfig()



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
        Yolov8_Cmd.check_path_exist(final_configs[k],v)
    

def get_model_path(path, mode):
    # if export mode, use 'best.pt'. 
    # otherwise:
    #   use 'best.engine' if it exists. otherwise use 'best.pt'
    # return None if not found any model weights
    names = Yolov8_Cmd.MODEL_NAMES[1:] if mode=='export' else Yolov8_Cmd.MODEL_NAMES
    for fname in names:
        p = os.path.join(path, fname)
        if os.path.isfile(p):
            Yolov8_Cmd.logger.info(f'Use the model weights: {p}')
            return p
    return None


def add_configs(final_configs:dict, configs:dict):
    """add to configs only if the configs do NOT exist. Modify the final_configs in-place.

    Args:
        final_configs (dict): the output configs
        configs (dict): the configs to be added
    """
    for k,v in configs.items():
        if k not in final_configs:
            Yolov8_Cmd.logger.info(f'Not found the config: {k}. Use the default: {v}')
            final_configs[k] = v



class Yolov8_Cmd:
    logger = logging.getLogger('yolov8_cmd')
    logger.setLevel(logging.INFO)
    
    # mounted locations in the docker container
    HYP_YAML = '/app/config/hyp.yaml'

    # default configs
    DATA_YAML = '/app/config/dataset.yaml'
    TRAIN_FOLDER = '/app/training'
    VAL_FOLDER = '/app/validation'
    PREDICT_FOLDER = '/app/prediction'
    MODEL_PATH = '/app/trained-inference-models'
    MODEL_NAMES = ['best.engine','best.pt']
    SOURCE_PATH = '/app/data'
    
    
    def __init__(self, hyp_yaml=None) -> None:
        if hyp_yaml is not None:
            self.HYP_YAML = hyp_yaml
        self.hyp = {}
        # check if configs are valid
        self.check_keys = {} # map < config key : True ckeck if it's a file else check if it's a dir >

    
    def load_hyp_yaml(self):
        # check if files exist
        check_path_exist(self.HYP_YAML, True)
            
        # load hyp yaml file
        with open(self.HYP_YAML) as f:
            hyp = yaml.safe_load(f)
        return hyp
    
    
    def update_hyp(self, hyp):
        """
        update the hyp with default configs if the key does not exist
        """
        # use today's date as the default output folder name
        defaults = {'name':date.today().strftime("%Y-%m-%d")}
        
        # add other default configs
        path_wts = get_model_path(self.MODEL_PATH, hyp['mode'])
        if hyp['mode'] == 'train':
            tmp = {'data':self.DATA_YAML, 'project':self.TRAIN_FOLDER}
            self.check_keys['data'] = True
        elif hyp['mode'] == 'export':
            tmp = {'model':path_wts}
            self.check_keys['model'] = True
        elif hyp['mode'] == 'predict':
            tmp = {'model':path_wts, 'source':self.SOURCE_PATH, 'project':self.PREDICT_FOLDER}
            self.check_keys['source'] = False
            self.check_keys['model'] = True
        elif hyp['mode'] == 'val':
            tmp = {'data':self.DATA_YAML, 'model':path_wts, 'project':self.VAL_FOLDER}
            self.check_keys['data'] = True
            self.check_keys['model'] = True
        else:
            raise Exception(f"Not support the mode: {hyp['mode']}. All supported modes are: train, val, predict, export.")
        defaults.update(tmp)
        add_configs(hyp, defaults)
    
    
    def run(self):
        """
        Run the yolo command with the configs from the hyp yaml file
        """
        # load hyp yaml file
        self.hyp = self.load_hyp_yaml()
        
        # update hyp with default configs
        self.update_hyp(self.hyp)
        
        # error checking
        sanity_check(self.hyp, self.check_keys)
        
        # get final command
        final_cmd = ['yolo'] + [f'{k}={v}' for k, v in self.hyp.items()]
        self.logger.info(f'cmd: {final_cmd}')
        
        # run final command
        subprocess.run(final_cmd, check=True)
    
    
