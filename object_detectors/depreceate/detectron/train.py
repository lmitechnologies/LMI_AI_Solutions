from numpy.core.fromnumeric import shape
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
    

def register_datasets(cfg):
    #register train data
    for dataset_name, dataset_path in zip(cfg.DATASETS.TRAIN, cfg.DATASETS.TRAIN_DIR):
        path_json_labels = os.path.join(dataset_path,"labels.json")
        if not os.path.isfile(path_json_labels):
            raise Exception("cannot find the annotation file labels.json in {}".format(dataset_path))
        register_coco_instances(dataset_name, {}, path_json_labels, dataset_path)

    

def train(cfg, resume=True):
    from detectron2.engine import DefaultTrainer
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, type=str, help='the input yaml file')
    ap.add_argument('--base_yaml', default="mask_rcnn_R_50_C4_1x.yaml", help='the base yaml file provided in https://github.com/facebookresearch/detectron2/tree/main/configs/COCO-InstanceSegmentation')
    args = vars(ap.parse_args())

    cfg = get_cfg()
    #create new customized keys in config file
    cfg.DATASETS.TRAIN_DIR = ()

    base_yaml = args['base_yaml']
    print(f'base yaml file: {base_yaml}')
    #load yaml files
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+base_yaml))
    cfg.merge_from_file(args['input'])
    #register train and test datasets
    register_datasets(cfg)
    #training
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+args['base_yaml'])
    train(cfg, resume=True)