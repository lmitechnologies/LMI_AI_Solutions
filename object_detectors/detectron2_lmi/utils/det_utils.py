from detectron2 import model_zoo
from detectron2.config import get_cfg
import os
from datetime import date
import yaml
from typing import Any, Dict
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances


logger = setup_logger()

def merge_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    """based from fvcore/common/config.py"""
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(b[k], dict), "Cannot inherit key '{}' from base!".format(
                k
            )
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


def register_datasets(dataset_dir: str, dataset_name: str):
    """
    Register the train and test datasets with Detectron2
    """
    if os.path.exists(dataset_dir):
        annot_file = os.path.join(dataset_dir, "annotations.json")
        images_path = os.path.join(dataset_dir, "images")
        if os.path.isfile(annot_file) and os.path.isdir(images_path):
            logger.info(f"Registering dataset {dataset_name} from {dataset_dir}")
            register_coco_instances(
                dataset_name,
                {},
                annot_file,
                images_path,
            )
        else:
            raise ValueError(f"Invalid dataset directory {dataset_dir} for dataset {dataset_name}")


def create_config(cfg_file_path, detectron2_config_file, output_dir):
    
    version = 1
    output_save_dir = os.path.join(output_dir, f"{date.today()}-v{version}")
    # determine the version number based on the existing directories
    while os.path.exists(output_save_dir):
        version += 1
        output_save_dir = os.path.join(output_dir, f"{date.today()}-v{version}")
        # create the output directory if it does not exist

    if not os.path.exists(output_save_dir):
        os.makedirs(output_save_dir)
    # get the default config
    cfg = get_cfg()

    # load the cfg file
    cfg_file_path = os.path.abspath(cfg_file_path)
    if not os.path.exists(cfg_file_path):
        raise ValueError(f"Config file {cfg_file_path} does not exist")

    configuration = yaml.safe_load(open(cfg_file_path, "r"))
    print(f"Configuration: {configuration}")
    # get the model configuration to use
    # load the config from the file
    # remove augmentations from the config
    
    augmentations = configuration.get("AUGMENTATIONS", None)
    hyperparameters_yaml_file  = cfg_file_path
    if augmentations:
        
        augmentations_file = os.path.join(output_save_dir, "augmentations.yaml")
        logger.info(f"Saving augmentations to {augmentations_file}")
        with open(augmentations_file, "w") as f:
            yaml.dump(augmentations, f)
        
        del configuration["AUGMENTATIONS"]
        # save the config file
        hyperparameters_yaml_file = os.path.join(os.path.dirname(cfg_file_path), 'hyp.yaml')
        with open(hyperparameters_yaml_file, "w") as f:
            yaml.dump(configuration, f)
        


    cfg.merge_from_file(model_zoo.get_config_file(detectron2_config_file))
    cfg.merge_from_file(hyperparameters_yaml_file)

    # set the model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detectron2_config_file)

    
    cfg.OUTPUT_DIR = output_save_dir
    logger.info(f"Output directory: {cfg.OUTPUT_DIR}")


    # save the config to the output directory
    cfg_file = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(cfg_file, "w") as f:
        f.write(yaml.dump(yaml.safe_load(cfg.dump())))
    return cfg, configuration
