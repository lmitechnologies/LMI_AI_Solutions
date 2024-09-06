from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
import argparse
import os
from datetime import date
import yaml
from typing import Any, Callable, Dict, IO, List, Union


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

def register_datasets(train_data_dir, val_data_dir, test_data_dir):
    """
    Register the train and test datasets with Detectron2

    :param train_data_dir: Path to the directory containing the training data
    :param test_data_dir: Path to the directory containing the test data
    """
    if os.path.exists(train_data_dir):
        json_path = os.path.join(train_data_dir, "train.json")
        images_path = os.path.join(train_data_dir, "images")
        if os.path.isfile(json_path) and os.path.exists(images_path):
            register_coco_instances(
                "train_dataset",
                {},
                json_path,
                images_path,
            )
    if os.path.exists(val_data_dir):
        json_path = os.path.join(val_data_dir, "val.json")
        images_path = os.path.join(val_data_dir, "images")
        if os.path.isfile(json_path) and os.path.exists(images_path):
            register_coco_instances(
                "val_dataset",
                {},
                json_path,
                images_path,
            )
    if os.path.exists(test_data_dir):
        json_path = os.path.join(test_data_dir, "test.json")
        images_path = os.path.join(test_data_dir, "images")
        if os.path.isfile(json_path) and os.path.exists(images_path):
            register_coco_instances(
                "test_dataset",
                {},
                json_path,
                images_path,
            )

def create_config(cfg_file_path):
    # get the default config
    cfg = get_cfg()

    # load the cfg file
    cfg_file_path = os.path.abspath(cfg_file_path)
    if not os.path.exists(cfg_file_path):
        raise ValueError(f"Config file {cfg_file_path} does not exist")

    configuration = yaml.safe_load(open(cfg_file_path, "r"))
    # get the model configuration to use
    detectron2_yaml = configuration.get("MODEL_CONFIG_FILE", None)
    if detectron2_yaml is None:
        raise ValueError("MODEL_CONFIG_FILE not specified in the config file")
    
    # load the config from the file
    cfg.merge_from_file(model_zoo.get_config_file(detectron2_yaml))

    # override the values in the config file
    merge_a_into_b(configuration, cfg)

    # set the model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detectron2_yaml)

    version = 1
    output_dir = f"{date.today()}-v{version}"
    # determine the version number based on the existing directories
    while os.path.exists(output_dir):
        version += 1
        output_dir = f"{date.today()}-v{version}"

    cfg.OUTPUT_DIR = output_dir

    # create the output directory if it does not exist

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the config to the output directory
    cfg_file = os.path.join(output_dir, "config.yaml")
    with open(cfg_file, "w") as f:
        f.write(yaml.dump(yaml.safe_load(cfg.dump())))
    return cfg


