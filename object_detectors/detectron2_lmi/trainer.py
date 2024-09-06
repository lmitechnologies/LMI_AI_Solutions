from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
import argparse
import os
from utils.det_utils import create_config
from detectron2.utils.logger import setup_logger
import concurrent.futures
import sys
import subprocess

DETACHED_PROCESS = 0x00000008
logger = setup_logger()

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
    else:
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")

def train_model(cfg):
    """
    Train the model using the given configuration
    """
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg.OUTPUT_DIR

def main(args):
    config_file = args.config_file

    cfg, original_config = create_config(config_file)
    # register the datasets train, test
    for dataset_name, dataset_path in zip(original_config['DATASETS']["TRAIN"], original_config['DATASETS']["TRAIN_DIR"]):
        register_datasets(dataset_dir=dataset_path, dataset_name=dataset_name)
    for dataset_name, dataset_path in zip(original_config['DATASETS']["TEST"], original_config['DATASETS']["TEST_DIR"]):
        register_datasets(dataset_dir=dataset_path, dataset_name=dataset_name)
    logger.info("Starting training run")
    training_runs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        training_runs.append(executor.submit(train_model, cfg))
    
    # start tensorboard
    logger.info("Starting tensorboard")
    
    # start tensorboard in a separate process
    pid = os.fork()
    if pid == 0:
        os.setsid()
        os.system(f"tensorboard --logdir {cfg.OUTPUT_DIR} --port 6006")
        sys.exit(0)
    
    # wait for the training runs to complete
    for training_run in concurrent.futures.as_completed(training_runs):
        logger.info(f"Training run completed: {training_run.result()}")
        os.kill(pid, 9)
    logger.info("Training run completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, help="Path to the config file")
    args = parser.parse_args()
    main(args=args)