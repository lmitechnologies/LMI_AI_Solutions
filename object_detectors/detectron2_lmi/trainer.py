from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
import argparse
import os
from utils.det_utils import create_config
from detectron2.utils.logger import setup_logger

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

def train_model(cfg):
    """
    Train the model using the given configuration
    """
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def main(args):
    config_file = args.config_file

    cfg, original_config = create_config(config_file)
    # register the datasets train, test
    for dataset_name, dataset_path in zip(original_config['DATASETS']["TRAIN"], original_config['DATASETS']["TRAIN_DIR"]):
        register_datasets(dataset_name, dataset_path)
    for dataset_name, dataset_path in zip(original_config['DATASETS']["TEST"], original_config['DATASETS']["TEST_DIR"]):
        register_datasets(dataset_name, dataset_path)
    logger.info("Starting training run")
    train_model(cfg)
    logger.info("Training run completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, help="Path to the config file")
    args = parser.parse_args()
    main(args=args)