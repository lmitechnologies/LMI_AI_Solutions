from detectron2.engine import DefaultTrainer
import argparse
import os
from .utils import register_datasets, create_config


def train_model(cfg):
    """
    Train the model using the given configuration
    """
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def main(args):
    config_file = args.config_file

    cfg = create_config(config_file)
    # register the datasets train, test
    for dataset_name, dataset_path in zip(cfg.DATASETS.TRAIN, cfg.DATASETS.TRAIN_DIR):
        register_datasets(dataset_name, dataset_path)
    for dataset_name, dataset_path in zip(cfg.DATASETS.TEST, cfg.DATASETS.TEST_DIR):
        register_datasets(dataset_name, dataset_path)
    # train the model
    train_model(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, help="Path to the config file")
    args = parser.parse_args()
    main()