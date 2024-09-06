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
    dataset_dir = args.dataset_dir
    config_file = args.config_file
    # check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")
    # get the training and test data directories
    train_data_dir = os.path.join(dataset_dir, "train")
    val_data_dir = os.path.join(dataset_dir, "val")
    test_data_dir = os.path.join(dataset_dir, "test")
    # register the datasets
    register_datasets(train_data_dir, val_data_dir, test_data_dir)
    # create the config
    cfg = create_config(config_file)
    # train the model
    train_model(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--config-file", type=str, help="Path to the config file")
    args = parser.parse_args()
    main()
