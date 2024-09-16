from detectron2.engine import DefaultTrainer
import argparse
import os
from utils.det_utils import create_config, register_datasets
from detectron2.utils.logger import setup_logger
import sys
import signal
import yaml

logger = setup_logger()

# TODO: Will be updated to use lauch for multi-distributed training
def train_model(cfg):
    """
    Train the model using the given configuration
    """
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg.OUTPUT_DIR


def main(args):
    # start tensorboard
    os.system("pkill -f tensorboard")
    pid = os.fork()
    if pid == 0:
        os.setsid()
        os.system(f"tensorboard --logdir {args.output_dir} --port 6006")
        sys.exit(0)
    else:
        logger.info(f"Tensorboard started with PID {pid}")
    
    config_file = args.config_file
    cfg, original_config = create_config(config_file, args.detectron2_config, output_dir=args.output_dir)
    logger.info(f"Dataset Directory: {args.dataset_dir}")
    logger.info(f"Output Directory: {cfg.OUTPUT_DIR}")
    # register the datasets train, test
    
    register_datasets(dataset_dir=os.path.join(args.dataset_dir, "train"), dataset_name=cfg.DATASETS.TRAIN[0])
    if len(cfg.DATASETS.TEST) > 0:
        register_datasets(dataset_dir=os.path.join(args.dataset_dir, "train"), dataset_name=cfg.DATASETS.TEST[0])
    
    logger.info("Starting training run")
    
    train_model(cfg)
    # kill tensorboard
    os.kill(pid, signal.SIGTERM)

    # update the config file with the output directory
    # load the yaml file in the output directory
    config = yaml.safe_load(open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "r"))
    config["OUTPUT_DIR"] = cfg.OUTPUT_DIR # update the output directory
    config["MODEL"]["WEIGHTS"] = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # update the weights path
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    logger.info("Training run completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, help="Path to the config file")
    parser.add_argument("--detectron2-config", type=str, help="Detectron2 config file", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--dataset-dir", type=str, help="Dataset dir", default="/home/data")
    parser.add_argument("--output-dir", type=str, help="Path to the output directory", default="/home/weights/")
    args = parser.parse_args()
    main(args=args)