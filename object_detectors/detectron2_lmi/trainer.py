from detectron2.engine import DefaultTrainer
import os
from detectron2_lmi.utils.det_utils import create_config, register_datasets
from detectron2.data import build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.utils.logger import setup_logger
import sys
import signal
import yaml
import glob
from detectron2.data import DatasetMapper


logger = setup_logger()


def build_augmentations(cfg, augmentations=None):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )

    if augmentations:
        if augmentations.get("BRIGHTNESS", None):
            augs.append(T.RandomBrightness(augmentations["BRIGHTNESS"]["MIN"], augmentations["BRIGHTNESS"]["MAX"]))
        if augmentations.get("FLIP_HORIZONTAL", None):
            augs.append(
                T.RandomFlip(
                    prob=augmentations["FLIP_HORIZONTAL"]["PROB"],
                    horizontal=True,
                    vertical=False,
                )
            )
        if augmentations.get("FLIP_VERTICAL", None):
            augs.append(
                T.RandomFlip(
                    prob=augmentations["FLIP_VERTICAL"]["PROB"],
                    horizontal=False,
                    vertical=True,
                )
            )
        if augmentations.get("ROTATION", None):
            augs.append(
                T.RandomRotation(
                    angle=[
                        augmentations["ROTATION"]["MIN"],
                        augmentations["ROTATION"]["MAX"],
                    ]
                )
            )
        if augmentations.get("LIGHTING", None):
            augs.append(T.RandomLighting(augmentations["LIGHTING"]["SCALE"]))
        if augmentations.get("CONTRAST", None):
            augs.append(T.RandomContrast(augmentations["CONTRAST"]["MIN"], augmentations["CONTRAST"]["MAX"]))
        if augmentations.get("SATURATION", None):
            augs.append(T.RandomSaturation(augmentations["SATURATION"]["MIN"], augmentations["SATURATION"]["MAX"]))

    return augs


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, "augmentations.yaml")):
            augmentations = yaml.safe_load(open(os.path.join(cfg.OUTPUT_DIR, "augmentations.yaml"), "r"))
            mapper = DatasetMapper(
                cfg,
                is_train=True,
                augmentations=build_augmentations(cfg=cfg, augmentations=augmentations),
            )
        else:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_augmentations(cfg=cfg))
        return build_detection_train_loader(cfg, mapper=mapper)


def train_model(cfg):
    """
    Train the model using the given configuration
    """
    trainer = Trainer(cfg)
    # trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg.OUTPUT_DIR


def training_run(args):
    # start tensorboard
    os.system("pkill -f tensorboard")
    pid = os.fork()
    if pid == 0:
        os.setsid()
        os.system(f"tensorboard --logdir {args.get('output')} --port 6006")
        sys.exit(0)
    else:
        logger.info(f"Tensorboard started with PID {pid}")

    config_file = args.get("config_file")
    cfg, _ = create_config(config_file, args.get("detectron2_config"), output_dir=args.get("output"))
    logger.info(f"Dataset Directory: {args.get('dataset_dir')}")
    logger.info(f"Output Directory: {cfg.OUTPUT_DIR}")
    # register the datasets train, test

    for dataset_name in cfg.DATASETS.TRAIN:
        register_datasets(
            dataset_dir=os.path.join(args.get("dataset_dir"), f"{dataset_name}"),
            dataset_name=dataset_name,
        )
        if not os.path.isfile(os.path.join(cfg.OUTPUT_DIR, "sample_image.png")):
            images_in_folder = glob.glob(os.path.join(args.get("dataset_dir"), f"{dataset_name}/images/*"))
            os.system(
                f"cp {os.path.join(args.get('dataset_dir'), f'{dataset_name}/images/{os.path.basename(images_in_folder[-1])}')} {cfg.OUTPUT_DIR}/sample_image.png"
            )
        logger.info(f"registered dataset: {dataset_name}")

    if len(cfg.DATASETS.TEST) > 0:
        register_datasets(
            dataset_dir=os.path.join(args.get("dataset_dir"), cfg.DATASETS.TEST[0]),
            dataset_name=cfg.DATASETS.TEST[0],
        )

    logger.info("Starting training run")

    train_model(cfg)
    # kill tensorboard
    os.kill(pid, signal.SIGTERM)

    # update the config file with the output directory
    # load the yaml file in the output directory
    config = yaml.safe_load(open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "r"))
    config["OUTPUT_DIR"] = cfg.OUTPUT_DIR  # update the output directory
    config["MODEL"]["WEIGHTS"] = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # update the weights path
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        yaml.dump(config, f)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config-file", type=str, help="Path to the config file", default="/home/config.yaml")
#     parser.add_argument("--detectron2-config", type=str, help="Detectron2 config file", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     parser.add_argument("--dataset-dir", type=str, help="Dataset dir", default="/home/data")
#     parser.add_argument("--output-dir", type=str, help="Path to the output directory", default="/home/weights/")
#     args = parser.parse_args()
#     main(args=args)
