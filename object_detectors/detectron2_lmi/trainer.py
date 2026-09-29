import os
import shutil
import subprocess
from datetime import date

import detectron2.data.transforms as T
import yaml
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

logger = setup_logger()

DEFAULT_BASE_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# AUGMENTATIONS keys in our training YAML, applied after detectron2's own resize, crop and flip
EXTRA_AUGMENTATIONS = {
    "BRIGHTNESS": lambda a: T.RandomBrightness(a["MIN"], a["MAX"]),
    "FLIP_HORIZONTAL": lambda a: T.RandomFlip(prob=a["PROB"], horizontal=True, vertical=False),
    "FLIP_VERTICAL": lambda a: T.RandomFlip(prob=a["PROB"], horizontal=False, vertical=True),
    "ROTATION": lambda a: T.RandomRotation(angle=[a["MIN"], a["MAX"]]),
    "LIGHTING": lambda a: T.RandomLighting(a["SCALE"]),
    "CONTRAST": lambda a: T.RandomContrast(a["MIN"], a["MAX"]),
    "SATURATION": lambda a: T.RandomSaturation(a["MIN"], a["MAX"]),
}


def build_config(config_file, base_config=None):
    """Build the detectron2 config and extra augmentations from our training YAML.

    The YAML holds detectron2 keys plus ``AUGMENTATIONS`` and ``MODEL_CONFIG_FILE`` (the model-zoo base config, used
    when ``base_config`` is None). Flips in ``AUGMENTATIONS`` replace detectron2's ``INPUT.RANDOM_FLIP``.
    """
    with open(config_file) as f:
        overrides = yaml.safe_load(f) or {}
    augmentations = overrides.pop("AUGMENTATIONS", None) or {}
    file_base_config = overrides.pop("MODEL_CONFIG_FILE", None)
    base_config = base_config or file_base_config or DEFAULT_BASE_CONFIG
    unknown = set(augmentations) - set(EXTRA_AUGMENTATIONS)
    if unknown:
        raise ValueError(f"Unknown AUGMENTATIONS {sorted(unknown)}; supported: {list(EXTRA_AUGMENTATIONS)}")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_config))
    cfg.merge_from_other_cfg(CfgNode(overrides))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config)
    if {"FLIP_HORIZONTAL", "FLIP_VERTICAL"} & set(augmentations):
        cfg.INPUT.RANDOM_FLIP = "none"
    extras = [build(augmentations[k]) for k, build in EXTRA_AUGMENTATIONS.items() if k in augmentations]
    return cfg, extras


def train_mapper_args(cfg, extra_augmentations):
    """detectron2's training DatasetMapper arguments with our extra augmentations appended."""
    mapper_args = DatasetMapper.from_config(cfg, is_train=True)
    mapper_args["augmentations"] = mapper_args["augmentations"] + list(extra_augmentations)
    return mapper_args


class Trainer(DefaultTrainer):
    def __init__(self, cfg, extra_augmentations=()):
        # set before super().__init__, which calls build_train_loader
        self.extra_augmentations = list(extra_augmentations)
        super().__init__(cfg)

    def build_train_loader(self, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(**train_mapper_args(cfg, self.extra_augmentations)))


def register_dataset(dataset_dir, name):
    annotations, images = os.path.join(dataset_dir, "annotations.json"), os.path.join(dataset_dir, "images")
    if not (os.path.isfile(annotations) and os.path.isdir(images)):
        raise ValueError(f"Dataset {name}: {dataset_dir} needs annotations.json and an images folder")
    register_coco_instances(name, {}, annotations, images)


def next_output_dir(output_dir):
    version = 1
    while os.path.exists(os.path.join(output_dir, f"{date.today()}-v{version}")):
        version += 1
    return os.path.join(output_dir, f"{date.today()}-v{version}")


def training_run(args):
    cfg, extras = build_config(args["config_file"], args.get("detectron2_config"))
    cfg.OUTPUT_DIR = next_output_dir(args["output"])
    os.makedirs(cfg.OUTPUT_DIR)
    logger.info(f"Output directory: {cfg.OUTPUT_DIR}")

    for name in (*cfg.DATASETS.TRAIN, *cfg.DATASETS.TEST):
        register_dataset(os.path.join(args["dataset_dir"], name), name)

    # the config convert reads: the trainer keeps using the pretrained weights in memory
    final = cfg.clone()
    final.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(yaml.dump(yaml.safe_load(final.dump())))

    tensorboard = None
    if shutil.which("tensorboard"):
        tensorboard = subprocess.Popen(["tensorboard", "--logdir", args["output"], "--port", "6006"])
    else:
        logger.warning("tensorboard not found; training without it")
    try:
        trainer = Trainer(cfg, extras)
        trainer.resume_or_load(resume=False)
        trainer.train()
    finally:
        if tensorboard:
            tensorboard.terminate()
