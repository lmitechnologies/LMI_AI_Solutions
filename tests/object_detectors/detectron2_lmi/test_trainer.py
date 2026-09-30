import glob
import os

import detectron2.data.transforms as T
import pytest
import yaml
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog

from object_detectors.detectron2_lmi import trainer
from object_detectors.detectron2_lmi.trainer import build_config, train_mapper_args

CONFIG_DIR = "object_detectors/detectron2_lmi/configs"
FASTER_RCNN = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


def _write(tmp_path, config):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


def _types(augs):
    return [type(a) for a in augs]


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))), ids=os.path.basename)
def test_shipped_configs_build(path):
    cfg, _ = build_config(path)
    assert cfg.MODEL.WEIGHTS.startswith("https://")


def test_model_config_file_picks_the_base_config_and_the_cli_overrides_it(tmp_path):
    path = _write(tmp_path, {"MODEL_CONFIG_FILE": FASTER_RCNN})
    assert not build_config(path)[0].MODEL.MASK_ON
    assert build_config(path, base_config=trainer.DEFAULT_BASE_CONFIG)[0].MODEL.MASK_ON


def test_config_file_overrides_the_base_config(tmp_path):
    cfg, _ = build_config(_write(tmp_path, {"MODEL": {"ROI_HEADS": {"NUM_CLASSES": 3}}, "SOLVER": {"MAX_ITER": 7}}))
    assert (cfg.MODEL.ROI_HEADS.NUM_CLASSES, cfg.SOLVER.MAX_ITER) == (3, 7)


def test_default_training_augmentations_follow_detectron2(tmp_path):
    cfg, extras = build_config(_write(tmp_path, {"INPUT": {"RANDOM_FLIP": "horizontal", "CROP": {"ENABLED": True}}}))
    assert extras == []
    assert _types(train_mapper_args(cfg, extras)["augmentations"]) == [T.RandomCrop, T.ResizeShortestEdge, T.RandomFlip]


def test_extra_augmentations_are_appended_and_flips_replace_random_flip(tmp_path):
    augmentations = {
        "BRIGHTNESS": {"MIN": 0.9, "MAX": 1.1},
        "FLIP_VERTICAL": {"PROB": 0.3},
        "CONTRAST": {"MIN": 0.9, "MAX": 1.1},
    }
    cfg, extras = build_config(_write(tmp_path, {"INPUT": {"RANDOM_FLIP": "horizontal"}, "AUGMENTATIONS": augmentations}))
    assert cfg.INPUT.RANDOM_FLIP == "none"
    augs = train_mapper_args(cfg, extras)["augmentations"]
    assert _types(augs) == [T.ResizeShortestEdge, T.RandomBrightness, T.RandomFlip, T.RandomContrast]
    assert (augs[2].prob, augs[2].horizontal, augs[2].vertical) == (0.3, False, True)


def test_unknown_augmentation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="BLUR"):
        build_config(_write(tmp_path, {"AUGMENTATIONS": {"BLUR": {"SIGMA": 1}}}))


def test_training_run_writes_what_convert_reads(tmp_path, monkeypatch):
    for name in ("my_train", "my_test"):
        os.makedirs(tmp_path / "data" / name / "images")
        (tmp_path / "data" / name / "annotations.json").write_text('{"images": [], "annotations": [], "categories": []}')
    os.makedirs(trainer.next_output_dir(str(tmp_path / "out")))
    trained = []
    monkeypatch.setattr(trainer.shutil, "which", lambda _: None)
    monkeypatch.setattr(trainer, "Trainer", lambda cfg, extras: trained.append(cfg) or _NoTrain())
    config = _write(tmp_path, {"DATASETS": {"TRAIN": ["my_train"], "TEST": ["my_test"]}})
    try:
        trainer.training_run({"config_file": config, "output": str(tmp_path / "out"), "dataset_dir": str(tmp_path / "data")})
    finally:
        for name in ("my_train", "my_test"):
            DatasetCatalog.remove(name)

    out_dir = trained[0].OUTPUT_DIR
    assert out_dir.endswith("-v2"), "an existing run folder must not be reused"
    assert trained[0].MODEL.WEIGHTS.startswith("https://"), "training starts from the pretrained weights"
    saved = get_cfg()
    saved.merge_from_file(os.path.join(out_dir, "config.yaml"))
    assert saved.MODEL.WEIGHTS == os.path.join(out_dir, "model_final.pth")


class _NoTrain:
    def resume_or_load(self, resume):
        pass

    def train(self):
        pass
