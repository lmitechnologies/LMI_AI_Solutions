import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import anomalib.models as ad_models
import yaml
from anomalib.data import Folder
from .models import TolerantAnomalyDINO  # noqa: F401 — registers class into anomalib.models

ad_models.TolerantAnomalyDINO = TolerantAnomalyDINO
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from torchvision.transforms import v2

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively converts list values in params to tuples (e.g. [224, 224] -> (224, 224))."""
    cleaned = {}
    for k, v in params.items():
        if isinstance(v, list):
            cleaned[k] = tuple(v)
        elif isinstance(v, dict):
            cleaned[k] = clean_params(v)
        else:
            cleaned[k] = v
    return cleaned


def build_augmentations(aug_config: Optional[List[Dict]]) -> Optional[v2.Compose]:
    """Dynamically builds the augmentation pipeline from a list of dicts."""
    if not aug_config:
        return None

    transforms_list = []
    for aug_item in aug_config:
        name = aug_item.get("class_name")
        params = aug_item.get("params", {}) or {}
        params = clean_params(params)

        try:
            if not hasattr(v2, name):
                logger.warning(f"Transform '{name}' not found in torchvision.transforms.v2. Skipping.")
                continue

            transform_class = getattr(v2, name)
            transforms_list.append(transform_class(**params))
            logger.info(f"Added augmentation: {name}")

        except Exception as e:
            logger.error(f"Error adding augmentation '{name}': {e}")

    return v2.Compose(transforms_list) if transforms_list else None


def build_model(model_config: Dict[str, Any]):
    """
    Dynamically builds the Anomalib model.
    Extracts 'image_size' for pre-processing but removes it before model init.
    """
    class_name = model_config.get("class_name")
    params = model_config.get("params", {}) or {}

    # Clean params (convert lists to tuples)
    params = clean_params(params)

    # 1. Get the model class dynamically
    if not hasattr(ad_models, class_name):
        raise ValueError(f"Model '{class_name}' not found in anomalib.models")

    model_class = getattr(ad_models, class_name)
    logger.info(f"Initializing Model: {class_name}")

    # 2. Extract and REMOVE image_size
    image_size = params.pop("image_size", None)

    # 3. Configure Pre-processor (if applicable) using the extracted image_size
    if hasattr(model_class, "configure_pre_processor") and image_size is not None:
        if "pre_processor" not in params:
            logger.info(f"Auto-configuring pre-processor for {class_name} with size {image_size}...")
            pre_processor = model_class.configure_pre_processor(image_size=image_size)
            params["pre_processor"] = pre_processor

    # 4. Instantiate the model
    try:
        model = model_class(**params)
        return model
    except TypeError as e:
        logger.error(f"Error initializing {class_name}: {e}")
        logger.error(f"Parameters provided: {list(params.keys())}")
        raise e


def build_data(data_config: Dict[str, Any]) -> Folder:
    if data_config.get("train_augmentations", None) is not None:
        data_config["train_augmentations"] = build_augmentations(data_config["train_augmentations"])
    elif data_config.get("val_augmentations", None) is not None:
        data_config["val_augmentations"] = build_augmentations(data_config["val_augmentations"])
    elif data_config.get("augmentations", None) is not None:
        data_config["augmentations"] = build_augmentations(data_config["augmentations"])
    return Folder(**data_config)


def get_image_size(model) -> Optional[tuple]:
    """Extracts image size from model's pre-processor if available."""
    for t in model.pre_processor.transform.transforms:
        if type(t).__name__ == "Resize":
            return t.size
    return None


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train Anomalib Model from YAML config")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- Build Model Dynamically ---
    model = build_model(cfg["model"])

    # --- Data Module Setup ---
    data_cfg = cfg["data"]

    datamodule = build_data(data_cfg)

    # --- Engine Setup ---
    eng_cfg = cfg["engine"]
    engine = Engine(
        max_epochs=eng_cfg["max_epochs"],
        accelerator=eng_cfg["accelerator"],
        devices=eng_cfg["devices"],
        default_root_dir=Path(eng_cfg["default_root_dir"]),
    )

    # --- Train ---
    logger.info("Starting training...")
    engine.fit(model=model, datamodule=datamodule)

    # --- Export to Torch---
    engine.export(model=model, export_type=ExportType.TORCH)

    # --- Export to ONNX---
    engine.export(model=model, export_type=ExportType.ONNX, input_size=get_image_size(model))


if __name__ == "__main__":
    main()
