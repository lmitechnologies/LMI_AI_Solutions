import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import anomalib.models as ad_models
import yaml
from anomalib.data import Folder
from anomalib.engine import Engine
from torchvision.transforms import v2


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
                print(f"Warning: Transform '{name}' not found in torchvision.transforms.v2. Skipping.")
                continue

            transform_class = getattr(v2, name)
            transforms_list.append(transform_class(**params))
            print(f"Added augmentation: {name}")

        except Exception as e:
            print(f"Error adding augmentation '{name}': {e}")

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
    print(f"Initializing Model: {class_name}")

    # 2. Extract and REMOVE image_size
    # We use .pop() so it is NOT passed to the model's __init__
    image_size = params.pop("image_size", None)

    # 3. Configure Pre-processor (if applicable) using the extracted image_size
    # PaDiM and others need this. If the user didn't provide image_size, we skip this.
    if hasattr(model_class, "configure_pre_processor") and image_size is not None:
        if "pre_processor" not in params:
            print(f"Auto-configuring pre-processor for {class_name} with size {image_size}...")
            pre_processor = model_class.configure_pre_processor(image_size=image_size)
            params["pre_processor"] = pre_processor

    # 4. Instantiate the model
    # params now contains 'pre_processor' (if added) but explicitly excludes 'image_size'
    try:
        model = model_class(**params)
        return model
    except TypeError as e:
        print(f"Error initializing {class_name}: {e}")
        print(f"Parameters provided: {list(params.keys())}")
        raise e


def build_data(data_config: Dict[str, Any]) -> Folder:
    if data_config.get("train_augmentations", None) is not None:
        data_config["train_augmentations"] = build_augmentations(data_config["train_augmentations"])
    elif data_config.get("val_augmentations", None) is not None:
        data_config["val_augmentations"] = build_augmentations(data_config["val_augmentations"])
    elif data_config.get("augmentations", None) is not None:
        data_config["augmentations"] = build_augmentations(data_config["augmentations"])
    return Folder(**data_config)


def main():
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
    print("Starting training...")
    engine.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
