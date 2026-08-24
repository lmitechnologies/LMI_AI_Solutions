import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import anomalib.models as ad_models
import torch
import yaml
from anomalib.data import Folder
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor

# import before anomalib to avoid partial-init circular import
from torchvision.transforms import v2

from .tiling import TilerConfigCallback

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
            if not isinstance(name, str):
                logger.warning("Transform class_name must be a string. Skipping.")
                continue

            if not hasattr(v2, name):
                logger.warning(f"Transform '{name}' not found in torchvision.transforms.v2. Skipping.")
                continue

            transform_class = getattr(v2, name)
            transforms_list.append(transform_class(**params))
            logger.info(f"Added augmentation: {name}")

        except Exception as e:
            logger.error(f"Error adding augmentation '{name}': {e}")

    return v2.Compose(transforms_list) if transforms_list else None


def build_preprocessor(preprocessor_config: Optional[List[Dict]]) -> PreProcessor:
    """Build an Anomalib preprocessor from torchvision v2 transform settings."""
    transforms = build_augmentations(preprocessor_config)
    return PreProcessor(transform=transforms)


def build_model(model_config: Dict[str, Any]):
    """
    Dynamically builds the Anomalib model.
    Extracts 'image_size' for pre-processing but removes it before model init.
    """
    class_name = model_config.get("class_name")
    params = model_config.get("params", {}) or {}
    for rm in ["tiler_type", "tile_size", "stride", "precision"]:
        params.pop(rm, None)

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
    if "pre_processor" not in params:
        if hasattr(model_class, "configure_pre_processor") and image_size is not None:
            logger.info(f"Auto-configuring pre-processor for {class_name} with size {image_size}...")
            pre_processor = model_class.configure_pre_processor(image_size=image_size)
            params["pre_processor"] = pre_processor
    else:
        params["pre_processor"] = build_preprocessor(params["pre_processor"])

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


def build_tiler(tile_size, stride, tiler_cls_name=None):
    if tile_size is None:
        return []
    return [TilerConfigCallback(enable=True, tile_size=tile_size, stride=stride, tiler_class=tiler_cls_name)]


def estimate_max_samples(config_path, num_samples=None):
    # Run in a subprocess so the model/CUDA context built for profiling is fully
    # released on process exit, leaving no lingering VRAM before engine.fit().
    cmd = [
        sys.executable,
        "-m",
        "anomaly_detectors.anomalib_lmi.v2.memory_estimation.estimate",
        "-i",
        config_path,
    ]
    if num_samples:
        cmd.extend(["-n", str(num_samples)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"Could not estimate training memory:\n{result.stderr.strip()}")
        return None, None

    max_images, peak_mib = None, None
    for line in result.stdout.strip().splitlines():
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        try:
            if key == "max_train_images":
                max_images = None if value == "None" else int(value)
            elif key == "estimated_peak_mib":
                peak_mib = float(value)
        except ValueError:
            continue

    if max_images is None and peak_mib is None:
        logger.warning(f"Unexpected estimate output:\n{result.stdout.strip()}")
    return max_images, peak_mib


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train Anomalib Model from YAML config")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--skip-mem-estimate", action="store_true", help="Run memory estimation before training")
    parser.add_argument("--ckpt-path", type=Path, help="Path to a Lightning checkpoint to resume")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- Build Model Dynamically ---
    model_params = cfg["model"]["params"]
    tiler_cls_name = model_params.pop("tiler_type", None)
    tiler_callbacks = build_tiler(model_params.pop("tile_size", None), model_params.pop("stride", None), tiler_cls_name=tiler_cls_name)
    model = build_model(cfg["model"])

    # --- Data Module Setup ---
    data_cfg = cfg["data"]

    datamodule = build_data(data_cfg)

    # --- Engine Setup ---
    eng_cfg = cfg["engine"]
    engine = Engine(
        max_epochs=eng_cfg["max_epochs"],
        accelerator=eng_cfg.get("accelerator", "gpu"),
        devices=eng_cfg["devices"],
        default_root_dir=Path(eng_cfg["default_root_dir"]),
        callbacks=tiler_callbacks,
    )

    if not args.skip_mem_estimate:
        max_dataset_size, estimated_peak_mib = estimate_max_samples(args.config)
        logger.info(f"Approximated max dataset size: {max_dataset_size or 'Could not estimate max dataset size'}")
        logger.info(
            "Estimated training memory: %s",
            f"{estimated_peak_mib:.2f} MiB" if estimated_peak_mib is not None else "Could not estimate training memory",
        )

    # --- Train ---
    logger.info("Starting training...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    engine.fit(model=model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    checkpoint_path = Path(eng_cfg["default_root_dir"]) / "model.ckpt"
    engine.trainer.save_checkpoint(checkpoint_path)
    logger.info(
        "cuda max allocated MiB: %.2f",
        torch.cuda.max_memory_allocated() / 1024**2,
    )
    logger.info(
        "cuda max reserved MiB: %.2f",
        torch.cuda.max_memory_reserved() / 1024**2,
    )

    inner = getattr(model, "model", model)

    for name in ["memory_bank", "embedding", "embeddings"]:
        if hasattr(inner, name):
            value = getattr(inner, name)
            if isinstance(value, torch.Tensor):
                logger.info(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")

    # --- Export to Torch---
    engine.export(model=model, export_type=ExportType.TORCH)

    # --- Export to ONNX---
    # Avoid unsupported data type (half precision) issues (ex. reflection_pad2d)
    model = model.float()

    def export_engine(external_data=False):
        onnx_kwargs = {"external_data": external_data}
        engine.export(model=model, export_type=ExportType.ONNX, input_size=get_image_size(model), onnx_kwargs=onnx_kwargs)

    try:
        export_engine()
    except RuntimeError as e:
        if "larger than 2GiB limit" in str(e):
            # Retry export with external data
            logger.info("2GiB onnx export limit exceeded, export will include additional files.")
            export_engine(external_data=True)


if __name__ == "__main__":
    main()
