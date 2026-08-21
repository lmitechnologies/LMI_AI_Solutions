import argparse
import logging
import os
from datetime import date
from typing import Any, Dict, List

import yaml

try:
    from rfdetr import (
        RFDETRLarge,
        RFDETRMedium,
        RFDETRNano,
        RFDETRSeg2XLarge,
        RFDETRSegLarge,
        RFDETRSegMedium,
        RFDETRSegNano,
        RFDETRSegSmall,
        RFDETRSegXLarge,
        RFDETRSmall,
    )
except ImportError as e:
    logging.error(f"Failed to import rfdetr models: {e}")
    raise

from object_detectors.rf_detr_lmi.checkpoint import load_from_checkpoint
from object_detectors.rf_detr_lmi.convert import convert_to_onnx, convert_to_tensorrt

logger = logging.getLogger(__name__)

# Constants
OPERATION_TRAIN = "train"
OPERATION_CONVERT = "convert"
OPERATION_EXPORT = "export"
TASK_OD = "od"
TASK_SEGMENTATION = "seg"
FORMAT_ONNX = "onnx"
FORMAT_TENSORRT = "tensorrt"

# Model registry: maps (task, model_type) -> model class
MODEL_REGISTRY = {
    (TASK_OD, "nano"): RFDETRNano,
    (TASK_OD, "small"): RFDETRSmall,
    (TASK_OD, "medium"): RFDETRMedium,
    (TASK_OD, "large"): RFDETRLarge,
    # (TASK_OD, "xlarge"): RFDETRXLarge,
    # (TASK_OD, "2xlarge"): RFDETR2XLarge,
    (TASK_SEGMENTATION, "nano"): RFDETRSegNano,
    (TASK_SEGMENTATION, "small"): RFDETRSegSmall,
    (TASK_SEGMENTATION, "medium"): RFDETRSegMedium,
    (TASK_SEGMENTATION, "large"): RFDETRSegLarge,
    (TASK_SEGMENTATION, "xlarge"): RFDETRSegXLarge,
    (TASK_SEGMENTATION, "2xlarge"): RFDETRSeg2XLarge,
}


DEFAULT_MODEL_TYPE = "small"


def supported_model_types(task: str) -> List[str]:
    """Model sizes registered for a task."""
    return sorted(m for t, m in MODEL_REGISTRY if t == task)


def validate_training_config(training_configs: Dict[str, Any]) -> None:
    """Validate training configuration parameters.

    Args:
        training_configs: Training configuration dictionary.

    Raises:
        ValueError: If required training parameters are missing.
    """
    if not training_configs:
        raise ValueError("Training configuration is missing.")
    if "output_dir" not in training_configs:
        raise ValueError("output_dir must be specified in training configuration.")
    if "resolution" not in training_configs:
        raise ValueError("resolution must be specified in training configuration.")


def validate_export_config(export_configs: Dict[str, Any], model_configs: Dict[str, Any]) -> None:
    """Validate export configuration parameters.

    Args:
        export_configs: Export configuration dictionary.
        model_configs: Model configuration dictionary.

    Raises:
        ValueError: If required export parameters are missing.
    """
    if not export_configs:
        raise ValueError("Export configuration is missing.")
    if "output_dir" not in export_configs:
        raise ValueError("output_dir must be specified in export configuration.")
    if "pretrain_weights" not in model_configs:
        raise ValueError("pretrain_weights must be specified for export.")
    if "resolution" not in export_configs:
        raise ValueError(
            "resolution must be specified in export configuration; use the resolution the model was trained at, "
            "which training_config.json records under model_config."
        )


def validate_conversion_config(conversion_configs: Dict[str, Any], format: str) -> None:
    """Validate conversion configuration parameters.

    Args:
        conversion_configs: Conversion configuration dictionary.
        format: Target conversion format.

    Raises:
        ValueError: If required conversion parameters are missing.
    """
    if not format:
        raise ValueError("Conversion format must be specified.")
    if not conversion_configs:
        raise ValueError("Conversion configuration is missing.")
    if "resolution" not in conversion_configs:
        raise ValueError(
            "resolution must be specified in conversion configuration; use the resolution the model was trained at, "
            "which training_config.json records under model_config."
        )


def parse_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse configuration parameters.

    Args:
        config: Configuration parameters.

    Returns:
        Parsed training parameters.

    Raises:
        ValueError: If configuration is invalid or incomplete.
    """
    model_configs = {
        # Absent means the variant comes from the checkpoint (convert/export); training has no checkpoint
        # to read and falls back to DEFAULT_MODEL_TYPE.
        "model_type": config.get("model_type"),
        "operation": config.get("operation", OPERATION_TRAIN),
        "task": config.get("task", TASK_OD),
    }
    # pretrain_weights is tri-state: an absent key uses the variant's default pretrained
    # weights, an explicit null skips base-weight loading entirely (e.g. when training.resume
    # restores weights from a checkpoint), and a path warm-starts from that checkpoint.
    if "pretrain_weights" in config:
        model_configs["pretrain_weights"] = config["pretrain_weights"]
    # Explicit class count for the model head. When absent, training auto-detects it from the
    # dataset; a warm start passes it so the prior checkpoint loads into a matching head instead
    # of rfdetr's config default (which would warn while auto-aligning).
    if "num_classes" in config:
        model_configs["num_classes"] = config["num_classes"]
    training_configs = config.get("training", {})
    conversion_configs = config.get("conversion", {})
    export_configs = config.get("export", {})
    format = config.get("format")

    operation = model_configs["operation"]
    if operation == OPERATION_TRAIN:
        validate_training_config(training_configs)
    elif operation == OPERATION_CONVERT:
        validate_conversion_config(conversion_configs, format)
    elif operation == OPERATION_EXPORT:
        validate_export_config(export_configs, model_configs)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    return {
        "model_configs": model_configs,
        "training_configs": training_configs,
        "conversion_configs": conversion_configs,
        "export_configs": export_configs,
        "format": format,
    }


def get_model_class(task: str, model_type: str) -> Any:
    """Look up the RF-DETR model class for a task and model size.

    Args:
        task: Task name (od or seg).
        model_type: Model size (nano, small, medium, ...).

    Returns:
        The RF-DETR model class.

    Raises:
        ValueError: If the model type or task is unsupported.
    """
    model_class = MODEL_REGISTRY.get((task, model_type))
    if model_class is None:
        supported = ", ".join(f"{t}/{m}" for t, m in MODEL_REGISTRY)
        raise ValueError(f"Unsupported combination: task={task}, model_type={model_type}; supported: {supported}")
    return model_class


def load_pretrained_model(model_configs: Dict[str, Any], weights: str, **kwargs: Any) -> Any:
    """Load an existing checkpoint, reading its variant from the file unless the config names one.

    Args:
        model_configs: Parsed model configuration; its optional model_type overrides what the checkpoint records.
        weights: Path to the checkpoint.
        **kwargs: Forwarded to the rfdetr model constructor (resolution, device, ...).

    Returns:
        An rfdetr model instance.
    """
    task = model_configs.get("task", TASK_OD)
    model_type = model_configs.get("model_type")
    if model_type:
        return get_model_class(task, model_type)(pretrain_weights=weights, **kwargs)
    logger.info(f"Reading the model variant from {weights}")
    return load_from_checkpoint(weights, supported_model_types(task), **kwargs)


def load_model(configs: Dict[str, Any]) -> Any:
    """Load the RF-DETR model based on the configuration.

    Args:
        configs: Configuration parameters containing model_configs and conversion_configs.

    Returns:
        An instance of the RF-DETR model.

    Raises:
        ValueError: If the model type or task is unsupported.
    """
    model_configs = configs["model_configs"]
    task = model_configs.get("task", TASK_OD)
    operation = model_configs.get("operation", OPERATION_TRAIN)

    # Instantiate model based on operation
    if operation == OPERATION_TRAIN:
        # Training may start from scratch, so there is no checkpoint to read the variant from.
        model_class = get_model_class(task, model_configs.get("model_type") or DEFAULT_MODEL_TYPE)
        kwargs = {}
        if "pretrain_weights" in model_configs:
            kwargs["pretrain_weights"] = model_configs["pretrain_weights"]
        if "num_classes" in model_configs:
            kwargs["num_classes"] = model_configs["num_classes"]
        return model_class(**kwargs)
    elif operation == OPERATION_CONVERT:
        conversion_configs = dict(configs.get("conversion_configs", {}))
        if not conversion_configs:
            raise ValueError("Conversion configuration is missing.")
        weights = conversion_configs.pop("pretrain_weights", None)
        if weights is None:
            # No checkpoint to read from — convert the variant's own COCO-pretrained weights.
            return get_model_class(task, model_configs.get("model_type") or DEFAULT_MODEL_TYPE)(**conversion_configs)
        return load_pretrained_model(model_configs, weights, **conversion_configs)
    else:
        raise ValueError(f"Unsupported operation: {operation}")


def get_versioned_output_dir(base_output_dir: str) -> str:
    """Generate a versioned output directory path.

    Args:
        base_output_dir: Base output directory path.

    Returns:
        Versioned output directory path that doesn't exist yet.
    """
    version = 1
    today = date.today()
    output_dir = os.path.join(base_output_dir, f"{today}-v{version}")

    while os.path.exists(output_dir):
        version += 1
        output_dir = os.path.join(base_output_dir, f"{today}-v{version}")

    return output_dir


def initiate_training(configs: Dict[str, Any]) -> Any:
    """Initiate model training with the given configuration.

    Args:
        configs: Configuration parameters containing training_configs.

    Returns:
        Trained model instance.

    Raises:
        ValueError: If output_dir is not specified in training configuration.
    """
    model = load_model(configs)

    training_params = configs["training_configs"].copy()
    base_output_dir = training_params.get("output_dir")

    if base_output_dir is None:
        raise ValueError("output_dir must be specified in training configuration.")

    # By default each run writes to a fresh date-versioned subdirectory; versioned_output_dir: false
    # trains directly in output_dir (for callers that need a predictable path)
    if training_params.pop("versioned_output_dir", True):
        training_params["output_dir"] = get_versioned_output_dir(base_output_dir)

    logger.info(f"Starting training with output directory: {training_params['output_dir']}")
    model.train(**training_params)

    return model


def get_conversion_output_dir(conversion_configs: Dict[str, Any]) -> str:
    """Get or determine the output directory for model conversion.

    Args:
        conversion_configs: Conversion configuration dictionary.

    Returns:
        Output directory path for converted models.
    """
    output_dir = conversion_configs.pop("output_dir", None)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    # Fallback: use the directory of pretrain_weights
    pretrain_weights = conversion_configs.get("pretrain_weights", "")
    return os.path.dirname(pretrain_weights) if pretrain_weights else "."


def convert_model_to_onnx(model: Any, output_dir: str) -> str:
    """Convert model to ONNX format.

    Args:
        model: The model instance to convert.
        output_dir: Directory to save the converted model.

    Returns:
        Path to the exported ONNX file; rfdetr names it after the model variant.
    """
    logger.info("Starting model conversion to ONNX format...")
    onnx_path = convert_to_onnx(model, output_dir)
    logger.info(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def convert_model_to_tensorrt(model: Any, output_dir: str) -> None:
    """Convert model to TensorRT format, exporting the ONNX it is built from first.

    Args:
        model: The model instance to convert.
        output_dir: Directory to save the converted model.
    """
    onnx_path = convert_model_to_onnx(model, output_dir)
    logger.info("Converting to TensorRT engine...")
    engine_path = convert_to_tensorrt(onnx_path)
    logger.info(f"TensorRT model saved to: {engine_path}")


def handle_conversion(configs: Dict[str, Any]) -> None:
    """Handle model conversion based on the specified format.

    Args:
        configs: Configuration parameters containing conversion settings.

    Raises:
        ValueError: If the conversion format is unsupported.
    """
    output_dir = get_conversion_output_dir(configs.get("conversion_configs", {}))
    model = load_model(configs)
    format = configs.get("format")

    conversion_handlers = {
        FORMAT_ONNX: convert_model_to_onnx,
        FORMAT_TENSORRT: convert_model_to_tensorrt,
    }

    handler = conversion_handlers.get(format)
    if handler is None:
        raise ValueError(f"Unsupported conversion format: {format}")

    handler(model, output_dir)


def handle_export(configs: Dict[str, Any]) -> None:
    """Export a checkpoint to ONNX under the fixed filename model.onnx.

    Same export path as the convert operation, which keeps rfdetr's variant-named file and can also build a TensorRT engine.

    Args:
        configs: Configuration parameters containing model_configs and export_configs.

    Raises:
        ValueError: If required export parameters are missing.
    """
    model_configs = configs["model_configs"]
    export_params = configs["export_configs"].copy()
    output_dir = export_params.pop("output_dir")
    opset_version = export_params.pop("opset_version", 17)
    # rfdetr defaults verbose to True, making torch.onnx.export dump the entire graph
    verbose = export_params.pop("verbose", False)
    os.makedirs(output_dir, exist_ok=True)

    # Remaining export params (resolution, device, ...) are model constructor kwargs
    model = load_pretrained_model(model_configs, model_configs["pretrain_weights"], **export_params)

    # rfdetr names the exported file after the model variant (e.g. rfdetr-small.onnx);
    # stage it as model.onnx instead
    onnx_path = convert_to_onnx(model, output_dir, opset_version=opset_version, verbose=verbose)
    final_path = os.path.join(output_dir, "model.onnx")
    os.replace(onnx_path, final_path)
    logger.info(f"ONNX model exported to: {final_path}")


def main() -> None:
    """Main entry point for the CLI application."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="RF-DETR-LMI Object Detector")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    configs = parse_config(config)

    operation = configs.get("model_configs", {}).get("operation")
    if operation == OPERATION_TRAIN:
        initiate_training(configs)
    elif operation == OPERATION_CONVERT:
        handle_conversion(configs)
    elif operation == OPERATION_EXPORT:
        handle_export(configs)
    else:
        raise ValueError(f"Unsupported operation: {operation}")


if __name__ == "__main__":
    main()
