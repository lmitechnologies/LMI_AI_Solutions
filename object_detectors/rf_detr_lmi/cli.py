import argparse
import logging
import os
from datetime import date
from typing import Any, Dict

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

from object_detectors.rf_detr_lmi.convert import convert_to_onnx, convert_to_tensorrt

logger = logging.getLogger(__name__)

# Constants
OPERATION_TRAIN = "train"
OPERATION_CONVERT = "convert"
TASK_OD = "od"
TASK_SEGMENTATION = "seg"
FORMAT_ONNX = "onnx"
FORMAT_TENSORRT = "tensorrt"
FORMAT_TORCHSCRIPT = "torchscript"

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


def setup_argparser() -> argparse.ArgumentParser:
    """Set up the argument parser for training configuration.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Train RF-DETR-LMI Object Detector")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the YAML configuration file.")
    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration parameters.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


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
        "model_type": config.get("model_type", "medium"),
        "operation": config.get("operation", OPERATION_TRAIN),
        "task": config.get("task", TASK_OD),
    }
    training_configs = config.get("training", {})
    conversion_configs = config.get("conversion", {})
    format = config.get("format")

    operation = model_configs["operation"]
    if operation == OPERATION_TRAIN:
        validate_training_config(training_configs)
    elif operation == OPERATION_CONVERT:
        validate_conversion_config(conversion_configs, format)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    return {
        "model_configs": model_configs,
        "training_configs": training_configs,
        "conversion_configs": conversion_configs,
        "format": format,
    }


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
    model_type = model_configs.get("model_type", "medium")
    operation = model_configs.get("operation", OPERATION_TRAIN)

    # Look up model class from registry
    model_key = (task, model_type)
    model_class = MODEL_REGISTRY.get(model_key)

    if model_class is None:
        raise ValueError(f"Unsupported combination: task={task}, model_type={model_type}")

    # Instantiate model based on operation
    if operation == OPERATION_TRAIN:
        return model_class()
    elif operation == OPERATION_CONVERT:
        conversion_configs = configs.get("conversion_configs", {})
        if not conversion_configs:
            raise ValueError("Conversion configuration is missing.")
        return model_class(**conversion_configs)
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

    # Create versioned output directory
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
    output_dir = conversion_configs.get("output_dir")
    if output_dir:
        return output_dir

    # Fallback: use the directory of pretrain_weights
    pretrain_weights = conversion_configs.get("pretrain_weights", "")
    return os.path.dirname(pretrain_weights) if pretrain_weights else "."


def convert_model_to_torchscript(model: Any, output_dir: str) -> None:
    """Convert model to TorchScript format.

    Args:
        model: The model instance to convert.
        output_dir: Directory to save the converted model.
    """
    logger.info("Optimizing model for TorchScript conversion...")
    model.optimize_for_inference()
    output_path = os.path.join(output_dir, "model.pt")
    model.model.inference_model.save(output_path)
    logger.info(f"TorchScript model saved to: {output_path}")


def convert_model_to_onnx(model: Any, output_dir: str) -> None:
    """Convert model to ONNX format.

    Args:
        model: The model instance to convert.
        output_dir: Directory to save the converted model.
    """
    logger.info("Starting model conversion to ONNX format...")
    convert_to_onnx(model, output_dir=output_dir)
    logger.info(f"ONNX model saved to: {output_dir}")


def convert_model_to_tensorrt(model: Any, output_dir: str) -> None:
    """Convert model to TensorRT format.

    Args:
        model: The model instance to convert.
        output_dir: Directory to save the converted model.
    """
    onnx_model_path = os.path.join(output_dir, "inference_model.onnx")

    # Convert to ONNX first if not already done
    if not os.path.exists(onnx_model_path):
        logger.info("ONNX model not found. Converting to ONNX format first...")
        convert_to_onnx(model, output_dir=output_dir)

    logger.info("Converting to TensorRT engine...")
    convert_to_tensorrt(onnx_model_path)
    logger.info(f"TensorRT model saved to: {output_dir}")


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
        FORMAT_TORCHSCRIPT: convert_model_to_torchscript,
        FORMAT_ONNX: convert_model_to_onnx,
        FORMAT_TENSORRT: convert_model_to_tensorrt,
    }

    handler = conversion_handlers.get(format)
    if handler is None:
        raise ValueError(f"Unsupported conversion format: {format}")

    handler(model, output_dir)


def main() -> None:
    """Main entry point for the CLI application."""
    logging.basicConfig(level=logging.INFO)
    parser = setup_argparser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        configs = parse_config(config)

        operation = configs.get("model_configs", {}).get("operation")

        if operation == OPERATION_TRAIN:
            initiate_training(configs)
        elif operation == OPERATION_CONVERT:
            handle_conversion(configs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
