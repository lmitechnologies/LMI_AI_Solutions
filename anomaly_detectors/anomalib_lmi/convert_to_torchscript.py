import argparse
import logging

import torch
import torch.nn as nn
from anomalib_lmi.base import to_list
from torchvision.transforms import v2

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SafeNormalize(nn.Module):
    """
    A Normalize layer that stores mean/std as buffers.
    This ensures they are saved with the model and automatically
    move to the correct device (CPU/GPU) without hardcoding.
    """

    def __init__(self, mean, std):
        super().__init__()
        # Ensure they are tensors
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)

        # Reshape to [1, C, 1, 1] to allow broadcasting over [B, C, H, W]
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def make_preprocessing_trace_safe(module, device):
    """
    Searches for torchvision Normalize layers and replaces them with SafeNormalize.
    """
    for _name, child in module.named_children():
        if isinstance(child, v2.Compose):
            new_transforms = []
            for t in child.transforms:
                if "Normalize" in t.__class__.__name__:
                    logger.info(f"Detected unsafe Normalize: {t}")
                    safe_norm = SafeNormalize(t.mean, t.std).to(device)
                    new_transforms.append(safe_norm)
                    logger.info(" -> Replaced with SafeNormalize")
                else:
                    new_transforms.append(t)
            child.transforms = new_transforms
    return module


def generate_traced_torchscript(model_path, output_path, device, version="v1", batch_size=1):
    """
    Generate a traced TorchScript model from the given model path.

    Args:
        model_path (str): Path to the model file.
        output_path (str): Path to save the converted model.
        device (str): Device to use for tracing.
        version (str): Version of the model. Default is 'v1'.
        batch_size (int): Batch size for tracing. Default is 1.

    Returns:
        torch.jit.ScriptModule: The traced TorchScript model.
    """
    if version == "v1":
        return convert_v1_torchscript(model_path, output_path, device, batch_size)
    else:
        raise ValueError(f"Unsupported version: {version}")


def convert_v1_torchscript(model_path, output_path, device, batch_size=1):
    """
    Convert a model to TorchScript format.

    Args:
        model_path (str): Path to the model file.
        output_path (str): Path to save the converted model.
        device (str): Device to use for tracing.
        batch_size (int): Batch size for tracing. Default is 1.

    Returns:
        torch.jit.ScriptModule: The converted TorchScript model.
    """
    logger.info(f"Converting {model_path} to TorchScript format on {device}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = ckpt["model"].eval()
    model = make_preprocessing_trace_safe(model, device=device)

    # Determine image size from model transforms
    image_size = None
    for d in model.transform.transforms:
        if isinstance(d, v2.Resize):
            image_size = to_list(d.size)
    image_size = [image_size[0] + 1, image_size[1] + 1]

    # trace the model
    inp = torch.rand(batch_size, 3, image_size[0], image_size[1]).to(device)
    traced_model = torch.jit.trace(model, inp, strict=False)
    torch.jit.save(traced_model, output_path)
    logger.info(f"Saved traced model to {output_path}")

    return traced_model


def main():
    parser = argparse.ArgumentParser(description="Convert a model to TorchScript format.")
    parser.add_argument("--input_path", "-i", type=str, required=True, help="Path to the model file.")
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
        help="Path to save the converted model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use for tracing.",
        required=True,
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version of the model. Default is v1.",
        required=False,
        choices=["v1"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Export batch size. Default is 1.",
        required=False,
    )

    args = parser.parse_args()
    generate_traced_torchscript(args.input_path, args.output_path, args.device, args.version, args.batch_size)


if __name__ == "__main__":
    main()
