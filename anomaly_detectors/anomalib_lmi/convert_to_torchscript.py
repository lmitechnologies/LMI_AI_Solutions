import logging 
import torch
import torch.nn as nn
from anomalib_lmi.base import to_list
from anomalib_lmi.anomaly_model2 import AnomalyModel2
import argparse
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
        self.register_buffer('mean', mean.view(1, -1, 1, 1))
        self.register_buffer('std', std.view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std
    
    
def make_preprocessing_trace_safe(module, device):
    """
    Recursively searches for standard torchvision Normalize layers 
    and replaces them with SafeNormalize.
    """
    # Handle standard nn.Sequential or nn.Module
    for name, child in module.named_children():            
        # Handle torchvision.transforms.Compose specifically (if it holds a list)
        if isinstance(child, v2.Compose):
            new_transforms = []
            for t in child.transforms:
                if "Normalize" in t.__class__.__name__:
                    logger.info(f"Detected unsafe Normalize in Compose list")
                    safe_norm = SafeNormalize(t.mean, t.std).to(device)
                    new_transforms.append(safe_norm)
                    logger.info(f" -> Replaced with SafeNormalize")
                else:
                    new_transforms.append(t)
            child.transforms = new_transforms
    return module


def generate_traced_torchscript(model_path, output_path, version='v1', batch_size=1):
    """
    Generate a traced TorchScript model from the given model path.
    
    Args:
        model_path (str): Path to the model file.
        output_path (str): Path to save the converted model.
        version (str): Version of the model. Default is 'v1'.
        batch_size (int): Batch size for tracing. Default is 1.
        
    Returns:
        torch.jit.ScriptModule: The traced TorchScript model.
    """
    if version == 'v1':
        return convert_v1_torchscript(model_path=model_path, output_path=output_path, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported version: {version}")

def convert_v1_torchscript(model_path, output_path, batch_size=1, device='cuda'):
    """
    Convert a model to TorchScript format.
    
    Args:
        model_path (str): Path to the model file.
        output_path (str): Path to save the converted model.
        batch_size (int): Batch size for tracing. Default is 1.
        device (str): Device to use for tracing. Default is 'cuda'.
        
    Returns:
        torch.jit.ScriptModule: The converted TorchScript model.
    """
    logger.info(f"Converting {model_path} to TorchScript format.")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = ckpt['model'].eval()
    model = make_preprocessing_trace_safe(model, device=device)
    image_size = None
    for d in model.transform.transforms:
        if isinstance(d, v2.Resize):
            image_size = to_list(d.size)
    image_size = [image_size[0]+1, image_size[1]+1]
    inp = torch.rand(batch_size,3,image_size[0], image_size[1]).to(device)
    traced_model = torch.jit.trace(model,inp,strict=False)
    torch.jit.save(traced_model, output_path)
    logger.info(f"Saved traced model to {output_path}")
    
    # Verify the traced model
    model = AnomalyModel2(output_path)
    inp = torch.randint(0,255,(image_size[0], image_size[1],3),dtype=torch.uint8).to(device)
    model.predict(inp)
    logger.info(f"Verified traced model by running a prediction.")
    
    return traced_model

def main():
    parser = argparse.ArgumentParser(description="Convert a model to TorchScript format.")
    parser.add_argument('--input_path', '-i', type=str, required=True, help='Path to the model file.')
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Path to save the converted model.')
    parser.add_argument('--version', type=str, default='v1', help='Version of the model. Default is v1.',required=False, choices=['v1'])
    parser.add_argument('--batch_size', type=int, default=1, help='Export batch size. Default is 1.', required=False)
    
    args = parser.parse_args()
    generate_traced_torchscript(args.input_path, args.output_path, args.version, args.batch_size)
    
    
if __name__ == "__main__":
    main()
