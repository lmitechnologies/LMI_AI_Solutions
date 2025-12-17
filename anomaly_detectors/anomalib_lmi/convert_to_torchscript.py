import logging 
import torch
from .base import to_list
import argparse
from torchvision.transforms import v2

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def generate_traced_torchscript(model_path,output_path,version='v1', batch_size=1):
    """
    Generate a traced TorchScript model from the given model path.
    
    Args:
        model_path (str): Path to the model file.
        version (str): Version of the model. Default is 'v1'.
        
    Returns:
        torch.jit.ScriptModule: The traced TorchScript model.
    """
    if version == 'v1':
        return convert_v1_torchscript(model_path=model_path, output_path=output_path, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported version: {version}")

def convert_v1_torchscript(model_path, output_path, batch_size=1):
    """
    Convert a model to TorchScript format.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        torch.jit.ScriptModule: The converted TorchScript model.
    """
    logger.info(f"Converting {model_path} to TorchScript format.")
    ckpt = torch.load(model_path, weights_only=False)
    model = ckpt['model'].eval().cuda()
    image_size = None
    for d in model.transform.transforms:
        if isinstance(d, v2.Resize):
            image_size = to_list(d.size)
    image_size = [image_size[0]+1, image_size[1]+1]
    inp = torch.rand(batch_size,3,image_size[0], image_size[1]).cuda()
    traced_model = torch.jit.trace(model,inp,strict=False)
    torch.jit.save(traced_model, output_path)
    logger.info(f"Saved traced model to {output_path}")
    return traced_model

def main():
    parser = argparse.ArgumentParser(description="Convert a model to TorchScript format.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the model file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the converted model.')
    parser.add_argument('--version', type=str, default='v1', help='Version of the model. Default is v1.',required=False, choices=['v1'])
    parser.add_argument('--batch_size', type=int, default=1, help='Export batch size. Default is 1.', required=False)
    
    args = parser.parse_args()
    
    generate_traced_torchscript(args.input_path, args.output_path, args.version, args.batch_size)

if __name__ == "__main__":
    main()
