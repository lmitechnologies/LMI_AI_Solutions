import subprocess
import os


def trtexec(onnx_dir:str, **kwargs) -> None:
    engine_dir = onnx_dir.replace(".onnx", f".engine")
    
    # Base trtexec command
    trt_command = " ".join([
        "trtexec",
            f"--onnx={onnx_dir}",
            f"--saveEngine={engine_dir}",
            f"--memPoolSize=workspace:4096 --fp16",
            f"--useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10",
            f"{'--verbose' if kwargs.get('verbose', False) else ''}"])
    
    if kwargs.get('profile', False):
        profile_dir = onnx_dir.replace(".onnx", f".nsys-rep")
        # Wrap with nsys profile command
        command = " ".join([
            "nsys profile",
                f"--output={profile_dir}",
                "--trace=cuda,nvtx",
                "--force-overwrite true",
                trt_command
        ])
        print(f'Profile data will be saved to: {profile_dir}')
    else:
        command = trt_command

    output = run_command_shell(command, kwargs.get('dry_run', False))

def run_command_shell(command, dry_run:bool = False) -> int:
    if dry_run:
        print("")
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} {command}")
        print("")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr.decode('utf-8')}")
        raise

def convert_to_tensorrt(onnx_path:str, **kwargs) -> None:
    """
    Convert an ONNX model to TensorRT engine.

    Args:
        onnx_path (str): Path to the ONNX model file.
        **kwargs: Additional keyword arguments for conversion options.
    """
    trtexec(onnx_path, **kwargs)

def convert_to_onnx(model, output_dir:str, **kwargs) -> None:
    model.export(output_dir=output_dir, opset_version=kwargs.get('opset_version', 17))