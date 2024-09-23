import os
import sys
import subprocess
import argparse

def convert_to_onnx(**kwargs):
    subprocess.run(f"python {os.path.join(os.path.dirname(os.path.realpath(__file__)), 'detectron2_lmi', 'converter/detectron2_to_onnx.py')}", shell=True)
    subprocess.run(f"python {os.path.join(os.path.dirname(os.path.realpath(__file__)), 'detectron2_lmi', 'converter/detectron2_onnx_trtonnx.py')} -b {kwargs.get('batch_size', 1)}", shell=True)

def convert_to_trt(**kwargs):
    subprocess.run(f'trtexec --onnx={os.path.join(f"/home/weights/onnx", "exported.onnx")} --saveEngine={os.path.join(f"/home/weights/", "model.trt")} --useCudaGraph --fp16', shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a model to ONNX and then to TensorRT.")
    parser.add_argument("-b","--batch-size", type=int, help="Batch size for the model", default=1)
    args = parser.parse_args()
    convert_to_onnx(batch_size=args.batch_size)
    convert_to_trt()