import subprocess
from detectron2_lmi.converter.detectron2_exporter import det2export
from detectron2_lmi.converter.detectron2_onnx_trtonnx import onnx_gs


def convert_to_trt(args):
    command = f"trtexec --onnx={args.get('onnx_file_path')} --saveEngine={args.get('trt_file_path')} --useCudaGraph"
    if args.get("fp16", False):
        command += " --fp16"

    subprocess.run(command, shell=True)


def convert(args):
    if args.get("pt", True):
        args["format"] = "pt"
        det2export(args)

    if args.get("onnx", True):
        args["format"] = "onnx"
        det2export(args)
        onnx_gs(args)

    if args.get("trt", True):
        convert_to_trt(args)
