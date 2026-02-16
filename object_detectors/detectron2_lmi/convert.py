import subprocess

<<<<<<< HEAD
from object_detectors.detectron2_lmi.converter.detectron2_exporter import det2export
from object_detectors.detectron2_lmi.converter.detectron2_onnx_trtonnx import onnx_gs

=======
>>>>>>> 49e84705 (updated)

def convert_to_trt(args):
    command = f"trtexec --onnx={args.get('onnx_file_path')} --saveEngine={args.get('trt_file_path')} --useCudaGraph"
    if args.get("fp16", False):
        command += " --fp16"

    subprocess.run(command, shell=True)


def convert(args):
    if args.get("pt", True):
        from detectron2_lmi.converter.detectron2_exporter import det2export

        args["format"] = "pt"
        det2export(args)

    if args.get("onnx", True):
        from detectron2_lmi.converter.detectron2_onnx_trtonnx import onnx_gs

        args["format"] = "onnx"
        det2export(args)
        onnx_gs(args)

    if args.get("trt", True):
        convert_to_trt(args)
