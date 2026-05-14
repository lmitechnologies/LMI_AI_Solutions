import os

from lmi_common.trt_convert import onnx_to_trt


def convert(args):
    from .cli import DET2_ONNX_EXPORT, DET2_PT_EXPORT, DET2_TRT_EXPORT

    output = args["output"]
    args["onnx_file_path"] = os.path.join(output, DET2_ONNX_EXPORT)
    args["trt_file_path"] = os.path.join(output, DET2_TRT_EXPORT)
    args["pt_file_path"] = os.path.join(output, DET2_PT_EXPORT)

    if args.get("pt", False):
        from .converter.detectron2_exporter import det2export

        args["format"] = "pt"
        det2export(args)

    if args.get("onnx", False):
        from .converter.detectron2_exporter import det2export
        from .converter.detectron2_onnx_trtonnx import onnx_gs

        args["format"] = "onnx"
        det2export(args)
        onnx_gs(args)

    if args.get("trt", False):
        onnx_to_trt(
            args["onnx_file_path"],
            args["trt_file_path"],
            fp16=args.get("fp16", False),
            workspace_gb=args.get("workspace_size", 4),
        )
