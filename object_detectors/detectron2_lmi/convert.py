import os

from lmi_common.trt_convert import onnx_to_trt


def convert(args):
    from .cli import DET2_ONNX_EXPORT, DET2_PT_EXPORT, DET2_TRT_EXPORT

    # the engine is built from the ONNX
    build_onnx = args.get("onnx", False) or args.get("trt", False)
    image_size = args.get("image_size")
    if build_onnx and (not image_size or len(image_size) != 2 or any(x <= 0 or x % 32 for x in image_size)):
        raise ValueError(f"ONNX/TensorRT conversion needs image_size (h, w) in positive multiples of 32, got {image_size}")

    output = args["output"]
    os.makedirs(output, exist_ok=True)
    args["onnx_file_path"] = os.path.join(output, DET2_ONNX_EXPORT)
    args["trt_file_path"] = os.path.join(output, DET2_TRT_EXPORT)
    args["pt_file_path"] = os.path.join(output, DET2_PT_EXPORT)

    if args.get("pt", False):
        from .converter.detectron2_exporter import det2export

        args["format"] = "pt"
        det2export(args)

    if build_onnx:
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
