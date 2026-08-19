import os

from lmi_common.model_metadata import embed_onnx_metadata
from lmi_common.trt_convert import onnx_to_trt


def convert_to_onnx(model, output_dir: str, **kwargs) -> str:
    """Export an rfdetr model to ONNX with its class names and num_select embedded, and return the .onnx path.

    Args:
        model: An rfdetr model instance.
        output_dir (str): Directory to write the export into; rfdetr names the file itself.
        **kwargs: opset_version (int, default 17), plus any rfdetr export kwargs.

    Returns:
        Path to the exported ONNX file.
    """
    # num_select varies by variant and is not recoverable from the exported graph.
    metadata = {"class_names": list(model.class_names), "num_select": int(model.model.postprocess.num_select)}
    onnx_path = model.export(output_dir=output_dir, opset_version=kwargs.pop("opset_version", 17), **kwargs)
    embed_onnx_metadata(onnx_path, metadata)
    return onnx_path


def convert_to_tensorrt(onnx_path: str, **kwargs) -> str:
    """Build a TensorRT engine next to the ONNX, carrying over the ONNX's embedded metadata.

    Args:
        onnx_path (str): Path to the ONNX model file.
        **kwargs: Forwarded to ``lmi_common.trt_convert.onnx_to_trt`` (fp16, workspace_gb, batch profile).

    Returns:
        Path to the built engine.
    """
    engine_path = os.path.splitext(onnx_path)[0] + ".engine"
    onnx_to_trt(onnx_path, engine_path, **kwargs)
    return engine_path
