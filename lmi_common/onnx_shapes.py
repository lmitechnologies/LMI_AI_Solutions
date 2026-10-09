import logging

logger = logging.getLogger(__name__)


def pin_onnx_output_dims(onnx_path: str) -> None:
    """Record static non-batch output dims that torch's legacy exporter leaves symbolic (e.g. ``Reshapeoutput_dim_1``), in place.

    ``ONNXEngine`` rejects symbolic non-batch dims.
    """
    try:
        import onnx
    except ImportError:
        logger.warning(f"onnx is not installed; output shapes in {onnx_path} stay symbolic and ONNXEngine will reject them.")
        return

    model = onnx.load(str(onnx_path), load_external_data=False)
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    # inference keeps existing output dims, so clear them to get the inferred ones
    for out in probe.graph.output:
        out.type.tensor_type.ClearField("shape")
    inferred = onnx.shape_inference.infer_shapes(probe, data_prop=True)

    for out, inf in zip(model.graph.output, inferred.graph.output):
        dims, inf_dims = out.type.tensor_type.shape.dim, inf.type.tensor_type.shape.dim
        if len(dims) != len(inf_dims):
            continue
        # dim 0 keeps its exported name, e.g. batch_size
        for dim, inf_dim in zip(dims[1:], inf_dims[1:]):
            if inf_dim.HasField("dim_value"):
                dim.dim_value = inf_dim.dim_value
    onnx.save(model, str(onnx_path))
