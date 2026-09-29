import onnx
import pytest

from object_detectors.detectron2_lmi.convert import convert

CONFIG_PATH = "tests/assets/models/od/detectron2/config.yaml"
WEIGHTS = "tests/assets/models/od/detectron2/model_final_f10217.pkl"


@pytest.fixture
def args(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(open(CONFIG_PATH).read().replace("DEVICE: cuda", "DEVICE: cpu"))
    return {"config_file": str(cfg_path), "weights": WEIGHTS, "output": str(tmp_path), "batch_size": 1, "onnx": True}


def test_onnx_input_is_the_requested_size(args, tmp_path):
    """The ONNX is traced, and its anchors built, at image_size; a non-square size works."""
    convert({**args, "image_size": [800, 1056]})
    dims = onnx.load(str(tmp_path / "model.onnx")).graph.input[0].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == [1, 3, 800, 1056]


@pytest.mark.parametrize("image_size", [None, [800], [800, 1000], [0, 800]])
def test_onnx_needs_an_image_size_in_multiples_of_32(args, image_size):
    with pytest.raises(ValueError, match="image_size"):
        convert({**args, "image_size": image_size})
