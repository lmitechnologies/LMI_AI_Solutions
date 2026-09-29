import cv2
import numpy as np
import onnx
import pytest
from detectron2.config import get_cfg

from object_detectors.detectron2_lmi.convert import convert, engine_input_size

CONFIG_PATH = "tests/assets/models/od/detectron2/config.yaml"
WEIGHTS = "tests/assets/models/od/detectron2/model_final_f10217.pkl"
SAMPLE = "tests/assets/models/od/detectron2/sample_image.png"


def _cfg(min_size, max_size):
    cfg = get_cfg()
    cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST = min_size, max_size
    return cfg


@pytest.mark.parametrize(
    "min_size,max_size,hw,expected",
    [
        (800, 1333, (640, 640), (800, 800)),
        (800, 1333, (480, 640), (800, 1056)),  # 1066.7 rounds to 33 * 32
        (800, 1000, (400, 800), (512, 992)),  # capped at 500 x 1000 by MAX_SIZE_TEST, then rounded
        (0, 1333, (500, 700), (512, 704)),  # MIN_SIZE_TEST 0 turns the resize off
    ],
)
def test_engine_input_size(min_size, max_size, hw, expected):
    assert engine_input_size(_cfg(min_size, max_size), *hw) == expected


def test_onnx_input_is_the_engine_size_for_a_non_square_sample(tmp_path):
    """The ONNX is traced, and its anchors built, at one size: the sample's test-time resize."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(open(CONFIG_PATH).read().replace("DEVICE: cuda", "DEVICE: cpu"))
    sample = tmp_path / "sample.png"
    cv2.imwrite(str(sample), cv2.resize(cv2.imread(SAMPLE), (640, 480)))

    convert(
        {
            "config_file": str(cfg_path),
            "weights": WEIGHTS,
            "sample_image": str(sample),
            "output": str(tmp_path),
            "batch_size": 1,
            "onnx": True,
        }
    )

    dims = onnx.load(str(tmp_path / "model.onnx")).graph.input[0].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == [1, 3, 800, 1056]
    assert np.array_equal(cv2.imread(str(sample)).shape, (480, 640, 3)), "the user's sample is left as is"
