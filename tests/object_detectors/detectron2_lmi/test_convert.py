import argparse
import json
import sys
from types import SimpleNamespace

import onnx
import pytest
from onnx import numpy_helper

from lmi_common.model_metadata import metadata_from_props
from object_detectors.detectron2_lmi import cli
from object_detectors.detectron2_lmi.convert import convert, export_metadata
from object_detectors.detectron2_lmi.infer import add_args as add_infer_args
from object_detectors.detectron2_lmi.infer import inference_run

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


def _write_class_map(path, n):
    path.write_text(json.dumps({str(i): f"class{i}" for i in range(n)}))


def test_onnx_carries_the_config_mean_color_order_and_class_names(args, tmp_path):
    """A custom PIXEL_MEAN and FORMAT reach the graph and the metadata; class_map.json beside the config is embedded."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_path.read_text().replace("FORMAT: BGR", "FORMAT: RGB").replace("- 103.53", "- 99.0"))
    _write_class_map(tmp_path / "class_map.json", 80)
    convert({**args, "image_size": [640, 640]})

    model = onnx.load(str(tmp_path / "model.onnx"))
    (mean,) = [n for n in model.graph.node if n.name == "preprocessor/mean"]
    (const,) = [i for i in model.graph.initializer if i.name == mean.input[1]]
    assert numpy_helper.to_array(const).flatten().tolist() == pytest.approx([99.0, 116.28, 123.675])
    props = {p.key: p.value for p in model.metadata_props}
    metadata = metadata_from_props(props)
    assert metadata == {"input_format": "RGB", "class_names": [f"class{i}" for i in range(80)]}


def _cfg(num_classes=3, input_format="BGR"):
    return SimpleNamespace(
        INPUT=SimpleNamespace(FORMAT=input_format), MODEL=SimpleNamespace(ROI_HEADS=SimpleNamespace(NUM_CLASSES=num_classes))
    )


def test_export_metadata_reads_the_class_map_beside_the_config(tmp_path):
    _write_class_map(tmp_path / "class_map.json", 3)
    assert export_metadata(_cfg(), None, str(tmp_path / "config.yaml"))["class_names"] == ["class0", "class1", "class2"]


def test_export_metadata_prefers_the_given_class_map(tmp_path):
    _write_class_map(tmp_path / "class_map.json", 3)
    given = tmp_path / "given.json"
    given.write_text(json.dumps({"2": "c", "0": "a", "1": "b"}))
    assert export_metadata(_cfg(), str(given), str(tmp_path / "config.yaml"))["class_names"] == ["a", "b", "c"]


def test_export_metadata_without_a_class_map_has_no_class_names(tmp_path, caplog):
    assert export_metadata(_cfg(), None, str(tmp_path / "config.yaml")) == {"input_format": "BGR"}
    assert "without class names" in caplog.text


@pytest.mark.parametrize("ids", [[0, 1], [1, 2, 3], [0, 1, 2, 3]])
def test_export_metadata_rejects_a_class_map_that_does_not_match_num_classes(tmp_path, ids):
    path = tmp_path / "class_map.json"
    path.write_text(json.dumps({str(i): str(i) for i in ids}))
    with pytest.raises(ValueError, match="NUM_CLASSES"):
        export_metadata(_cfg(), str(path), str(tmp_path / "config.yaml"))


@pytest.mark.parametrize("class_map", [{"dent": 0, "scratch": 1, "chip": 2}, {"0": "dent", "1": None, "2": {"name": "chip"}}])
def test_export_metadata_explains_a_class_map_that_is_not_ids_to_names(tmp_path, class_map):
    path = tmp_path / "class_map.json"
    path.write_text(json.dumps(class_map))
    with pytest.raises(ValueError, match="0-based class ids to names"):
        export_metadata(_cfg(), str(path), str(tmp_path / "config.yaml"))


def test_export_metadata_rejects_an_unsupported_color_order(tmp_path):
    with pytest.raises(ValueError, match="INPUT.FORMAT"):
        export_metadata(_cfg(input_format="YUV-BT.601"), None, str(tmp_path / "config.yaml"))


@pytest.mark.parametrize(
    "flags, error",
    [(["--pt", "-m", "x.json"], "only by --onnx and --trt"), (["--onnx", "-is", "640", "640", "-m", "missing.json"], "not found")],
)
def test_cli_checks_the_class_map_flag(tmp_path, monkeypatch, capsys, flags, error):
    monkeypatch.setattr(sys, "argv", ["cli", "convert", "-w", "w.pth", "-o", str(tmp_path), *flags])
    with pytest.raises(SystemExit):
        cli.main()
    assert error in capsys.readouterr().err


def test_standalone_infer_cli_does_not_need_a_class_map():
    parser = argparse.ArgumentParser()
    add_infer_args(parser)
    assert parser.parse_args(["-w", "model.engine", "-i", "in", "-o", "out"]).class_map is None


def test_infer_needs_a_class_map_file_for_a_pt(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.json"):
        inference_run({"weights": "model.pt", "input": str(tmp_path), "output": str(tmp_path), "class_map": str(tmp_path / "missing.json")})
