import argparse

import cv2
import numpy as np
import pytest

from classifiers.cls_core.infer_cli import TILE_PLOT_DIR, add_infer_args, check_infer_args, predict_tiled, save_tile_plot
from lmi_utils.preprocess_utils.ops import TileMeta
from lmi_utils.preprocess_utils.preprocessor import Preprocessor


def _parse(*argv, **defaults):
    parser = argparse.ArgumentParser()
    add_infer_args(parser, **defaults)
    args = parser.parse_args(["-w", "m.pt", "-i", "in", "-o", "out", *argv])
    check_infer_args(parser, args)
    return args


def _tile_boxes(image, step):
    _, history = Preprocessor().preprocess([image], [step])
    return history[0].tiler(0).tile_boxes()


def test_defaults():
    args = _parse()
    assert args.image_size is None and args.tile_step is None


def test_paths_with_defaults_are_optional():
    parser = argparse.ArgumentParser()
    add_infer_args(parser, weights="/w/m.pt", input="/in", output="/out")
    args = parser.parse_args([])
    assert (args.weights, args.input, args.output) == ("/w/m.pt", "/in", "/out")


@pytest.mark.parametrize(("argv", "size"), [(["-s", "224"], [224, 224]), (["--image_size", "128", "256"], [128, 256])])
def test_image_size_is_h_w(argv, size):
    assert _parse(*argv).image_size == size


def test_single_int_tile_and_stride():
    step = _parse("--tile", "512", "--stride", "256").tile_step
    assert step.tile_size == 512 and step.stride == 256


def test_two_int_tile_and_stride_are_h_w():
    step = _parse("--tile", "512", "640", "--stride", "256", "320").tile_step
    assert step.tile_size == [512, 640] and step.stride == [256, 320]


@pytest.mark.parametrize(
    "argv",
    [
        ["--tile", "512"],
        ["--stride", "256"],
        ["--tile", "1", "2", "3", "--stride", "1"],
        ["--tile", "0", "--stride", "1"],
        ["-s", "224", "224", "3"],
    ],
)
def test_bad_sizes_exit_with_usage_error(argv):
    with pytest.raises(SystemExit):
        _parse(*argv)


def test_classifier_tile_step_has_no_merge_origin():
    # classifiers have no spatial outputs; merge_origin is an OD-only concept
    step = _parse("--tile", "512", "--stride", "256").tile_step
    assert not step.report_merge_origin


def test_predict_tiled_passes_the_tiles_and_their_history_and_returns_the_grid():
    class EchoModel:
        def predict(self, tiles, operators=None, **kwargs):
            return {"n_tiles": len(tiles), "operators": operators, **kwargs}, {}

    step = _parse("--tile", "512", "--stride", "256").tile_step
    results, _, tile_boxes = predict_tiled(EchoModel(), np.zeros((1024, 1024, 3), np.uint8), step, batch_size=4)
    assert results["n_tiles"] == 9
    assert isinstance(results["operators"][0], TileMeta)
    assert results["batch_size"] == 4
    assert tile_boxes.tolist()[:2] == [[0, 0, 512, 512], [256, 0, 768, 512]]


def test_save_tile_plot_writes_the_grid_without_touching_the_input(tmp_path):
    image = np.zeros((256, 256, 3), np.uint8)
    step = _parse("--tile", "128", "--stride", "128").tile_step
    save_tile_plot(str(tmp_path), "out.png", image, _tile_boxes(image, step))
    assert not image.any()
    saved = cv2.imread(str(tmp_path / TILE_PLOT_DIR / "out.png"))
    assert saved.shape == image.shape and saved.any()


# ── confidence flag ──────────────────────────────────────────────────────────


def test_confidence_default_is_zero():
    assert _parse().confidence == 0


def test_confidence_parses_float():
    assert _parse("-c", "0.75").confidence == 0.75
    assert _parse("--confidence", "0.5").confidence == 0.5


def test_negative_confidence_exits_with_usage_error():
    with pytest.raises(SystemExit):
        _parse("-c", "-0.1")


def test_predict_tiled_passes_confidence_through():
    class EchoModel:
        def predict(self, tiles, operators=None, **kwargs):
            return {"n_tiles": len(tiles), "operators": operators, **kwargs}, {}

    step = _parse("--tile", "512", "--stride", "256").tile_step
    results, _, _ = predict_tiled(EchoModel(), np.zeros((1024, 1024, 3), np.uint8), step, confidence=0.5)
    assert results["confidence"] == 0.5
