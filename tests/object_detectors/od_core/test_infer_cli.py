import argparse

import numpy as np
import pytest

from lmi_utils.dataset_utils.representations import AnnotationType, Dataset
from object_detectors.od_core.infer_cli import (
    MERGE_ORIGIN_LEGEND,
    PredictionsJson,
    add_infer_args,
    check_infer_args,
    plot_tile_merge,
    predict_tiled,
)


def _parse(*argv, **defaults):
    parser = argparse.ArgumentParser()
    add_infer_args(parser, confidence=0.3, **defaults)
    args = parser.parse_args(["-w", "m.pt", "-i", "in", "-o", "out", *argv])
    check_infer_args(parser, args)
    return args


def test_defaults():
    args = _parse()
    assert (args.weights, args.input, args.output, args.confidence) == ("m.pt", "in", "out", 0.3)
    assert args.image_size is None and args.tile_step is None and not args.json


def test_paths_with_a_default_are_optional():
    parser = argparse.ArgumentParser()
    add_infer_args(parser, confidence=0.5, weights="/w/model.pt", input="/in", output="/out")
    args = parser.parse_args([])
    assert (args.weights, args.input, args.output) == ("/w/model.pt", "/in", "/out")


@pytest.mark.parametrize(("argv", "size"), [(["-s", "640"], [640, 640]), (["--image_size", "480", "640"], [480, 640])])
def test_image_size_is_h_w(argv, size):
    assert _parse(*argv).image_size == size


def test_a_single_int_is_a_square_tile_and_stride():
    step = _parse("--tile", "512", "--stride", "256").tile_step
    assert step.tile_size == 512 and step.stride == 256


def test_two_ints_are_height_then_width():
    step = _parse("--tile", "512", "640", "--stride", "256", "320").tile_step
    assert step.tile_size == [512, 640] and step.stride == [256, 320]


@pytest.mark.parametrize(
    "argv",
    [
        ["--tile", "512"],
        ["--stride", "256"],
        ["--tile", "1", "2", "3", "--stride", "1"],
        ["--tile", "0", "--stride", "1"],
        ["-s", "640", "480", "3"],
    ],
)
def test_bad_sizes_exit_with_a_usage_error(argv):
    with pytest.raises(SystemExit):
        _parse(*argv)


def test_tiling_asks_the_merge_for_its_origin_codes():
    assert _parse("--tile", "512", "--stride", "256").tile_step.report_merge_origin


def test_predict_tiled_returns_the_tile_grid():
    class EchoModel:
        def predict(self, tiles, operators=None, **kwargs):
            return {"n_tiles": len(tiles)}, {}

    step = _parse("--tile", "512", "--stride", "256").tile_step
    results, _, tile_boxes = predict_tiled(EchoModel(), np.zeros((1024, 1024, 3), np.uint8), step)
    assert results["n_tiles"] == 9
    assert tile_boxes.tolist()[:2] == [[0, 0, 512, 512], [256, 0, 768, 512]]


def test_plot_tile_merge_colors_each_detection_by_origin():
    colors = {code: color for code, _, color in MERGE_ORIGIN_LEGEND}
    image = np.zeros((400, 400, 3), np.uint8)
    boxes = np.array([[250.0, 100, 350, 200], [250, 250, 350, 350]])
    out = plot_tile_merge(image, {"boxes": boxes, "classes": ["a", "b"], "merge_origin": np.array([2, 3])}, np.zeros((0, 4)))
    assert tuple(out[300, 350]) == colors[3]  # right edge of the second box, clear of its label
    assert tuple(out[150, 350]) == colors[2]
    assert not image.any()  # the input is not drawn on


def test_plot_tile_merge_puts_the_legend_beside_the_image():
    image = np.full((400, 300, 3), 7, np.uint8)
    out = plot_tile_merge(image, {"boxes": np.zeros((0, 4)), "classes": []}, np.zeros((0, 4)))
    assert out.shape[0] == 400 and out.shape[1] > 300
    assert (out[:, :300] == image).all()  # the image is left uncovered
    assert out[:, 300:].any()


@pytest.mark.parametrize("flag", ["--no_label", "--no-label"])
def test_no_label_takes_both_spellings(flag):
    assert not _parse().no_label
    assert _parse(flag).no_label


def test_plot_tile_merge_can_hide_labels():
    outputs = {"boxes": np.array([[250.0, 250, 350, 350]]), "classes": ["a"], "merge_origin": np.array([0])}
    labeled = plot_tile_merge(np.zeros((400, 400, 3), np.uint8), outputs, np.zeros((0, 4)))
    hidden = plot_tile_merge(np.zeros((400, 400, 3), np.uint8), outputs, np.zeros((0, 4)), hide_label=True)
    above_box = (slice(200, 248), slice(250, 350))  # where the label is drawn
    assert labeled[above_box].any() and not hidden[above_box].any()


def test_line_thickness_must_be_positive():
    assert _parse().line_thickness is None
    assert _parse("--line_thickness", "1").line_thickness == 1
    assert _parse("--line-thickness", "2").line_thickness == 2
    with pytest.raises(SystemExit):
        _parse("--line_thickness", "0")


def test_plot_tile_merge_draws_thinner_lines_when_asked():
    outputs = {"boxes": np.array([[200.0, 250, 300, 350]]), "classes": ["a"], "merge_origin": np.array([0])}
    grid = np.array([[0.0, 0, 400, 400]])
    drawn = [
        plot_tile_merge(np.zeros((400, 400, 3), np.uint8), outputs, grid, hide_label=True, line_thickness=t)[:, :400].any(axis=2).sum()
        for t in (1, 5)
    ]  # the image area, without the legend panel
    assert 0 < drawn[0] < drawn[1]


def test_plot_tile_merge_treats_missing_origin_codes_as_whole():
    boxes = np.array([[250.0, 250, 350, 350]])
    out = plot_tile_merge(np.zeros((400, 400, 3), np.uint8), {"boxes": boxes, "classes": ["a"]}, np.zeros((0, 4)))
    assert tuple(out[300, 350]) == MERGE_ORIGIN_LEGEND[0][2]


def test_predictions_json_round_trips_boxes_masks_rotated_boxes_and_keypoints(tmp_path):
    writer = PredictionsJson(["a", "b"])
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:5, 3:9] = 1
    writer.add("seg.png", 20, 30, {"boxes": np.array([[3.0, 2.0, 9.0, 5.0]]), "scores": np.array([0.9]), "classes": ["a"], "masks": [mask]})
    corners = np.array([[[2.0, 2.0], [12.0, 2.0], [12.0, 6.0], [2.0, 6.0]]])
    writer.add("obb.png", 20, 30, {"boxes": corners, "scores": np.array([0.8]), "classes": ["b"]})
    pose = {
        "boxes": np.array([[1.0, 1.0, 10.0, 10.0]]),
        "scores": np.array([0.7]),
        "classes": ["a"],
        "points": np.array([[[2.0, 3.0, 0.9]]]),
    }
    writer.add("sub/pose.png", 20, 30, pose)
    writer.add("empty.png", 20, 30, {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "classes": []})
    writer.save(str(tmp_path / "predictions.json"))

    dataset = Dataset.load(str(tmp_path / "predictions.json"))
    assert [label.id for label in dataset.labels] == ["a", "b"]
    seg, obb, pose, empty = dataset.files
    assert (seg.path, seg.height, seg.width) == ("seg.png", 20, 30)
    assert seg.predictions[0].type == AnnotationType.MASK and seg.predictions[0].confidence == pytest.approx(0.9)
    assert seg.predictions[0].value.to_numpy(h=20, w=30).sum() == mask.sum()
    assert obb.predictions[0].type == AnnotationType.BOX and obb.predictions[0].value.to_polygon().to_numpy().shape == (4, 2)
    box, keypoint = pose.predictions
    assert keypoint.type == AnnotationType.KEYPOINT and keypoint.bounding_box_id == box.id
    assert (keypoint.value.x, keypoint.value.y) == (2.0, 3.0)
    assert empty.predictions == []
    assert len({a.id for f in dataset.files for a in f.predictions}) == 4
