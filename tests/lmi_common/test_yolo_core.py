"""How YoloCore decides the input size a model runs at.

Ultralytics records that size differently per format. An exported model carries the real [h, w] in its
metadata. A .pt carries only its training args, and a run with rect=True stores just the long side there --
a scalar that is not a shape, because rect fits each batch to its own aspect ratio. Reading that scalar as
a square is wrong in both directions: it invents a shape the model never saw, and it hides the one check
such a model can support.
"""

import logging

import pytest

from lmi_common.yolo_core import YoloCore


def core(metadata=None, args=None):
    """A YoloCore with only the loaded-model state the size helpers read; __init__ wants a real file."""
    instance = YoloCore.__new__(YoloCore)
    backend = type("Backend", (), {})()
    backend.metadata = metadata
    backend.model = type("Weights", (), {"args": args})() if args is not None else None
    instance.model = backend
    return instance


# ---------------------------------------------------------------------------
# _infer_image_size
# ---------------------------------------------------------------------------


def test_exported_metadata_gives_the_real_pair():
    """The exporter writes the [h, w] it built the graph at; nothing else knows it as exactly."""
    assert core(metadata={"imgsz": [480, 640]})._infer_image_size() == [480, 640]


def test_metadata_wins_over_training_args():
    """An exported rectangle must not be overridden by the square long side left in the training args."""
    assert core(metadata={"imgsz": [384, 640]}, args={"imgsz": 640, "rect": True})._infer_image_size() == [384, 640]


def test_square_checkpoint_reports_its_square():
    """Without rect every image is letterboxed to imgsz, so the scalar really is the shape."""
    assert core(metadata={"imgsz": None}, args={"imgsz": 640, "rect": False})._infer_image_size() == [640, 640]


def test_rect_checkpoint_reports_nothing():
    """rect stores only the long side, so there is no shape to report and none may be invented."""
    assert core(metadata={"imgsz": None}, args={"imgsz": 640, "rect": True})._infer_image_size() is None


@pytest.mark.parametrize(
    "imgsz, expected",
    [([480, 640], [480, 640]), ((480, 640), [480, 640]), ([640], [640, 640])],
)
def test_a_pair_in_the_training_args_is_taken_as_written(imgsz, expected):
    assert core(args={"imgsz": imgsz, "rect": True})._infer_image_size() == expected


@pytest.mark.parametrize("metadata, args", [(None, None), ({}, None), ({"imgsz": None}, {})])
def test_a_model_that_records_no_size_reports_nothing(metadata, args):
    assert core(metadata=metadata, args=args)._infer_image_size() is None


def test_training_args_may_be_an_object_rather_than_a_dict():
    """AutoBackend hands back a namespace for some formats and a dict for others."""
    args = type("Args", (), {"imgsz": 640, "rect": False})()
    assert core(args=args)._infer_image_size() == [640, 640]


# ---------------------------------------------------------------------------
# _infer_long_side
# ---------------------------------------------------------------------------


def test_a_rect_checkpoint_still_knows_its_long_side():
    """The one fact rect does record, and the only check such a model can support."""
    assert core(args={"imgsz": 640, "rect": True})._infer_long_side() == 640


@pytest.mark.parametrize("args", [{"imgsz": 640, "rect": False}, {"imgsz": [480, 640], "rect": True}, {}, None])
def test_no_long_side_when_the_shape_is_already_known(args):
    """Anything that yields a full shape is checked against that shape instead."""
    assert core(args=args)._infer_long_side() is None


# ---------------------------------------------------------------------------
# _resolve_image_size
# ---------------------------------------------------------------------------


def test_the_caller_wins_over_the_model(caplog):
    """The caller is the only party that can know a rect model's shape, so it is never overridden."""
    instance = core(metadata={"imgsz": [480, 640]})
    with caplog.at_level(logging.WARNING):
        instance._resolve_image_size([640, 640])
    assert instance.image_size == [640, 640]
    assert "!= model's trained imgsz [480, 640]" in caplog.text


def test_a_size_matching_the_exported_graph_is_silent(caplog):
    instance = core(metadata={"imgsz": [480, 640]})
    with caplog.at_level(logging.INFO):
        instance._resolve_image_size([480, 640])
    assert instance.image_size == [480, 640]
    assert caplog.text == ""


def test_a_rect_model_at_the_trained_scale_is_silent(caplog):
    """[480, 640] and [640, 640] both letterbox a 640 long side at the training scale."""
    instance = core(args={"imgsz": 640, "rect": True})
    with caplog.at_level(logging.INFO):
        instance._resolve_image_size([480, 640])
    assert instance.image_size == [480, 640]
    assert caplog.text == ""


def test_a_rect_model_at_the_wrong_scale_is_reported(caplog):
    """A shorter long side shrinks every object relative to training, which no padding can undo."""
    instance = core(args={"imgsz": 640, "rect": True})
    with caplog.at_level(logging.WARNING):
        instance._resolve_image_size([320, 480])
    assert instance.image_size == [320, 480]
    assert "long side 640" in caplog.text


def test_an_unspecified_size_adopts_the_exported_shape(caplog):
    with caplog.at_level(logging.INFO):
        instance = core(metadata={"imgsz": [480, 640]})
        instance._resolve_image_size(None)
    assert instance.image_size == [480, 640]
    assert "using model's trained imgsz" in caplog.text


def test_an_unspecified_size_falls_back_when_the_model_cannot_say(caplog):
    """A rect .pt with no caller-supplied size leaves nothing to go on; say so rather than guess quietly."""
    instance = core(args={"imgsz": 640, "rect": True})
    with caplog.at_level(logging.WARNING):
        instance._resolve_image_size(None)
    assert instance.image_size == [640, 640]
    assert "trained imgsz unavailable" in caplog.text


def test_a_requested_size_is_coerced_to_ints():
    """Sizes arrive from CLI parsing and config files, not always as ints."""
    instance = core()
    instance._resolve_image_size(["480", "640"])
    assert instance.image_size == [480, 640]
