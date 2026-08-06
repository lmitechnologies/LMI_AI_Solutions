"""What ultralytics itself records about a model's input size.

The unit tests in test_yolo_core.py hand YoloCore a stub, so they pin our reading of a given input and stay
green no matter what ultralytics emits. These run the real thing instead: they fail if a future ultralytics
drops the export metadata, stops collapsing a configured pair to the long side, or starts recording a rect
run's size as a pair. Each of those silently changes which branch of _infer_image_size a real model takes.
"""

import numpy as np
import pytest
import torch
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_imgsz

from lmi_common.yolo_core import YoloCore

ASSET_MODEL = "tests/assets/models/od/ultralytics/yolo11n.pt"
LONG_SIDE = 640
DEPLOY_SHAPE = [480, 640]  # deliberately non-square: a square hides every ordering mistake


@pytest.fixture(scope="module")
def rect_dataset(tmp_path_factory):
    """A two-image dataset written to disk, the minimum a rect training run needs."""
    cv2 = pytest.importorskip("cv2")
    yaml = pytest.importorskip("yaml")
    root = tmp_path_factory.mktemp("rect_data")
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), np.zeros((1080, 1920, 3), np.uint8))
            (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "a"}}))
    return str(data)


@pytest.fixture(scope="module")
def rect_checkpoint(rect_dataset, tmp_path_factory):
    """A checkpoint from a real rect run, configured with a rectangle to prove the rectangle is dropped."""
    out = tmp_path_factory.mktemp("rect_run")
    model = YOLO("yolo11n.yaml", task="detect")
    model.train(
        data=rect_dataset,
        imgsz=[LONG_SIDE, 480],
        rect=True,
        epochs=1,
        batch=2,
        device="cpu",
        workers=0,
        project=str(out),
        name="run",
        exist_ok=True,
        plots=False,
        val=False,
        verbose=False,
    )
    return out / "run" / "weights" / "last.pt"


@pytest.fixture(scope="module")
def exported_onnx(tmp_path_factory):
    """A stock checkpoint exported at a non-square size, as the trainer exports a deployable model."""
    pytest.importorskip("onnx")
    source = tmp_path_factory.mktemp("export") / "model.pt"
    source.write_bytes(open(ASSET_MODEL, "rb").read())
    return YOLO(str(source), task="detect").export(format="onnx", imgsz=DEPLOY_SHAPE, simplify=False, verbose=False)


def test_training_still_collapses_a_configured_pair_to_the_long_side():
    """If this stops holding, a rect checkpoint's imgsz becomes a shape and the scalar branch is wrong."""
    assert check_imgsz([LONG_SIDE, 480], max_dim=1) == LONG_SIDE


def test_a_rect_run_records_a_scalar_and_its_rect_flag(rect_checkpoint):
    """The scalar plus the flag are exactly what _infer_image_size needs to refuse to invent a shape."""
    train_args = torch.load(rect_checkpoint, map_location="cpu", weights_only=False)["train_args"]
    assert train_args["rect"] is True
    assert isinstance(train_args["imgsz"], int), "a pair here would mean the shape is knowable after all"
    assert train_args["imgsz"] == LONG_SIDE


def test_a_rect_checkpoint_yields_no_shape_but_keeps_its_long_side(rect_checkpoint):
    """The end the unit tests assert with a stub, reached from a real file."""
    core = YoloCore(str(rect_checkpoint), device="cpu", image_size=DEPLOY_SHAPE)
    assert core._infer_image_size() is None
    assert core._infer_long_side() == LONG_SIDE
    assert core.image_size == DEPLOY_SHAPE


def test_export_still_writes_the_pair_into_the_model_metadata(exported_onnx):
    """Losing this silently removes the only size check an onnx or engine model has."""
    backend = AutoBackend(str(exported_onnx), torch.device("cpu"), fuse=False)
    assert backend.metadata.get("imgsz") == DEPLOY_SHAPE


def test_the_metadata_pair_is_height_first(exported_onnx):
    """Proven against the graph rather than assumed, since a transposed pair still looks well-formed."""
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(exported_onnx), providers=["CPUExecutionProvider"])
    assert list(session.get_inputs()[0].shape[2:]) == DEPLOY_SHAPE  # NCHW


def test_an_exported_model_reports_the_shape_it_was_built_at(exported_onnx):
    core = YoloCore(str(exported_onnx), device="cpu", image_size=None)
    assert core._infer_image_size() == DEPLOY_SHAPE
    assert core.image_size == DEPLOY_SHAPE
