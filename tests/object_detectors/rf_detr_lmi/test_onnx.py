import glob
import logging
import os

import cv2
import numpy as np
import pytest
import torch
from rfdetr.assets.coco_classes import COCO_CLASSES

from object_detectors.od_core.object_detector import ObjectDetector
from object_detectors.rf_detr_lmi.model import RfdetrONNX

from .test_model import _assert_empty_out, _assert_nonempty_out, _assert_scores_geq

logger = logging.getLogger(__name__)

COCO_DIR = "tests/assets/images/coco"
PTH_FILE = "tests/assets/models/od/rf_detr/rf-detr-seg-small.pth"
KEYS = ["boxes", "scores", "masks", "segments", "classes"]
IMAGE_SIZE = 384
MODEL_TYPE = "seg-small"
METADATA = dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr")


@pytest.fixture(scope="module")
def imgs_coco():
    paths = sorted(p for p in glob.glob(os.path.join(COCO_DIR, "*")) if p.endswith((".png", ".jpg")))
    return [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE)) for p in paths[:2]]


@pytest.fixture(scope="module")
def onnx_file(tmp_path_factory):
    """Export the checkpoint with the repo's own converter — cheap enough (~2s) to avoid a committed asset."""
    from rfdetr import RFDETRSegSmall

    from object_detectors.rf_detr_lmi.convert import convert_to_onnx

    out_dir = str(tmp_path_factory.mktemp("rfdetr_onnx"))
    convert_to_onnx(RFDETRSegSmall(pretrain_weights=PTH_FILE, resolution=IMAGE_SIZE, device="cpu"), out_dir)
    produced = glob.glob(os.path.join(out_dir, "*.onnx"))
    assert produced, f"rfdetr export produced no .onnx in {out_dir}"
    return produced[0]


@pytest.fixture(scope="module")
def onnx_model(onnx_file):
    return ObjectDetector(
        metadata=METADATA, model_path=onnx_file, class_map=COCO_CLASSES, image_size=[IMAGE_SIZE, IMAGE_SIZE], device="cpu"
    )


@pytest.fixture(scope="module")
def pth_model():
    return ObjectDetector(
        metadata=METADATA,
        model_path=PTH_FILE,
        model_type=MODEL_TYPE,
        class_map=COCO_CLASSES,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
        device="cpu",
    )


def test_dispatch_and_shapes(onnx_model):
    assert isinstance(onnx_model, RfdetrONNX)
    assert onnx_model.image_size == [IMAGE_SIZE, IMAGE_SIZE]
    assert onnx_model.fixed_batch_size == 1  # rfdetr exports a static batch


def test_warmup(onnx_model):
    onnx_model.warmup()


def test_empty(onnx_model):
    out, _ = onnx_model.predict(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8), configs=0.5)
    _assert_empty_out({k: out[k][0] for k in KEYS})


def test_matches_pth(imgs_coco, onnx_model, pth_model):
    """Same weights through a different runtime: detections must agree, which also pins the dets/labels/masks output order."""
    out, _ = onnx_model.predict(imgs_coco, configs=0.5)
    ref, _ = pth_model.predict(imgs_coco, configs=0.5)
    for i in range(len(imgs_coco)):
        _assert_nonempty_out({k: out[k][i] for k in KEYS})
        _assert_scores_geq({"scores": out["scores"][i]}, 0.5)
        assert np.array_equal(ref["classes"][i], out["classes"][i]), f"image {i}: class mismatch"
        assert np.asarray(out["masks"][i]).shape == np.asarray(ref["masks"][i]).shape, f"image {i}: mask shape mismatch"
        # ONNX Runtime and torch order float ops differently, so boxes agree only to sub-pixel noise.
        np.testing.assert_allclose(ref["boxes"][i], out["boxes"][i], atol=0.5)
        np.testing.assert_allclose(ref["scores"][i], out["scores"][i], atol=5e-3)


def test_class_map_from_embedded_metadata(imgs_coco, onnx_file, pth_model):
    """convert_to_onnx embeds the ordered class names in the file; names read back from it must match an explicit class_map."""
    model = ObjectDetector(metadata=METADATA, model_path=onnx_file, image_size=[IMAGE_SIZE, IMAGE_SIZE], device="cpu")
    assert model.class_map == COCO_CLASSES

    out, _ = model.predict(imgs_coco, configs=0.5)
    ref, _ = pth_model.predict(imgs_coco, configs=0.5)
    for i in range(len(imgs_coco)):
        assert np.array_equal(ref["classes"][i], out["classes"][i]), f"image {i}: embedded names disagree with explicit class_map"


def test_num_select_from_embedded_metadata(onnx_model, pth_model):
    """The engine backend must reuse the checkpoint's num_select (100 for seg-small), not a hardcoded 300."""
    assert pth_model.postprocessor.num_select == 100, "rfdetr changed seg-small's num_select; update this test"
    assert onnx_model.postprocessor.num_select == pth_model.postprocessor.num_select


def test_num_select_falls_back_when_absent(tmp_path, onnx_file):
    """An ONNX exported before num_select was embedded must still load, at rfdetr's 300 default."""
    import onnx

    from lmi_common.model_metadata import embed_onnx_metadata

    legacy = str(tmp_path / "no_num_select.onnx")
    onnx.save(onnx.load(onnx_file), legacy)
    embed_onnx_metadata(legacy, {"class_names": list(COCO_CLASSES.values())})

    assert RfdetrONNX(legacy, device="cpu").postprocessor.num_select == 300


def test_missing_class_names_raises(tmp_path, onnx_file):
    """A model carrying no embedded names and no class_map must fail with a clear error, not a wrong class map."""
    import onnx

    from lmi_common.model_metadata import ONNX_METADATA_KEY

    stripped = str(tmp_path / "no_metadata.onnx")
    model = onnx.load(onnx_file)
    del model.metadata_props[[p.key for p in model.metadata_props].index(ONNX_METADATA_KEY)]
    onnx.save(model, stripped)

    with pytest.raises(ValueError, match="no class names embedded"):
        ObjectDetector(metadata=METADATA, model_path=stripped, image_size=[IMAGE_SIZE, IMAGE_SIZE], device="cpu")


def test_cuda(imgs_coco, onnx_file):
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device.")
    try:
        model = ObjectDetector(
            metadata=METADATA, model_path=onnx_file, class_map=COCO_CLASSES, image_size=[IMAGE_SIZE, IMAGE_SIZE], device="cuda"
        )
    except RuntimeError as e:
        pytest.skip(f"onnxruntime CUDAExecutionProvider unavailable: {e}")
    out, _ = model.predict(imgs_coco, configs=0.5)
    for i in range(len(imgs_coco)):
        _assert_nonempty_out({k: out[k][i] for k in KEYS})
