import logging
import os
from importlib.metadata import version

import cv2
import numpy as np
import pytest
import torch
from packaging.version import Version
from rfdetr import RFDETRSegSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

from object_detectors.od_core.object_detector import ObjectDetector
from object_detectors.rf_detr_lmi.model import RfdetrBase, RfdetrModel

logger = logging.getLogger(__name__)

COCO_DIR = "tests/assets/images/coco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PTH_FILE = "tests/assets/models/od/rf_detr/rf-detr-seg-small.pth"
TRT_MODEL = "tests/assets/models/od/rf_detr/inference_model.engine"
OUT_DIR = "tests/outputs/od/rf_detr"
IMAGE_SIZE = 384
MODEL_TYPE = "seg-small"
OFF_SIZES = [(512, 640), (576, 704), (704, 512)]  # (h, w), non-square
# rfdetr 1.9.0 dropped antialiasing from predict(); older versions can't match preprocess() pixel for pixel.
RFDETR_ANTIALIASES_PREDICT = Version(version("rfdetr")) < Version("1.9.0")


def load_image(path):
    im = cv2.imread(path)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb


@pytest.fixture(scope="module")
def imgs_coco():
    paths = [os.path.join(COCO_DIR, img) for img in os.listdir(COCO_DIR)]
    images = []
    for p in paths:
        if "png" not in p and "jpg" not in p:
            continue
        rgb = load_image(p)
        h, w = rgb.shape[:2]
        images.append(rgb)
    return images


@pytest.fixture(scope="module")
def rf_model():
    model = RFDETRSegSmall(pretrain_weights=PTH_FILE, device=DEVICE)
    return model


@pytest.fixture(scope="module")
def obj_detector():
    obj_detector = ObjectDetector(
        metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
        model_path=PTH_FILE,
        device=DEVICE,
        class_map=COCO_CLASSES,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
    )
    return obj_detector


@pytest.fixture(scope="module")
def trt_model():
    if DEVICE != "cuda":
        pytest.skip("TensorRT model can only be tested on CUDA device.")
    try:
        return ObjectDetector(
            metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
            model_path=TRT_MODEL,
            class_map=COCO_CLASSES,
            image_size=[IMAGE_SIZE, IMAGE_SIZE],
        )
    except Exception as e:
        pytest.skip(f"Failed to load TRT engine: {e}")


@pytest.fixture(scope="module")
def cpu_models():
    rf_model = RFDETRSegSmall(pretrain_weights=PTH_FILE, device="cpu")

    od_pth = ObjectDetector(
        metadata=dict(version="v1", model_name="rfdetr", task="od", framework="rfdetr"),
        model_path=PTH_FILE,
        device="cpu",
        class_map=COCO_CLASSES,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
    )
    return rf_model, od_pth


KEYS = ["boxes", "scores", "masks", "segments", "classes"]


def _assert_empty_out(out, keys=None):
    """Assert all specified per-image output keys are empty."""
    if keys is None:
        keys = KEYS
    for k in keys:
        assert len(out[k]) == 0, f"Expected empty out['{k}'], got {len(out[k])}"


def _assert_lengths_equal(out, keys=None):
    """Assert all specified per-image output keys have equal lengths."""
    if keys is None:
        keys = KEYS
    lengths = [len(out[k]) for k in keys]
    assert len(set(lengths)) == 1, f"Unequal lengths: {dict(zip(keys, lengths))}"


def _assert_nonempty_out(out, keys=None):
    """Assert all specified per-image output keys are non-empty and have equal lengths."""
    if keys is None:
        keys = KEYS
    _assert_lengths_equal(out, keys)
    for k in keys:
        assert len(out[k]) > 0, f"Expected non-empty out['{k}']"


def _assert_scores_geq(out, min_conf):
    """Assert all per-image scores are >= min_conf."""
    for sc in out["scores"]:
        assert sc >= min_conf, f"Score {sc} < {min_conf}"


def assert_outputs_match_rf(rf_preds, outputs, label):
    rf_boxes = rf_preds.xyxy
    rf_masks = rf_preds.mask
    rf_classes = [COCO_CLASSES[c] for c in rf_preds.class_id]
    assert rf_boxes.shape[0] == outputs["boxes"].shape[0], f"{label}: Number of boxes mismatch"
    assert np.array_equal(rf_boxes, outputs["boxes"]), f"{label}: Box coordinates mismatch"
    assert np.array_equal(rf_preds.confidence, outputs["scores"]), f"{label}: Confidence scores mismatch"
    assert np.array_equal(rf_classes, outputs["classes"]), f"{label}: Class labels mismatch"
    if rf_masks is not None:
        assert np.array_equal(rf_masks, outputs["masks"]), f"{label}: Masks mismatch"


def test_nonsquare_image_size_rejected():
    """A non-square image_size must fail at load, not on the first frame in production."""
    with pytest.raises(ValueError, match="square resolution"):
        RfdetrModel(PTH_FILE, device="cpu", class_map=COCO_CLASSES, image_size=[IMAGE_SIZE, IMAGE_SIZE + 24])


def test_class_map_inferred_from_model(imgs_coco, obj_detector):
    """With no class_map, names come from the model itself and must match an explicit COCO map."""
    inferred = RfdetrModel(PTH_FILE, device=DEVICE, image_size=[IMAGE_SIZE, IMAGE_SIZE])
    assert inferred.class_map == COCO_CLASSES

    img = cv2.resize(imgs_coco[0], (IMAGE_SIZE, IMAGE_SIZE))
    out, _ = inferred.predict(img, configs=0.5)
    ref, _ = obj_detector.predict(img, configs=0.5)
    assert np.array_equal(ref["classes"][0], out["classes"][0])


def test_nonpositive_batch_size_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        RfdetrModel(PTH_FILE, device="cpu", class_map=COCO_CLASSES, image_size=[IMAGE_SIZE, IMAGE_SIZE], batch_size=0)


def test_batch_size_chunks_and_pads(imgs_coco):
    """batch_size=2 over 3 images must chunk and zero-pad the short last chunk, returning one result per image."""
    imgs = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in imgs_coco[:3]]
    batched = RfdetrModel(PTH_FILE, device=DEVICE, class_map=COCO_CLASSES, image_size=[IMAGE_SIZE, IMAGE_SIZE], batch_size=2)
    assert batched.fixed_batch_size == 2

    out, _ = batched.predict(imgs, configs=0.5)
    assert len(out["boxes"]) == len(imgs)
    for i in range(len(imgs)):
        _assert_nonempty_out({k: out[k][i] for k in KEYS})


def test_variant_read_from_checkpoint():
    """The default path resolves the variant from the checkpoint, matching an explicit model_type."""
    inferred = RfdetrModel(PTH_FILE, device="cpu", class_map=COCO_CLASSES)
    override = RfdetrModel(PTH_FILE, model_type=MODEL_TYPE, device="cpu", class_map=COCO_CLASSES)
    assert type(inferred.model) is type(override.model)
    assert inferred.image_size == override.image_size == (IMAGE_SIZE, IMAGE_SIZE)


def test_resolution_comes_from_the_checkpoint(tmp_path):
    """A model trained at a non-default resolution must run at it, not at the variant's default."""
    path = str(tmp_path / "trained_at_432.pth")
    ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
    ckpt["model_config"]["resolution"] = 432
    torch.save(ckpt, path)

    inferred = RfdetrModel(path, device="cpu", class_map=COCO_CLASSES)
    override = RfdetrModel(path, model_type=MODEL_TYPE, device="cpu", class_map=COCO_CLASSES)
    assert inferred.image_size == override.image_size == (432, 432)


def test_resolution_survives_a_stripped_checkpoint(tmp_path):
    """rfdetr strips model_config out of checkpoint_best_total.pth; the position grid must still give the resolution."""
    path = str(tmp_path / "stripped.pth")
    ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
    del ckpt["model_config"]
    key = "backbone.0.encoder.encoder.embeddings.position_embeddings"
    position = ckpt["model"][key]
    ckpt["model"][key] = torch.zeros(1, 36 * 36 + 1, position.shape[2], dtype=position.dtype)  # 36 patches of 12 px
    torch.save(ckpt, path)

    assert RfdetrModel(path, device="cpu", class_map=COCO_CLASSES).image_size == (432, 432)


def test_explicit_image_size_still_wins(tmp_path):
    """image_size is an override; the checkpoint's own resolution must not shadow it."""
    path = str(tmp_path / "trained_at_432.pth")
    ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
    ckpt["model_config"]["resolution"] = 432
    torch.save(ckpt, path)

    model = RfdetrModel(path, device="cpu", class_map=COCO_CLASSES, image_size=[IMAGE_SIZE, IMAGE_SIZE])
    assert model.image_size == (IMAGE_SIZE, IMAGE_SIZE)


def test_checkpoint_without_variant_asks_for_model_type(tmp_path):
    """Starter weights record no variant; the error must name model_type rather than surface rfdetr's KeyError."""
    path = str(tmp_path / "no_variant.pth")
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="Specify model_type explicitly"):
        RfdetrModel(path, device="cpu", class_map=COCO_CLASSES)


def test_model_class_comparison(obj_detector):
    direct = RfdetrModel(PTH_FILE, device=DEVICE, class_map=COCO_CLASSES, image_size=[IMAGE_SIZE, IMAGE_SIZE])
    api = obj_detector
    assert type(direct) is type(api), f"direct={type(direct).__name__}, api={type(api).__name__}"


class Test_Rfdetr_Model:
    def test_compare_with_rfdetr(self, imgs_coco, cpu_models):
        "Use cpu to avoid gpu non-determinism issues."

        rf_model, pth_model = cpu_models

        for img in imgs_coco:
            resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            rf_preds = rf_model.predict(resized, threshold=0.5)

            batch_pth, _ = pth_model.predict(resized, configs=0.5)
            outputs_pth = {k: batch_pth[k][0] for k in ("boxes", "scores", "classes", "masks")}

            assert_outputs_match_rf(rf_preds, outputs_pth, "pth_model")

    @pytest.mark.xfail(
        RFDETR_ANTIALIASES_PREDICT,
        reason="rfdetr < 1.9.0 antialiases in predict(); preprocess() follows the training resize instead",
        strict=False,
    )
    def test_compare_with_rfdetr_nonsquare(self, imgs_coco, cpu_models):
        """Non-square inputs exercise the off-size resize guard.

        Our guard fits inputs to the square model input with an antialias-free stretch on the float tensor, matching
        rfdetr >= 1.9.0's own predict(), so detections must match exactly. Only this test resizes; the square case
        is already model-sized.
        """

        rf_model, pth_model = cpu_models

        for i, img in enumerate(imgs_coco):
            rh, rw = OFF_SIZES[i % len(OFF_SIZES)]
            resized = cv2.resize(img, (rw, rh))
            rf_preds = rf_model.predict(resized, threshold=0.5)

            batch_pth, _ = pth_model.predict(resized, configs=0.5)
            outputs_pth = {k: batch_pth[k][0] for k in ("boxes", "scores", "classes", "masks")}

            assert_outputs_match_rf(rf_preds, outputs_pth, "pth_model")

    def test_warmup(self, obj_detector):
        obj_detector.warmup()

    def test_empty(self, obj_detector):
        empty_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        batch_outputs, _ = obj_detector.predict(empty_img, configs=0.5)
        out = {k: v[0] for k, v in batch_outputs.items()}
        _assert_empty_out(out)

    def test_confidence(self, imgs_coco, obj_detector):
        img = cv2.resize(imgs_coco[0], (IMAGE_SIZE, IMAGE_SIZE))
        batch_outputs, _ = obj_detector.predict(img, configs=1.0)
        out = {k: v[0] for k, v in batch_outputs.items()}
        _assert_empty_out(out)

    def test_operators(self, imgs_coco, obj_detector):
        for idx, img in enumerate(imgs_coco):
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            from lmi_utils.preprocess_utils.ops import ResizeMeta

            operators = [ResizeMeta(src_sizes=[[w, h]], dst_sizes=[[IMAGE_SIZE, IMAGE_SIZE]], pads=[[0, 0, 0, 0]])]

            batch_outputs, _ = obj_detector.predict(img_resized, configs=0.5, operators=operators, return_segments=False)
            out = {k: v[0] for k, v in batch_outputs.items()}
            _assert_nonempty_out(out, ["boxes", "scores", "classes", "masks"])
            _assert_empty_out(out, ["segments"])
            _assert_scores_geq(out, 0.5)

            # boxes with operators should be scaled to the original image size
            assert np.all(out["boxes"][:, 0] <= w)
            assert np.all(out["boxes"][:, 1] <= h)
            assert np.all(out["boxes"][:, 2] <= w)
            assert np.all(out["boxes"][:, 3] <= h)

            annotated_image = obj_detector.annotate_image(out, img)
            out_name = f"out_operators_{idx}.jpg"
            os.makedirs(OUT_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(OUT_DIR, out_name), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    def test_operators_batch(self, imgs_coco, obj_detector):
        original_sizes = [(img.shape[1], img.shape[0]) for img in imgs_coco]  # (w, h)
        from lmi_utils.preprocess_utils.ops import ResizeMeta

        operators = [
            ResizeMeta(
                src_sizes=[[w, h] for w, h in original_sizes],
                dst_sizes=[[IMAGE_SIZE, IMAGE_SIZE] for _ in original_sizes],
                pads=[[0, 0, 0, 0] for _ in original_sizes],
            )
        ]
        imgs_resized = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in imgs_coco]

        batch_outputs, _ = obj_detector.predict(imgs_resized, configs=0.5, operators=operators)

        assert len(batch_outputs["boxes"]) == len(imgs_coco)

        os.makedirs(OUT_DIR, exist_ok=True)
        for idx, img in enumerate(imgs_coco):
            w, h = original_sizes[idx]
            out = {k: v[idx] for k, v in batch_outputs.items()}
            _assert_nonempty_out(out)
            _assert_scores_geq(out, 0.5)

            # boxes/masks with operators should be scaled back to the original image size
            assert np.all(out["boxes"][:, [0, 2]] <= w)
            assert np.all(out["boxes"][:, [1, 3]] <= h)
            assert out["masks"].shape[1] == h
            assert out["masks"].shape[2] == w

            annotated_image = obj_detector.annotate_image(out, img)
            out_name = f"out_operators_batch_{idx}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, out_name), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    def test_tensor_input(self, imgs_coco, obj_detector):
        """predict() accepts uint8 CUDA HWC tensors and returns the same detections as numpy."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img_np = cv2.resize(imgs_coco[0], (IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = torch.from_numpy(img_np).cuda()

        out_np, _ = obj_detector.predict(img_np, configs=0.5)
        out_tensor, _ = obj_detector.predict(img_tensor, configs=0.5)

        assert out_np.keys() == out_tensor.keys()
        assert len(out_np["boxes"][0]) == len(out_tensor["boxes"][0])

    def test_tensor_input_batch(self, imgs_coco, obj_detector):
        """predict() accepts a list of uint8 CUDA HWC tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        imgs_resized = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in imgs_coco]
        tensor_batch = [torch.from_numpy(img).cuda() for img in imgs_resized]

        out, _ = obj_detector.predict(tensor_batch, configs=0.5)
        assert len(out["boxes"]) == len(imgs_coco)


# A class count that differs from rfdetr's config default so the checkpoint/model mismatch path
# (and its warning) is exercised, mirroring a fine-tuned FSP model with few classes.
NUM_CLASSES_CANARY = 7


class _WarningCollector(logging.Handler):
    """Collects WARNING+ messages emitted on a logger."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture(scope="module")
def finetuned_checkpoint(tmp_path_factory):
    """A checkpoint whose detection head is resized to NUM_CLASSES_CANARY classes.

    Built from the installed rfdetr so the saved parameter names and head layout track the pinned
    version — the point the canary tests below guard.
    """
    model = RFDETRSegSmall(pretrain_weights=PTH_FILE, device="cpu")
    assert hasattr(model.model.model, "reinitialize_detection_head"), (
        "rfdetr no longer exposes reinitialize_detection_head; the detection-head layout that "
        "RfdetrBase._num_classes_from_checkpoint depends on may have changed"
    )
    model.model.model.reinitialize_detection_head(NUM_CLASSES_CANARY + 1)  # +1 for the background row
    path = str(tmp_path_factory.mktemp("rf_detr_ckpt") / "finetuned.pth")
    torch.save({"model": model.model.model.state_dict(), "args": {}}, path)
    return path


class Test_Num_Classes_From_Checkpoint:
    """Guards RfdetrBase._num_classes_from_checkpoint against rfdetr checkpoint-format changes.

    The wrapper reads a checkpoint's class count and passes it to rfdetr as num_classes so a
    fine-tuned model loads without the "Checkpoint has N classes but model is configured for 90"
    warning. That relies on rfdetr storing the head bias as class_embed.bias with a single
    background row above the class rows. These tests fail loudly if a version bump breaks either
    assumption instead of silently regressing the suppression.
    """

    def test_helper_reads_head_count_from_both_layouts(self, finetuned_checkpoint, tmp_path):
        state_dict = torch.load(finetuned_checkpoint, map_location="cpu", weights_only=False)["model"]
        assert "class_embed.bias" in state_dict, (
            "rfdetr no longer stores the detection-head bias as class_embed.bias; update "
            "RfdetrBase._num_classes_from_checkpoint to match the new checkpoint format"
        )
        assert state_dict["class_embed.bias"].shape[0] == NUM_CLASSES_CANARY + 1, (
            "class_embed.bias no longer has exactly one background row above the class rows; the "
            "'-1' convention in RfdetrBase._num_classes_from_checkpoint is no longer valid"
        )

        # BestModelCallback / legacy layout: raw keys under "model" (the fixture file itself).
        assert RfdetrBase._num_classes_from_checkpoint(finetuned_checkpoint) == NUM_CLASSES_CANARY

        # PyTorch Lightning native .ckpt layout: "model."-prefixed keys under "state_dict".
        ptl_path = str(tmp_path / "ptl.ckpt")
        torch.save({"state_dict": {f"model.{k}": v for k, v in state_dict.items()}}, ptl_path)
        assert RfdetrBase._num_classes_from_checkpoint(ptl_path) == NUM_CLASSES_CANARY

    def test_helper_returns_none_when_head_bias_absent(self, tmp_path):
        path = str(tmp_path / "no_head.pth")
        torch.save({"model": {"backbone.weight": torch.zeros(3)}}, path)
        assert RfdetrBase._num_classes_from_checkpoint(path) is None

    def test_helper_count_suppresses_mismatch_warning(self, finetuned_checkpoint):
        """End-to-end: the count the helper returns must actually silence rfdetr's warning.

        Loading through the installed rfdetr ties the helper to rfdetr's loader, so a version bump
        that alters the head convention or the warning fails here rather than regressing quietly.
        """
        from rfdetr.utilities.logger import get_logger

        num_classes = RfdetrBase._num_classes_from_checkpoint(finetuned_checkpoint)
        assert num_classes == NUM_CLASSES_CANARY

        rf_logger = get_logger()

        def _mismatch_warned(**kwargs) -> bool:
            collector = _WarningCollector()
            rf_logger.addHandler(collector)
            try:
                RFDETRSegSmall(pretrain_weights=finetuned_checkpoint, device="cpu", **kwargs)
            finally:
                rf_logger.removeHandler(collector)
            return any("Checkpoint has" in m and "configured for" in m for m in collector.messages)

        assert _mismatch_warned(), (
            "expected rfdetr to warn about the class-count mismatch when num_classes is not passed; "
            "the premise RfdetrBase._num_classes_from_checkpoint addresses no longer holds"
        )
        assert not _mismatch_warned(num_classes=num_classes), (
            "passing the checkpoint's own class count did not suppress rfdetr's mismatch warning; "
            "rfdetr changed its head or loader behavior and the wrapper's suppression is broken"
        )


def test_clamp_boxes_to_image():
    """The shared postprocess must return boxes clamped to the image (rfdetr's PostProcess clamps).

    Backend-agnostic: built without an engine, feed synthetic raw head outputs (one query with an
    oversized cxcywh box that decodes past every edge, one inside) through RfdetrBase.postprocess.
    """
    from rfdetr.models.postprocess import PostProcess

    from object_detectors.rf_detr_lmi.model import RfdetrTRT

    model = object.__new__(RfdetrTRT)
    model.device = torch.device("cpu")
    model._init_common()
    model.postprocessor = PostProcess(num_select=300)  # normally from the engine's metadata
    model._setup_class_map({0: "person"})

    image_h, image_w = 100, 200
    images = [np.zeros((image_h, image_w, 3), dtype=np.uint8)]
    # cxcywh normalized: query 0 is 2x the image (decodes to [-0.5,-0.5,1.5,1.5]); query 1 is inside.
    pred_boxes = torch.zeros((1, 2, 4))
    pred_boxes[0, 0] = torch.tensor([0.5, 0.5, 2.0, 2.0])
    pred_boxes[0, 1] = torch.tensor([0.5, 0.5, 0.2, 0.2])
    pred_logits = torch.full((1, 2, 1), 10.0)  # both queries confidently class 0
    outputs = [pred_boxes, pred_logits]

    results = model.postprocess(outputs, images=images, configs=0.0, return_segments=False)
    out = results[0].boxes

    assert (out[:, 0::2] >= 0).all() and (out[:, 0::2] <= image_w).all()
    assert (out[:, 1::2] >= 0).all() and (out[:, 1::2] <= image_h).all()
    # the oversized query is clamped to the full image frame
    full = torch.tensor([0.0, 0.0, float(image_w), float(image_h)])
    assert any(torch.allclose(b, full) for b in out)
