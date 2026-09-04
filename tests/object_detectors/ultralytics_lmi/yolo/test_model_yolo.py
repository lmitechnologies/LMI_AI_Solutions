import logging
import os

import cv2
import numpy as np
import pytest
import torch
from ultralytics import YOLO

from object_detectors.od_core.object_detector import ObjectDetector
from object_detectors.ultralytics_lmi.yolo.model import Yolo, YoloObb, YoloPose, YoloSeg

logger = logging.getLogger(__name__)


COCO_DIR = "tests/assets/images/coco"
DOTA8_DIR = "tests/assets/images/dota8"
OUT_DIR = "tests/outputs/od/ultralytics/yolo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = [640, 640]
OBB_IMGSZ = [1024, 1024]
OFF_SIZES = [(500, 661), (576, 704), (704, 512)]  # (h, w), non-square

OD_DET_MODELS = [
    "tests/assets/models/od/ultralytics/yolo26n.pt",
    "tests/assets/models/od/ultralytics/yolo11n.pt",
    "tests/assets/models/od/ultralytics/yolov8n.pt",
]

OD_SEG_MODELS = [
    "tests/assets/models/od/ultralytics/yolo26n-seg.pt",
    "tests/assets/models/od/ultralytics/yolo11n-seg.pt",
    "tests/assets/models/od/ultralytics/yolov8n-seg.pt",
]

OD_OBB_DOTA_8 = [
    "tests/assets/models/od/ultralytics/yolo26n-obb.pt",
    "tests/assets/models/od/ultralytics/yolo11n-obb.pt",
    "tests/assets/models/od/ultralytics/yolov8n-obb.pt",
]

OD_POSE_MODELS = [
    "tests/assets/models/od/ultralytics/yolo26n-pose.pt",
    "tests/assets/models/od/ultralytics/yolo11n-pose.pt",
    "tests/assets/models/od/ultralytics/yolov8n-pose.pt",
]


def _model_name(path):
    return os.path.splitext(os.path.basename(path))[0]


@pytest.fixture(scope="module")
def yolo_models():
    models = {}
    keys = ["det", "seg", "obb_dota8", "pose"]
    model_lists = [
        OD_DET_MODELS,
        OD_SEG_MODELS,
        OD_OBB_DOTA_8,
        OD_POSE_MODELS,
    ]
    model_classes = [Yolo, YoloSeg, YoloObb, YoloPose]
    image_sizes = [IMGSZ, IMGSZ, OBB_IMGSZ, IMGSZ]
    for k, ml, mc, imsz in zip(keys, model_lists, model_classes, image_sizes):
        instances = []
        for path in ml:
            m = mc(path, device=DEVICE, image_size=imsz)
            m.test_name = _model_name(path)
            instances.append(m)
        models[k] = instances
    return models


@pytest.fixture(scope="module")
def yolo_models_api():
    models = {}
    keys = ["det", "seg", "obb_dota8", "pose"]
    tasks = ["od", "seg", "obb", "pose"]
    model_lists = [
        OD_DET_MODELS,
        OD_SEG_MODELS,
        OD_OBB_DOTA_8,
        OD_POSE_MODELS,
    ]
    image_sizes = [IMGSZ, IMGSZ, OBB_IMGSZ, IMGSZ]
    for k, ml, task, imsz in zip(keys, model_lists, tasks, image_sizes):
        instances = []
        for path in ml:
            m = ObjectDetector(
                metadata=dict(
                    version="v1",
                    model_name="yolov8" if "yolov8n" in path else "yolov11",
                    task=task,
                    framework="ultralytics",
                    model_path=path,
                    image_size=imsz,
                ),
                device=DEVICE,
            )
            m.test_name = _model_name(path)
            instances.append(m)
        models[k] = instances
    return models


@pytest.fixture(scope="module")
def all_models(yolo_models_api):
    return yolo_models_api


def test_model_class_comparison(yolo_models, yolo_models_api):
    for key in yolo_models:
        for d, a in zip(yolo_models[key], yolo_models_api[key]):
            assert type(d) is type(a), f"{key} [{d.test_name}]: direct={type(d).__name__}, api={type(a).__name__}"


def load_image(path):
    im = cv2.imread(path)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb


@pytest.fixture(scope="module")
def imgs_coco():
    return _load_images(COCO_DIR, 640)


@pytest.fixture(scope="module")
def imgs_dota8():
    return _load_images(DOTA8_DIR, 1024)


def _load_images(directory, im_dim):
    """Load and resize images from a directory.

    Returns:
        (images, resized, ops) where ``ops`` is a unified preprocessing history with
        per-image metadata (one ``resize`` entry whose metadata has length == batch).
    """
    from lmi_utils.preprocess_utils.ops import ResizeMeta

    paths = [os.path.join(directory, img) for img in os.listdir(directory)]
    images, resized_images = [], []
    src_sizes, dst_sizes, pads = [], [], []
    for p in paths:
        if "png" not in p and "jpg" not in p:
            continue
        rgb = load_image(p)
        h, w = rgb.shape[:2]
        resized_images.append(cv2.resize(rgb, (im_dim, im_dim)))
        images.append(rgb)
        src_sizes.append([w, h])
        dst_sizes.append([im_dim, im_dim])
        pads.append([0, 0, 0, 0])
    ops = [ResizeMeta(src_sizes=src_sizes, dst_sizes=dst_sizes, pads=pads)]
    return images, resized_images, ops


def _nonsquare_batch(images):
    """Resize each image to a cycling non-square size and build a matching unified history.

    Returns ``(resized, ops)`` where ``ops`` is a single-entry ``[ResizeMeta(...)]`` whose
    per-image metadata lists have length == batch (same shape as ``_load_images``).
    """
    from lmi_utils.preprocess_utils.ops import ResizeMeta

    resized, src_sizes, dst_sizes, pads = [], [], [], []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        rh, rw = OFF_SIZES[i % len(OFF_SIZES)]
        resized.append(cv2.resize(img, (rw, rh)))
        src_sizes.append([w, h])
        dst_sizes.append([rw, rh])
        pads.append([0, 0, 0, 0])
    ops = [ResizeMeta(src_sizes=src_sizes, dst_sizes=dst_sizes, pads=pads)]
    return resized, ops


def _shared_ops(per_image_history):
    """Take a per-image history and produce a single-image (broadcast) variant.

    Used to exercise the "single chain applied to all images" code path: each
    history entry's per-image fields are collapsed to their first element.
    """
    from dataclasses import fields

    out = []
    for entry in per_image_history:
        fresh = type(entry).__new__(type(entry))
        for f in fields(entry):
            v = getattr(entry, f.name)
            object.__setattr__(fresh, f.name, v[:1] if isinstance(v, list) else v)
        out.append(fresh)
    return out


def _assert_empty_output(out, keys, batch_size=1):
    """Assert each key in out has batch_size items and all are empty."""
    for key in keys:
        assert len(out[key]) == batch_size
        for i in range(batch_size):
            assert len(out[key][i]) == 0


def _assert_batch_counts(out, keys, n=None):
    """Assert each key in out has exactly n items."""
    if n is None:
        return
    for key in keys:
        assert len(out[key]) == n


def _assert_batch_scores(out, key, min_conf):
    """Assert all scores in a batch output are >= min_conf."""
    for sc_list in out[key]:
        for sc in sc_list:
            assert sc >= min_conf


def _assert_batch_nonempty(out, keys):
    """Assert each per-image entry for every key is non-empty."""
    for key in keys:
        for i, item in enumerate(out[key]):
            assert len(item) > 0, f"out['{key}'][{i}] is empty"


def _assert_batch_output(out, keys, n=None, min_conf=0.5):
    """Assert batch size, minimum scores, and non-empty entries for every key."""
    _assert_batch_counts(out, keys, n)
    _assert_batch_scores(out, "scores", min_conf)
    _assert_batch_nonempty(out, keys)


def _write_annotated_images(model, out, images, filename_prefix):
    """Annotate each image with detection results and write to OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for img_idx, img in enumerate(images):
        per_img = {k: v[img_idx] for k, v in out.items()}
        im_out = model.annotate_image(per_img, img)
        im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT_DIR, f"{filename_prefix}-{img_idx}.png"), im_out)


def _assert_batch_cuda(out, keys, idx=0):
    """Assert all tensors at the given image index across keys are on CUDA."""
    for tensors in zip(*[out[k][idx] for k in keys]):
        for t in tensors:
            assert t.is_cuda


class Test_Yolo_Det:
    KEYS = ["boxes", "scores", "classes"]

    def test_compare_with_ultralytics_nonsquare(self, imgs_coco):
        images, _, _ = imgs_coco
        resized_images, _ = _nonsquare_batch(images)
        for model_path in OD_DET_MODELS:
            ults_model = YOLO(model_path)
            our_model = Yolo(model_path, device="cpu", image_size=IMGSZ)
            batch_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in resized_images]

            results = ults_model(batch_bgr, conf=0.5, iou=0.4, max_det=300, device="cpu")
            out, _ = our_model.predict(resized_images, configs=0.5, iou=0.4, max_det=300)
            for ults_result, our_boxes, our_scores in zip(results, out["boxes"], out["scores"]):
                ults_out = ults_result.cpu().numpy()
                assert np.array_equal(np.array(our_boxes), ults_out.boxes.xyxy)
                assert np.array_equal(np.array(our_scores), ults_out.boxes.conf)

    def test_warmup(self, all_models):
        for model in all_models["det"]:
            model.warmup()

    def test_predict_batch_empty(self, all_models):
        batch = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2
        for model in all_models["det"]:
            out, _ = model.predict(batch, configs=0.5)
            _assert_empty_output(out, self.KEYS, batch_size=2)

    def test_predict_batch(self, all_models, imgs_coco):
        images, _, _ = imgs_coco
        if len(images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(images)
        resized_images, ops_list = _nonsquare_batch(images)

        for model in all_models["det"]:
            # per-image operators
            model.predict(resized_images, configs=0.5, operators=ops_list)

            # shared operators (single chain broadcast to all images)
            model.predict(resized_images, configs=0.5, operators=_shared_ops(ops_list))

            # no operators
            model.predict(resized_images, configs=0.5)

            if torch.cuda.is_available():
                tensor_batch = [torch.from_numpy(img).cuda() for img in resized_images]
                out_gpu, _ = model.predict(tensor_batch, configs=0.5, operators=ops_list)
                for img_idx in range(num_imgs):
                    _assert_batch_cuda(out_gpu, self.KEYS[:-1], img_idx)
                _write_annotated_images(model, out_gpu, images, filename_prefix=model.test_name)

    def test_predict_batch_square(self, all_models, imgs_coco):
        _, resized_images, ops_list = imgs_coco
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(resized_images)
        for model in all_models["det"]:
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)
            _assert_batch_output(out, self.KEYS, num_imgs)

    def test_predict_batch_invalid_operators(self, yolo_models, imgs_coco):
        _, resized_images, _ops_list = imgs_coco
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        model = yolo_models["det"][0]
        # Per-image metadata length doesn't match batch size (and != 1, so no broadcast).
        from lmi_utils.preprocess_utils.ops import ResizeMeta

        # 7 entries — not 1 (broadcast) and not batch size, so should raise.
        bad = [ResizeMeta(src_sizes=[[10, 10]] * 7, dst_sizes=[[640, 640]] * 7, pads=[[0, 0, 0, 0]] * 7)]
        with pytest.raises(ValueError, match="not 1"):
            model.predict(resized_images, configs=0.5, operators=bad)

    def test_insize_input_no_warning(self, imgs_coco, caplog):
        """An in-size input (== image_size) is passed through with no resize warning."""
        _, resized_images, _ = imgs_coco
        model = Yolo(OD_DET_MODELS[0], device="cpu", image_size=IMGSZ)
        with caplog.at_level(logging.WARNING):
            model.predict([resized_images[0]], configs=0.5)
        assert not [r for r in caplog.records if "model input" in r.message]


class Test_Yolo_Seg:
    KEYS = ["boxes", "masks", "scores", "segments", "classes"]

    def test_compare_with_ultralytics_nonsquare(self, imgs_coco):
        images, _, _ = imgs_coco
        resized_images, _ = _nonsquare_batch(images)
        for model_path in OD_SEG_MODELS:
            ults_model = YOLO(model_path)
            our_model = YoloSeg(model_path, device="cpu", image_size=IMGSZ)
            batch_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in resized_images]
            results = ults_model(batch_bgr, conf=0.5, iou=0.4, max_det=300, retina_masks=True, device="cpu")
            out, _ = our_model.predict(resized_images, configs=0.5, iou=0.4, max_det=300, return_segments=True)
            for ults_result, our_boxes, our_scores, our_masks, our_segs in zip(
                results, out["boxes"], out["scores"], out["masks"], out["segments"]
            ):
                ults_out = ults_result.cpu().numpy()
                assert np.array_equal(np.array(our_boxes), ults_out.boxes.xyxy)
                assert np.array_equal(np.array(our_scores), ults_out.boxes.conf)
                assert np.array_equal(np.array(our_masks), ults_out.masks.data)
                assert len(our_segs) == len(ults_result.masks.xy)
                for s1, s2 in zip(our_segs, ults_result.masks.xy):
                    assert np.array_equal(s1, s2)

    def test_warmup(self, all_models):
        for model in all_models["seg"]:
            model.warmup()

    def test_predict_batch_empty(self, all_models):
        batch = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2
        for model in all_models["seg"]:
            out, _ = model.predict(batch, configs=0.5)
            _assert_empty_output(out, self.KEYS, batch_size=2)

    def test_predict_batch(self, all_models, imgs_coco):
        images, _, _ = imgs_coco
        if len(images) < 2:
            pytest.skip("Not enough images for batch test")

        num_images = len(images)
        resized_images, batch_ops = _nonsquare_batch(images)
        for model in all_models["seg"]:
            # per-image operators
            out, _ = model.predict(resized_images, configs=0.5, operators=batch_ops, return_segments=False)
            for img_idx in range(num_images):
                assert len(out["segments"][img_idx]) == 0

            # shared operators (single chain broadcast to all images)
            model.predict(resized_images, configs=0.5, operators=_shared_ops(batch_ops))

            # no operators
            model.predict(resized_images, configs=0.5)

            if torch.cuda.is_available():
                tensor_batch = [torch.from_numpy(img).cuda() for img in resized_images]
                out_gpu, _ = model.predict(tensor_batch, configs=0.5, operators=batch_ops)
                for img_idx in range(num_images):
                    _assert_batch_cuda(out_gpu, self.KEYS[:-1], img_idx)
                _write_annotated_images(model, out_gpu, images, filename_prefix=model.test_name)

    def test_predict_batch_square(self, all_models, imgs_coco):
        _, resized_images, ops_list = imgs_coco
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(resized_images)
        for model in all_models["seg"]:
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)
            _assert_batch_output(out, self.KEYS, num_imgs)


class Test_Yolo_Obb:
    KEYS = ["boxes", "scores", "classes"]

    def test_compare_with_ultralytics_nonsquare(self, imgs_dota8):
        images, _, _ = imgs_dota8
        for model_path in OD_OBB_DOTA_8:
            ults_model = YOLO(model_path)
            our_model = YoloObb(model_path, device="cpu", image_size=OBB_IMGSZ)
            resized_images, _ = _nonsquare_batch(images)
            batch_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in resized_images]
            results = ults_model(batch_bgr, conf=0.5, iou=0.4, max_det=300, device="cpu")
            out, _ = our_model.predict(resized_images, configs=0.5, iou=0.4, max_det=300)
            for ults_result, our_boxes, our_scores in zip(results, out["boxes"], out["scores"]):
                ults_out = ults_result.cpu().numpy()
                assert np.allclose(np.array(our_boxes), ults_out.obb.xyxyxyxy, atol=1e-5)
                assert np.array_equal(np.array(our_scores), ults_out.obb.conf)

    def test_warmup_dota8(self, all_models):
        for model in all_models["obb_dota8"]:
            model.warmup()

    def test_predict_batch_empty(self, all_models):
        batch = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2
        for model in all_models["obb_dota8"]:
            out, _ = model.predict(batch, configs=0.5)
            _assert_empty_output(out, self.KEYS, batch_size=2)

    def test_predict_batch(self, all_models, imgs_dota8):
        images, _, _ = imgs_dota8
        if len(images) < 2:
            pytest.skip("Not enough images for batch test")
        resized_images, ops_list = _nonsquare_batch(images)

        for model in all_models["obb_dota8"]:
            # per-image operators
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)

            # shared operators (single chain broadcast to all images)
            model.predict(resized_images, configs=0.5, operators=_shared_ops(ops_list))

            # no operators
            out3, _ = model.predict(resized_images, configs=0.5)

            if torch.cuda.is_available():
                tensor_batch = [torch.from_numpy(img).cuda() for img in resized_images]
                out_gpu, _ = model.predict(tensor_batch, configs=0.5, operators=ops_list)
                _write_annotated_images(model, out_gpu, images, filename_prefix=model.test_name)

    def test_predict_batch_square(self, all_models, imgs_dota8):
        _, resized_images, ops_list = imgs_dota8
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(resized_images)
        for model in all_models["obb_dota8"]:
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)
            _assert_batch_output(out, self.KEYS, num_imgs)


class Test_Yolo_Pose:
    KEYS = ["boxes", "scores", "points", "classes"]

    def test_compare_with_ultralytics_nonsquare(self, imgs_coco):
        images, _, _ = imgs_coco
        resized_images, _ = _nonsquare_batch(images)
        for model_path in OD_POSE_MODELS:
            ults_model = YOLO(model_path)
            our_model = YoloPose(model_path, device="cpu", image_size=IMGSZ)
            batch_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in resized_images]
            results = ults_model(batch_bgr, conf=0.5, iou=0.4, max_det=300, device="cpu")
            out, _ = our_model.predict(resized_images, configs=0.5, iou=0.4, max_det=300)
            for ults_result, our_boxes, our_scores, our_points in zip(results, out["boxes"], out["scores"], out["points"]):
                ults_out = ults_result.cpu().numpy()
                assert np.array_equal(np.array(our_boxes), ults_out.boxes.xyxy)
                assert np.array_equal(np.array(our_scores), ults_out.boxes.conf)
                assert np.array_equal(np.array(our_points), ults_out.keypoints.data)

    def test_warmup(self, all_models):
        for model in all_models["pose"]:
            model.warmup()

    def test_predict_batch_empty(self, all_models):
        batch = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2
        for model in all_models["pose"]:
            out, _ = model.predict(batch, configs=0.5)
            _assert_empty_output(out, self.KEYS, batch_size=2)

    def test_predict_batch(self, all_models, imgs_coco):
        # No _assert_batch_output here because pose is more sensitive to distortion and drop some detections.
        images, _, _ = imgs_coco
        if len(images) < 2:
            pytest.skip("Not enough images for batch test")
        resized_images, ops_list = _nonsquare_batch(images)
        for model in all_models["pose"]:
            # per-image operators
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)

            # shared operators (single chain broadcast to all images)
            model.predict(resized_images, configs=0.5, operators=_shared_ops(ops_list))

            # no operators
            out3, _ = model.predict(resized_images, configs=0.5)

            if torch.cuda.is_available():
                tensor_batch = [torch.from_numpy(img).cuda() for img in resized_images]
                out_gpu, _ = model.predict(tensor_batch, configs=0.5, operators=ops_list)
                _write_annotated_images(model, out_gpu, images, filename_prefix=model.test_name)

    def test_predict_batch_square(self, all_models, imgs_coco):
        _, resized_images, ops_list = imgs_coco
        if len(resized_images) < 2:
            pytest.skip("Not enough images for batch test")
        num_imgs = len(resized_images)
        for model in all_models["pose"]:
            out, _ = model.predict(resized_images, configs=0.5, operators=ops_list)
            _assert_batch_output(out, self.KEYS, num_imgs)


def test_clamp_boxes_to_image(yolo_models):
    """Detection boxes must be clamped to the original image (ultralytics scale_boxes clips)."""
    model = yolo_models["det"][0]
    image_h, image_w = 100, 200
    orig_img = np.zeros((image_h, image_w, 3), dtype=np.uint8)
    net_h, net_w = model.image_size
    img = torch.zeros((1, 3, net_h, net_w))
    confs = {name: 0.0 for name in model.model.names.values()}
    # A box spanning the whole network canvas maps well past the (smaller) original frame.
    pred = torch.tensor([[0.0, 0.0, float(net_w), float(net_h), 0.9, 0.0]])

    result, _ = model.construct_result(pred, img, orig_img, confs)
    boxes = result.boxes

    assert len(boxes) == 1
    assert (boxes[:, 0::2] >= 0).all() and (boxes[:, 0::2] <= image_w).all()
    assert (boxes[:, 1::2] >= 0).all() and (boxes[:, 1::2] <= image_h).all()


def test_clamp_keypoints_to_image(yolo_models):
    """Pose keypoint xy must be clamped to the image while the visibility column is preserved."""
    model = yolo_models["pose"][0]
    image_h, image_w = 100, 200
    orig_img = np.zeros((image_h, image_w, 3), dtype=np.uint8)
    net_h, net_w = model.image_size
    img = torch.zeros((1, 3, net_h, net_w))
    confs = {name: 0.0 for name in model.model.names.values()}
    n_kpt = model.model.kpt_shape[0]

    box = torch.tensor([[10.0, 10.0, 100.0, 100.0, 0.9, 0.0]])
    kpts = torch.zeros((1, n_kpt, 3))
    kpts[..., 0] = float(net_w)  # x at the far edge of the network canvas
    kpts[..., 1] = float(net_h)  # y at the far edge
    kpts[..., 2] = 0.7  # visibility must survive untouched
    pred = torch.cat([box, kpts.reshape(1, -1)], dim=1)

    result, _ = model.construct_result(pred, img, orig_img, confs)
    pts = result.points

    assert (pts[..., 0] >= 0).all() and (pts[..., 0] <= image_w).all()
    assert (pts[..., 1] >= 0).all() and (pts[..., 1] <= image_h).all()
    assert torch.allclose(pts[..., 2], torch.full_like(pts[..., 2], 0.7))


def _tile_and_predict(model, images, conf, **tile_kwargs):
    """Tile ``images``, run the detector on the tiles, and merge back to image space."""
    from lmi_utils.preprocess_utils import steps
    from lmi_utils.preprocess_utils.preprocessor import Preprocessor

    net_h, net_w = model.image_size
    configs = [
        steps.tile(tile_size=[320, 320], scale_mode="padding", **tile_kwargs),
        steps.resize(width=net_w, height=net_h, preserve_aspect=False),
    ]
    tiles, history = Preprocessor().preprocess(images, configs)
    out, _ = model.predict(tiles, configs=conf, operators=history)
    return tiles, out


def test_tiled_predict_merges_tiles_back_to_source_images(yolo_models, imgs_coco):
    """predict() takes the tiles but returns one entry per source image, in source coordinates."""
    model = yolo_models["det"][0]
    images = imgs_coco[0][:2]

    tiles, out = _tile_and_predict(model, images, 0.25, stride=[320, 320])

    assert len(tiles) > len(images), "tiling must produce more images than it was given"
    for key in ("boxes", "scores", "classes"):
        assert len(out[key]) == len(images), f"out['{key}'] has {len(out[key])} entries, expected {len(images)}"

    for idx, img in enumerate(images):
        h, w = img.shape[:2]
        boxes = out["boxes"][idx]
        assert len(boxes) == len(out["scores"][idx]) == len(out["classes"][idx])
        if len(boxes):
            assert boxes[:, 0::2].min() >= 0 and boxes[:, 1::2].min() >= 0
            assert boxes[:, 0::2].max() <= w and boxes[:, 1::2].max() <= h


def test_tiled_predict_merge_fragments_rejoins_seam_splits(yolo_models, imgs_coco):
    """Overlapping tiles plus merge_fragments must not leave more detections than the split run."""
    model = yolo_models["det"][0]
    images = imgs_coco[0][:1]
    overlapping = {"stride": [256, 256], "nms_iou": 0.45, "containment": 0.8}

    _, split = _tile_and_predict(model, images, 0.25, merge_fragments=False, **overlapping)
    _, merged = _tile_and_predict(model, images, 0.25, merge_fragments=True, **overlapping)

    assert len(merged["boxes"][0]) < len(split["boxes"][0]), "merging should collapse seam fragments"


def test_tiled_predict_rejects_a_tile_count_that_does_not_match(yolo_models, imgs_coco):
    from lmi_utils.preprocess_utils import steps
    from lmi_utils.preprocess_utils.preprocessor import Preprocessor

    model = yolo_models["det"][0]
    images = imgs_coco[0][:1]
    tiles, history = Preprocessor().preprocess(images, [steps.tile(tile_size=[320, 320], stride=[320, 320], scale_mode="padding")])

    with pytest.raises(ValueError, match="tiles, but"):
        model.predict(tiles[:-1], configs=0.25, operators=history)
