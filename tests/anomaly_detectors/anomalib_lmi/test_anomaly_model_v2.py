import glob
import logging
import os
import platform
import tempfile
import time
from typing import List, Tuple

import cv2
import numpy as np
import pytest
import torch
from anomalib.data.utils import read_image
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer

from anomaly_detectors.ad_core.anomaly_detector import AnomalyDetector
from anomaly_detectors.anomalib_lmi.anomaly_model_v2 import AnomalyModel_V2
from anomaly_detectors.anomalib_lmi.convert_to_torchscript import convert_v2_torchscript
from lmi_utils.gadget_utils import pipeline_utils

os.environ["TRUST_REMOTE_CODE"] = "1"

logger = logging.getLogger(__name__)


DATA_PATH = "tests/assets/images/nvtec-ad"
MODEL_PATH = "tests/assets/models/ad/model_v2.pt"
OUTPUT_PATH = "tests/outputs/ad/anomalib_v2"
TRACED_MODEL_PATH = "tests/assets/models/ad/model_v2_trace.pt"

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
BASE_CONFIG = {
    "framework": "anomalib2",
    "model_name": "padim",
    "version": "v2",
    "model_path": MODEL_PATH,
    "task": "seg",
}
IS_ARM = platform.machine().startswith(("arm", "aarch64"))


@pytest.fixture
def test_data():
    paths = glob.glob(os.path.join(DATA_PATH, "*.png"))
    out = []
    names = []
    for p in paths:
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out.append(rgb)
        names.append(os.path.basename(p))
    return out, names


def compare_results(anomalib_model: TorchInferencer, ais_models: List[AnomalyModel_V2]):
    paths = glob.glob(os.path.join(DATA_PATH, "*.png"))
    for p in paths:
        # using anomalib code
        img = read_image(p, as_tensor=False)
        preds = anomalib_model.predict(img)
        pred = preds.anomaly_map
        pred = pred.cpu().numpy().squeeze()

        # using AIS code
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        preds = [model.predict(rgb) for model in ais_models]

        for pred2 in preds:
            if USE_GPU:
                assert np.array_equal(pred, pred2)
            else:
                assert np.allclose(pred, pred2, atol=1e-5)


def test_compare_results_with_anomalib():
    """
    compare prediction results between current implementation and anomalib
    """
    model1 = TorchInferencer(MODEL_PATH, device=DEVICE)
    model2 = AnomalyModel_V2(MODEL_PATH, device=DEVICE)
    if USE_GPU:
        compare_results(model1, [model2])
    else:
        model3 = AnomalyModel_V2(TRACED_MODEL_PATH, device=DEVICE)
        compare_results(model1, [model2, model3])


def test_compare_results_with_anomalib_api():
    """
    compare prediction results between current implementation and anomalib
    """
    model1 = TorchInferencer(MODEL_PATH, device=DEVICE)
    model2 = AnomalyDetector(BASE_CONFIG, device=DEVICE)
    if USE_GPU:
        compare_results(model1, [model2])
    else:
        config = {
            **BASE_CONFIG,
            "model_path": TRACED_MODEL_PATH,
        }  # replace model path with traced model path
        model3 = AnomalyDetector(config, device=DEVICE)
        compare_results(model1, [model2, model3])


@pytest.mark.parametrize("init_args, warmup_size", [((224, 112), [672, 640]), ((), [256, 224])])
def test_warmup(init_args: Tuple, warmup_size: List[int]):
    """
    Test AnomalyModel2 warmup with default and specific sizes.
    """
    ad = AnomalyModel_V2(MODEL_PATH, *init_args, device=DEVICE)
    ad.warmup()
    ad.warmup(warmup_size)


@pytest.mark.parametrize("warmup_size", [[672, 640], [256, 224]])
def test_warmup_api(warmup_size: List[int]):
    """
    Test warmup with different input dimensions.
    """
    ad = AnomalyDetector(BASE_CONFIG, 224, 112, device=DEVICE)
    ad.warmup()
    ad.warmup(warmup_size)


@pytest.mark.parametrize(
    "init_args, sub_dir",
    [
        ((MODEL_PATH, 224, 224, "resize"), "tile-resize"),
        ((MODEL_PATH, 224, 224), "tile-pad"),
        ((MODEL_PATH,), None),
    ],
)
def test_model(init_args: Tuple, sub_dir: str):
    """
    Test AnomalyModel_V2 with various initialization parameters.
    """
    model = AnomalyModel_V2(*init_args, device=DEVICE)

    # specific output path if sub_dir exists, else default OUTPUT_PATH
    save_path = os.path.join(OUTPUT_PATH, sub_dir) if sub_dir else OUTPUT_PATH
    model.test(DATA_PATH, save_path)


@pytest.mark.parametrize("extra_args", [(224, 224, "resize"), ()])
def test_model_api(extra_args: Tuple):
    """
    Test AnomalyDetector API with and without resize arguments.
    """
    ad = AnomalyDetector(BASE_CONFIG, *extra_args, device=DEVICE)
    ad.test(DATA_PATH, OUTPUT_PATH)


def test_annotate(
    test_data,
):
    def old_func(img, ad_scores, ad_threshold, ad_max):
        # Resize AD score to match input image
        h_img, w_img = img.shape[:2]
        ad_scores = pipeline_utils.resize_image(ad_scores, H=h_img, W=w_img)
        # Set all low score pixels to threshold to improve heat map precision
        indices = np.where(ad_scores < ad_threshold)
        ad_scores[indices] = ad_threshold
        # Set upper limit on anomaly score.
        ad_scores[ad_scores > ad_max] = ad_max
        # Generate heat map
        ad_norm = (ad_scores - ad_threshold) / (ad_max - ad_threshold)
        ad_gray = (ad_norm * 255).astype(np.uint8)
        ad_bgr = cv2.applyColorMap(np.expand_dims(ad_gray, -1), cv2.COLORMAP_TURBO)
        residual_rgb = cv2.cvtColor(ad_bgr, cv2.COLOR_BGR2RGB)
        # Overlay anomaly heat map with input image
        annot = cv2.addWeighted(img.astype(np.uint8), 0.6, residual_rgb, 0.4, 0)
        indices = np.where(ad_gray == 0)
        # replace all below-threshold pixels with input image indicating no anomaly
        annot[indices] = img[indices]
        return annot

    ad = AnomalyModel_V2(MODEL_PATH, device=DEVICE)
    for _ in range(1):
        ad.warmup()

    out_path = os.path.join(OUTPUT_PATH, "annotate")
    os.makedirs(out_path, exist_ok=1)

    imgs, names = test_data
    for im, name in zip(imgs, names):
        pred = ad.predict(im)
        mean, max = pred.mean(), pred.max()

        t0 = time.time()
        out1 = old_func(im, pred, mean, max)
        t1 = time.time() - t0

        out2 = ad.annotate(im, pred, mean, max)
        assert np.array_equal(out1, out2)

        bgr = cv2.cvtColor(out2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_path, name), bgr)

        if USE_GPU:
            im = torch.from_numpy(im).cuda()
            pred = torch.from_numpy(pred).cuda()

            t0 = time.time()
            out3 = ad.annotate(im, pred, mean, max)
            t2 = time.time() - t0
            logger.info(f"improved proc time from {t1:.4f} to {t2:.4f}")

            assert np.array_equal(out1, out2)
            assert np.array_equal(out2, out3)


def test_convert_to_torchscript():
    with tempfile.TemporaryDirectory() as t:
        outpath = os.path.join(t, "trace.pt")
        convert_v2_torchscript(MODEL_PATH, outpath, device="cpu")
        assert os.path.isfile(outpath)

        model = AnomalyModel_V2(outpath, device="cpu")
        inp = torch.randint(0, 255, (256, 256, 3), dtype=torch.uint8)
        model.predict(inp)

        if USE_GPU:
            outpath = os.path.join(t, "trace_gpu.pt")
            convert_v2_torchscript(MODEL_PATH, outpath, device="cuda")
            assert os.path.isfile(outpath)

            model = AnomalyModel_V2(outpath, device="cuda")
            model.predict(inp.cuda())


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8])
def test_mini_batch(batch_size):
    """
    Test mini-batch inference combined with tiling.
    """
    ad = AnomalyModel_V2(MODEL_PATH, tile=224, stride=224, device="cpu")
    test_img = np.random.randint(0, 255, (672, 640, 3), dtype=np.uint8)

    result_normal = ad.predict(test_img)
    inference_settings = {"inference_batch_size": batch_size}
    result_batched = ad.predict(test_img, inference_settings=inference_settings, verbose=True)

    logger.info(f"max diff: {np.max(np.abs(result_normal - result_batched))}")
    atol = 5e-2 if IS_ARM else 1e-5
    assert np.allclose(result_normal, result_batched, atol=atol)


def test_predict_invalid_overlap_mode():
    """
    Test that AnomalyModel2.predict raises ValueError for invalid overlap modes.
    """
    ad = AnomalyModel_V2(MODEL_PATH, tile=224, stride=224, device=DEVICE)
    test_img = np.zeros((224, 224, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid overlap mode"):
        ad.predict(test_img, tiling_settings={"overlap_mode": "invalid_mode"})


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_predict_batch_size_edge_cases(batch_size):
    """
    Test mini-batch inference with various batch sizes relative to number of tiles.
    """
    ad = AnomalyModel_V2(MODEL_PATH, tile=224, stride=224, device=DEVICE)
    test_img = np.zeros((448, 448, 3), dtype=np.uint8)

    inference_settings = {"inference_batch_size": batch_size}
    result = ad.predict(test_img, inference_settings=inference_settings)
    assert result.shape == (448, 448)


def test_predict_input_variants():
    """
    Test predict with different input formats (numpy, torch tensor, grayscale).
    """
    ad = AnomalyModel_V2(MODEL_PATH, device=DEVICE)

    # Numpy RGB
    img_np = np.zeros((224, 224, 3), dtype=np.uint8)
    res1 = ad.predict(img_np)
    assert res1.shape == (224, 224)

    # Grayscale
    img_gray = np.zeros((224, 224), dtype=np.uint8)
    res3 = ad.predict(img_gray)
    assert res3.shape == (224, 224)


def test_predict_error_handling(monkeypatch):
    """
    Test that AnomalyModel2.predict raises RuntimeError when _infer returns None.
    """
    ad = AnomalyModel_V2(MODEL_PATH, device=DEVICE)
    test_img = np.zeros((224, 224, 3), dtype=np.uint8)

    # Mock _infer to return None
    monkeypatch.setattr(ad, "_infer", lambda x: None)
    with pytest.raises(RuntimeError, match="Model inference failed to produce output"):
        ad.predict(test_img)

    # Mock _perform_batched_inference to return None
    monkeypatch.setattr(ad, "_perform_batched_inference", lambda x, y: None)
    with pytest.raises(RuntimeError, match="Model inference failed to produce output"):
        ad.predict(test_img, inference_settings={"inference_batch_size": 1})
