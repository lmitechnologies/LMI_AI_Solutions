import glob
import logging
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Sequence
from typing import List

import cv2
import numpy as np
import pytest
import torch
from anomalib.data.utils import read_image
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer

from anomaly_detectors.ad_core.anomaly_detector import AnomalyDetector
from anomaly_detectors.anomalib_lmi.convert_to_torchscript import convert_v1_torchscript
from anomaly_detectors.anomalib_lmi.v1.model import AnomalyModel as AnomalyModelV1
from lmi_utils.gadget_utils import pipeline_utils

logger = logging.getLogger(__name__)


DATA_PATH = "tests/assets/images/nvtec-ad"
MODEL_PATH = "tests/assets/models/ad/model_v1.pt"
TRACED_MODEL_PATH = "tests/assets/models/ad/model_v1_trace.pt"
ENGINE_PATH = "tests/assets/models/ad/model_v1.engine"
OUTPUT_PATH = "tests/outputs/ad/anomalib_v1"
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
BASE_CONFIG = {
    "framework": "anomalib1",
    "model_name": "padim",
    "version": "v1",
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


@pytest.fixture(scope="module")
def api_model():
    return AnomalyDetector(BASE_CONFIG, device=DEVICE)


@pytest.fixture(scope="module")
def ad_model():
    return AnomalyModelV1(MODEL_PATH, device=DEVICE)


@pytest.fixture(scope="module")
def cpu_models():
    ad1 = AnomalyModelV1(MODEL_PATH, device="cpu")
    ad2 = AnomalyModelV1(TRACED_MODEL_PATH, device="cpu")
    ad_api = AnomalyDetector(BASE_CONFIG, device="cpu")
    return [ad1, ad2, ad_api]


@pytest.fixture(scope="module")
def trt_model():
    if not USE_GPU:
        pytest.skip("GPU not available, skipping TRT model fixture")
    return AnomalyModelV1(ENGINE_PATH, device="cuda")


def compare_results(anomalib_model: TorchInferencer, ais_models: List[AnomalyModelV1]):
    paths = glob.glob(os.path.join(DATA_PATH, "*.png"))
    for p in paths:
        # using anomalib code
        tensor = read_image(p, as_tensor=True)
        pred = anomalib_model.forward(anomalib_model.pre_process(tensor))
        if isinstance(pred, dict):
            pred = pred["anomaly_map"]
        elif isinstance(pred, Sequence):
            pred = pred[1]
        elif isinstance(pred, torch.Tensor):
            pass
        else:
            raise Exception(f"Not supported output: {type(pred)}")
        pred = pred.cpu().numpy().squeeze()

        # using AIS code
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        preds = [model.predict(rgb) for model in ais_models]

        for pred2 in preds:
            assert np.allclose(pred, pred2, atol=1e-5)


def test_compare_results_with_anomalib(cpu_models):
    """
    compare prediction results between current implementation and anomalib
    """
    model1 = TorchInferencer(MODEL_PATH, device="cpu")
    model2, model3, model4 = cpu_models
    compare_results(model1, [model2, model3, model4])


@pytest.mark.parametrize("warmup_size", [[672, 640], [256, 224]])
def test_warmup_api(api_model, warmup_size: List[int]):
    """
    Test warmup with different input dimensions.
    """
    api_model.warmup()
    api_model.warmup(warmup_size)


def test_model_api(api_model):
    api_model.test(DATA_PATH, OUTPUT_PATH)


def test_trt_model(trt_model):
    trt_model.warmup()
    trt_model.test(DATA_PATH, OUTPUT_PATH)


def test_annotate(ad_model, test_data):
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

    ad = ad_model
    for _ in range(1):
        ad.warmup()

    out_path = os.path.join(OUTPUT_PATH, "annotate")
    os.makedirs(out_path, exist_ok=1)

    imgs, names = test_data
    for im, name in zip(imgs, names):
        pred = ad.predict(im)[0]
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
            logger.debug(f"improved proc time from {t1:.4f} to {t2:.4f}")

            assert np.array_equal(out1, out2)
            assert np.array_equal(out2, out3)


def test_convert_to_torchscript():
    with tempfile.TemporaryDirectory() as t:
        outpath = os.path.join(t, "trace.pt")
        convert_v1_torchscript(MODEL_PATH, outpath, device="cpu")
        assert os.path.isfile(outpath)

        model = AnomalyModelV1(outpath, device="cpu")
        inp = torch.randint(0, 255, (256, 256, 3), dtype=torch.uint8)
        model.predict(inp)

        if USE_GPU:
            outpath = os.path.join(t, "trace_gpu.pt")
            convert_v1_torchscript(MODEL_PATH, outpath, device="cuda")
            assert os.path.isfile(outpath)

            model = AnomalyModelV1(outpath, device="cuda")
            model.predict(inp.cuda())


def test_cmds():
    """smoke-test: verify CLI commands run without errors on a single image"""
    with tempfile.TemporaryDirectory() as t:
        my_env = os.environ.copy()
        cmd = f"python -m anomaly_detectors.anomalib_lmi.v1.model test -i {MODEL_PATH} -d {DATA_PATH} \
                -o {str(t)} -g -p --tile 224 224 --stride 224 224 --limit 1"
        logger.info(f"running cmd: {cmd}")
        result = subprocess.run(cmd, shell=True, env=my_env, capture_output=True, text=True)
        logger.info(result.stdout)
        logger.info(result.stderr)

        assert result.returncode == 0, f"Command failed:\n{result.stderr}"
        assert len(glob.glob(os.path.join(t, "*_annot.png"))) == 1

        if USE_GPU:
            t2 = os.path.join(t, "recon")
            cmd = f"python -m anomaly_detectors.anomalib_lmi.v1.model convert -i {MODEL_PATH} -o {t2}"
            logger.info(f"running cmd: {cmd}")
            result = subprocess.run(cmd, shell=True, env=my_env, capture_output=True, text=True)
            logger.info(result.stdout)
            logger.info(result.stderr)

            out_engine = os.path.join(t2, "model.engine")
            assert os.path.isfile(out_engine)


def test_predict_input_variants(ad_model):
    """Test predict with different input formats (numpy, torch tensor, grayscale)."""
    # Numpy RGB
    img_np = np.zeros((224, 224, 3), dtype=np.uint8)
    res1 = ad_model.predict(img_np)[0]
    assert res1.shape == (224, 224)

    # Grayscale
    img_gray = np.zeros((224, 224), dtype=np.uint8)
    res3 = ad_model.predict(img_gray)[0]
    assert res3.shape == (224, 224)


@pytest.mark.parametrize("n_images", [1, 2, 4])
def test_predict_batch(cpu_models, n_images):
    """Test predict with a batch of images: list input, BHWC input, and GPU tensors if available."""
    ad = cpu_models[0]
    imgs_np = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(n_images)]

    # list of numpy arrays → list of numpy arrays
    results = ad.predict(imgs_np)
    assert isinstance(results, list) and len(results) == n_images
    for r in results:
        assert isinstance(r, np.ndarray) and r.shape == (224, 224)

    # batch_size kwarg chunks the inference but output must be identical
    results_chunked = ad.predict(imgs_np, batch_size=max(1, n_images // 2))
    assert len(results_chunked) == n_images
    for r_ref, r_chunk in zip(results, results_chunked):
        atol = 0.05 if IS_ARM else 1e-5
        logger.info(f"max diff: {np.abs(r_ref - r_chunk).max()}")
        assert np.allclose(r_ref, r_chunk, atol=atol)


@pytest.mark.parametrize("n_images", [1, 2, 4])
def test_predict_gpu(ad_model, n_images):
    if not USE_GPU:
        pytest.skip("GPU not available, skipping GPU predict test")

    imgs_np = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(n_images)]
    bhwc = np.stack(imgs_np)  # [N,H,W,C]
    ad = ad_model

    # list of uint8 GPU tensors → list of GPU tensors
    imgs_gpu = [torch.from_numpy(img).cuda() for img in imgs_np]
    results_gpu = ad.predict(imgs_gpu)
    assert isinstance(results_gpu, list) and len(results_gpu) == n_images
    for r in results_gpu:
        assert isinstance(r, torch.Tensor) and r.shape == (224, 224)

    # BHWC uint8 GPU tensor → list of GPU tensors
    bhwc_gpu = torch.from_numpy(bhwc).cuda()
    results_bhwc_gpu = ad.predict(bhwc_gpu)
    assert isinstance(results_bhwc_gpu, list) and len(results_bhwc_gpu) == n_images
    for r in results_bhwc_gpu:
        assert isinstance(r, torch.Tensor) and r.shape == (224, 224)

    for r_gpu1, r_gpu2 in zip(results_bhwc_gpu, results_gpu):
        assert torch.allclose(r_gpu1, r_gpu2, atol=1e-5)
