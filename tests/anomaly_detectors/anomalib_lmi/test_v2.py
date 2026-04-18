import glob
import logging
import os
import platform
import tempfile
from typing import List

import cv2
import numpy as np
import pytest
import torch
from anomalib.data.utils import read_image
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer

from anomaly_detectors.ad_core.anomaly_detector import AnomalyDetector
from anomaly_detectors.anomalib_lmi.convert_to_torchscript import convert_v2_torchscript
from anomaly_detectors.anomalib_lmi.v2.model import AnomalyModel as AnomalyModelV2

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


@pytest.fixture(scope="module")
def ad_models():
    ad1 = AnomalyModelV2(MODEL_PATH, device=DEVICE)
    ad2 = AnomalyDetector(BASE_CONFIG, device=DEVICE)
    return [ad1, ad2]


@pytest.fixture(scope="module")
def cpu_models():
    ad1 = AnomalyModelV2(MODEL_PATH, device="cpu")
    ad2 = AnomalyDetector(BASE_CONFIG, device="cpu")
    return [ad1, ad2]


def compare_results(anomalib_model: TorchInferencer, ais_models: List[AnomalyModelV2]):
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
            assert np.allclose(pred, pred2, atol=1e-5)


def test_compare_results_with_anomalib(cpu_models):
    """
    compare prediction results between current implementation and anomalib
    """
    model1 = TorchInferencer(MODEL_PATH, device="cpu")
    model2 = cpu_models[0]
    model3 = cpu_models[1]
    model4 = AnomalyModelV2(TRACED_MODEL_PATH, device="cpu")
    compare_results(model1, [model2, model3, model4])


@pytest.mark.parametrize("warmup_size", [[672, 640], [256, 224]])
def test_warmup_api(ad_models, warmup_size: List[int]):
    """
    Test warmup with different input dimensions.
    """
    ad = ad_models[1]
    ad.warmup()
    ad.warmup(warmup_size)


def test_model_api(ad_models):
    """
    Test AnomalyDetector API with and without resize arguments.
    """
    ad = ad_models[1]
    ad.test(DATA_PATH, OUTPUT_PATH)


def test_convert_to_torchscript():
    with tempfile.TemporaryDirectory() as t:
        outpath = os.path.join(t, "trace.pt")
        convert_v2_torchscript(MODEL_PATH, outpath, device="cpu")
        assert os.path.isfile(outpath)

        model = AnomalyModelV2(outpath, device="cpu")
        inp = torch.randint(0, 255, (256, 256, 3), dtype=torch.uint8)
        model.predict(inp)

        if USE_GPU:
            outpath = os.path.join(t, "trace_gpu.pt")
            convert_v2_torchscript(MODEL_PATH, outpath, device="cuda")
            assert os.path.isfile(outpath)

            model = AnomalyModelV2(outpath, device="cuda")
            model.predict(inp.cuda())


def test_predict_input_variants():
    """Test predict with different input formats (numpy, torch tensor, grayscale)."""
    ad = AnomalyModelV2(MODEL_PATH, device=DEVICE)

    # Numpy RGB
    img_np = np.zeros((224, 224, 3), dtype=np.uint8)
    res1 = ad.predict(img_np)[0]
    assert res1.shape == (224, 224)

    # Grayscale
    img_gray = np.zeros((224, 224), dtype=np.uint8)
    res3 = ad.predict(img_gray)[0]
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

    # BHWC numpy array → list of numpy arrays
    bhwc = np.stack(imgs_np)  # [N,H,W,C]
    results_bhwc = ad.predict(bhwc)
    assert isinstance(results_bhwc, list) and len(results_bhwc) == n_images
    for r in results_bhwc:
        assert isinstance(r, np.ndarray) and r.shape == (224, 224)

    # batch_size kwarg chunks the inference but output must be identical
    results_chunked = ad.predict(imgs_np, batch_size=max(1, n_images // 2))
    assert len(results_chunked) == n_images
    for r_ref, r_chunk in zip(results, results_chunked):
        atol = 1e-2 if IS_ARM else 1e-5
        assert np.allclose(r_ref, r_chunk, atol=atol)


@pytest.mark.parametrize("n_images", [1, 2, 4])
def test_predict_gpu_batch(ad_models, n_images):
    if not USE_GPU:
        pytest.skip("GPU not available, skipping GPU batch test.")

    ad = ad_models[0]
    imgs_np = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(n_images)]
    bhwc = np.stack(imgs_np)  # [N,H,W,C]

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

    for r1, r2 in zip(results_bhwc_gpu, results_gpu):
        assert torch.allclose(r1, r2, atol=1e-5)
