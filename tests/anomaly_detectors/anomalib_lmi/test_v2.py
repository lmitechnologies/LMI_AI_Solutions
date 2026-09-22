import glob
import hashlib
import logging
import os
import platform
import subprocess
import sys
import tempfile
from functools import cache
from typing import List

import cv2
import numpy as np
import pytest
import torch
import yaml
from anomalib import __version__ as anomalib_version
from anomalib.data.utils import read_image
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer

from anomaly_detectors.ad_core.anomaly_detector import AnomalyDetector
from anomaly_detectors.anomalib_lmi.convert_to_torchscript import convert_v2_torchscript
from anomaly_detectors.anomalib_lmi.v2.model import AnomalyModel as AnomalyModelV2
from anomaly_detectors.anomalib_lmi.v2.train import build_data, build_preprocessor

os.environ["TRUST_REMOTE_CODE"] = "1"

logger = logging.getLogger(__name__)


DATA_PATH = "tests/assets/images/nvtec-ad"
MODEL_PATH = "tests/assets/models/ad/model_v2/model.pt"
TS_PATH = "tests/assets/models/ad/model_v2/model.ts"
ONNX_PATH = "tests/assets/models/ad/model_v2/model.onnx"
ENGINE_PATH = "tests/assets/models/ad/model_v2/model.engine"
OUTPUT_PATH = "tests/outputs/ad/anomalib_v2"

IS_ARM = platform.machine() in ["aarch64", "arm64"]
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
BASE_CONFIG = {
    "framework": "anomalib2",
    "model_name": "padim",
    "version": "v2",
    "model_path": MODEL_PATH,
    "task": "seg",
}


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
def trt_model():
    if not USE_GPU:
        pytest.skip("GPU not available, skipping TRT model fixture")
    return AnomalyModelV2(ENGINE_PATH, device="cuda")


@pytest.fixture(scope="module")
def cpu_models():
    ad1 = AnomalyModelV2(MODEL_PATH, device="cpu")
    ad2 = AnomalyDetector(BASE_CONFIG, device="cpu")
    ad3 = AnomalyModelV2(ONNX_PATH, device="cpu")
    ad4 = AnomalyModelV2(TS_PATH, device="cpu")
    return [ad1, ad2, ad3, ad4]


def test_model_class_comparison(ad_models):
    direct = ad_models[0]
    api = ad_models[1]
    assert type(direct) is type(api), f"direct={type(direct).__name__}, api={type(api).__name__}"


@cache
def _fixture_digest(path: str) -> str:
    """Short sha256 of a model fixture, so a stale or partial LFS checkout is visible in failures."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{h.hexdigest()[:12]}/{os.path.getsize(path)}B"


def _model_id(model) -> str:
    path = getattr(model, "model_path", None)
    return f"{type(model).__name__}({os.path.basename(path)})" if path else type(model).__name__


def test_compare_with_anomalib(cpu_models):
    """
    compare prediction results between current implementation and anomalib
    """
    anomalib_model = TorchInferencer(MODEL_PATH, device="cpu")
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
        for model in cpu_models:
            pred2 = model.predict(rgb)
            atol = 1e-2 if IS_ARM else 1e-5
            if not np.allclose(pred, pred2, atol=atol, rtol=0.05):
                diff = np.abs(pred - np.squeeze(pred2))
                over = int((diff > atol + 0.05 * np.abs(np.squeeze(pred2))).sum())
                raise AssertionError(
                    f"mismatch for {_model_id(model)} on {os.path.basename(p)}: "
                    f"max|diff|={diff.max():.3e} mean={diff.mean():.3e}, {over}/{diff.size} px over tol "
                    f"(atol={atol}, rtol=0.05) | machine={platform.machine()} IS_ARM={IS_ARM} "
                    f"torch={torch.__version__} anomalib={anomalib_version} gpu={USE_GPU} | fixtures "
                    f"pt={_fixture_digest(MODEL_PATH)} ts={_fixture_digest(TS_PATH)} onnx={_fixture_digest(ONNX_PATH)}"
                )


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
    ad = AnomalyDetector(BASE_CONFIG, device=DEVICE)

    # Numpy RGB
    img_np = np.zeros((224, 224, 3), dtype=np.uint8)
    res1 = ad.predict(img_np)[0]
    assert res1.shape == (224, 224)

    # Grayscale
    img_gray = np.zeros((224, 224), dtype=np.uint8)
    res3 = ad.predict(img_gray)[0]
    assert res3.shape == (224, 224)


@pytest.mark.parametrize("n_images", [1, 2, 4, 7])
def test_predict_batch(cpu_models, n_images):
    """Test predict with a batch of images: list input, BHWC input, and GPU tensors if available."""
    ad = cpu_models[1]
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

    results_chunked = ad.predict(imgs_np, batch_size=max(1, n_images // 2))
    assert len(results_chunked) == n_images
    for r in results_chunked:
        assert isinstance(r, np.ndarray) and r.shape == (224, 224)


@pytest.mark.parametrize("n_images", [1, 2, 4, 7])
def test_predict_gpu_batch(ad_models, n_images):
    if not USE_GPU:
        pytest.skip("GPU not available, skipping GPU batch test.")

    ad = ad_models[1]
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


def test_compare_trt_onnx(trt_model):
    """Compare TRT and ONNX predictions on resized images; tolerates FP16 vs FP32 precision."""
    onnx_model = AnomalyModelV2(ONNX_PATH, device="cuda")
    paths = glob.glob(os.path.join(DATA_PATH, "*.png"))
    for p in paths:
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pred_trt = trt_model.predict(rgb)[0]
        pred_onnx = onnx_model.predict(rgb)[0]
        assert np.allclose(pred_trt, pred_onnx, atol=0.01, rtol=0.05), f"TRT vs ONNX mismatch for {os.path.basename(p)}"


def test_folder_dataset_non_empty():
    """Ensure build_data produces a non-empty samples frame with correct label values.

    Regression guard: under pandas 3 StringDtype, anomalib < 2.3 compared labels against
    DirType members rather than their .value, yielding an empty samples frame.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal normal_dir structure: tmpdir/normal/img.png
        normal_dir = os.path.join(tmpdir, "normal")
        os.makedirs(normal_dir)
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        # Two images so the 0.5 synthetic split yields at least 1 image per subset (floor(2*0.5)=1).
        cv2.imwrite(os.path.join(normal_dir, "img0.png"), dummy)
        cv2.imwrite(os.path.join(normal_dir, "img1.png"), dummy)

        datamodule = build_data(
            {
                "name": "test_dataset",
                "root": tmpdir,
                "normal_dir": "normal",
                "extensions": [".png"],
                "train_batch_size": 1,
                "eval_batch_size": 1,
                "num_workers": 0,
                "test_split_mode": "synthetic",
                "test_split_ratio": 0.5,
                "val_split_mode": "same_as_test",
                "val_split_ratio": 0.5,
            }
        )

        datamodule.setup()
        samples = datamodule.train_data.samples

        assert len(samples) > 0, "Dataset is empty — labels may be stored as 'DirType.NORMAL' instead of 'normal'."

        label_col = "label_index" if "label_index" in samples.columns else "label"
        assert label_col in samples.columns, f"Expected label column not found; columns: {list(samples.columns)}"

        if "label" in samples.columns:
            # Use .value if the stored object is an enum, else fall back to str().
            bad = [v for v in samples["label"].unique() if "DirType" in str(getattr(v, "value", v))]
            assert not bad, f"Labels contain raw StrEnum repr: {bad}."


def test_build_preprocessor():
    pre_processor = build_preprocessor(
        [
            {"class_name": "Resize", "params": {"size": [64, 32]}},
            {"class_name": "Normalize", "params": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        ]
    )

    assert pre_processor.transform is not None
    assert [type(transform).__name__ for transform in pre_processor.transform.transforms] == ["Resize", "Normalize"]


@pytest.mark.parametrize(
    ["tiler_type", "expected"],
    [(None, "CallbackTiler"), ("CallbackTiler", "CallbackTiler"), ("AnomalibTiler", "Tiler")],
)
def test_tiler_type_selects_the_tiler(tiler_type, expected):
    from anomaly_detectors.anomalib_lmi.v2.tiling import TilerConfigCallback

    callback = TilerConfigCallback(enable=True, tile_size=224, stride=112, tiler_class=tiler_type)
    tiler = callback.tiler_class(tile_size=224, stride=112, mode=callback.mode)

    assert type(tiler).__name__ == expected
    tiles = tiler.tile(torch.rand(1, 3, 448, 448))
    assert tuple(tiles.shape) == (9, 3, 224, 224)
    # both tilers untile a feature map back to the feature scale, not the image scale
    assert tuple(tiler.untile(torch.nn.functional.interpolate(tiles, size=(28, 28), mode="nearest")).shape) == (1, 3, 56, 56)


@pytest.mark.parametrize("tiler_type", ["Tiler", "logging", "NotATiler"])
def test_unknown_tiler_type_is_rejected_at_config_time(tiler_type):
    # a globals() lookup also reached imports and the raw Tiler, which has no mode argument and only
    # failed once setup ran, well after the config was accepted
    from anomaly_detectors.anomalib_lmi.v2.tiling import TilerConfigCallback

    with pytest.raises(ValueError, match="Unknown tiler_type"):
        TilerConfigCallback(enable=True, tile_size=224, stride=112, tiler_class=tiler_type)


def _write_padim_config(root, tile_size, stride, image_size=(448, 448)):
    """A config for the training CLI, with the shipped assets copied into the normal_dir layout Folder expects."""
    normal = root / "data" / "train"
    normal.mkdir(parents=True)
    for p in sorted(glob.glob(os.path.join(DATA_PATH, "*good*.png"))):
        cv2.imwrite(str(normal / os.path.basename(p)), cv2.imread(p))
    assert list(normal.iterdir()), f"no training images under {DATA_PATH}"

    params = {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"], "pre_trained": False, "image_size": list(image_size)}
    if tile_size is not None:
        params |= {"tile_size": tile_size, "stride": stride}
    config = {
        "model": {"class_name": "Padim", "params": params},
        "data": {
            "name": "padim_tiling",
            "root": str(root / "data"),
            "normal_dir": "train",
            "extensions": [".png"],
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
            "test_split_mode": "synthetic",
            "test_split_ratio": 0.5,
            "val_split_mode": "same_as_test",
            "val_split_ratio": 0.5,
        },
        "engine": {"max_epochs": 1, "accelerator": "auto", "devices": 1, "default_root_dir": str(root / "out")},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _train_via_cli(config_path):
    """Run the training entry point the way a user does, and return the checkpoint it writes."""
    result = subprocess.run(
        [sys.executable, "-m", "anomaly_detectors.anomalib_lmi.v2.train", "--config", str(config_path), "--skip-mem-estimate"],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0, f"training CLI failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return result


def test_padim_trains_through_the_cli_with_224_tiles_and_112_stride(tmp_path):
    """A 448 image at tile 224 / stride 112 is a 3x3 grid; resnet18 layer1 puts the embedding at 1/4 scale."""
    config_path = _write_padim_config(tmp_path, tile_size=224, stride=112)

    result = _train_via_cli(config_path)
    assert "Tiling enabled: tile_size=224, stride=112, tiler=CallbackTiler" in result.stdout + result.stderr

    ckpt = torch.load(tmp_path / "out" / "model.ckpt", map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    # the gaussian is fit over the untiled embedding grid, so tiling must have round-tripped at feature scale
    assert state["model.gaussian.mean"].shape[-1] == (448 // 4) ** 2
    assert torch.isfinite(state["model.gaussian.mean"]).all()
    assert torch.isfinite(state["model.gaussian.inv_covariance"]).all()


def test_padim_cli_tiled_and_untiled_agree_on_the_embedding_grid(tmp_path):
    # tiling changes what the backbone sees, not the shape contract downstream
    tiled = _train_via_cli(_write_padim_config(tmp_path / "tiled", tile_size=224, stride=112))
    untiled = _train_via_cli(_write_padim_config(tmp_path / "plain", tile_size=None, stride=None))
    assert "Tiling enabled" in tiled.stdout + tiled.stderr
    assert "Tiling enabled" not in untiled.stdout + untiled.stderr

    def mean_shape(name):
        ckpt = torch.load(tmp_path / name / "out" / "model.ckpt", map_location="cpu", weights_only=False)
        return ckpt["state_dict"]["model.gaussian.mean"].shape

    assert mean_shape("tiled") == mean_shape("plain")
