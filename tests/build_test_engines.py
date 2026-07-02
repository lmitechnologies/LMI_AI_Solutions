"""Rebuild the OD TensorRT test engines from committed weights, for the current platform.

``.engine`` files are gitignored: TensorRT plan files are platform-specific — tied to the exact
GPU architecture and TensorRT build they were serialized on — and are **not** portable (a foreign
engine fails to deserialize with a "platform tag mismatch"). So they are not committed; regenerate
them locally for your platform before running the engine-gated TRT suites, which otherwise skip:

    tests/object_detectors/detectron2_lmi/test_trt.py    (rf_detr, detectron2)
    tests/object_detectors/rf_detr_lmi/test_trt.py
    tests/anomaly_detectors/anomalib_lmi/test_v1.py       (ad_v1)  / test_v2.py (ad_v2)

Run this inside the test container (see ``tests/dockerfile.tests``) from the committed weights/ONNX:

    python -m tests.build_test_engines --backend all
    python -m tests.build_test_engines --backend rf_detr,ad_v1   # a comma-separated subset
    python -m tests.build_test_engines --backend detectron2 --no-fp16

``tests/run_tests.sh`` calls this automatically before each suite with ``--skip-existing
--if-available``, so engines are built on demand:
    --skip-existing : leave engines that already exist untouched.
    --if-available  : exit 0 when no GPU/TensorRT is present, and downgrade per-backend build
                      failures to warnings (the corresponding TRT tests then skip) instead of aborting.

Requires a GPU and TensorRT. Per-backend deps: rf_detr needs ``rfdetr``; detectron2 needs
``detectron2`` plus ``onnx-graphsurgeon`` (EfficientNMS graph surgery). The ad_v1/ad_v2 engines
build straight from their committed ONNX and need only TensorRT. Engines are written next to their
weights.
"""

import argparse
import glob
import logging
import os

logger = logging.getLogger("build_test_engines")

ASSETS = "tests/assets/models/od"

RF_DETR_DIR = os.path.join(ASSETS, "rf_detr")
RF_DETR_PTH = os.path.join(RF_DETR_DIR, "rf-detr-seg-small.pth")
RF_DETR_ENGINE = os.path.join(RF_DETR_DIR, "inference_model.engine")
RF_DETR_RESOLUTION = 384  # matches IMAGE_SIZE in rf_detr_lmi/test_model.py

DET2_DIR = os.path.join(ASSETS, "detectron2")
DET2_WEIGHTS = os.path.join(DET2_DIR, "model_final_f10217.pkl")
DET2_CONFIG_FILE = os.path.join(DET2_DIR, "config.yaml")  # committed, resolved Mask R-CNN config
DET2_SAMPLE = os.path.join(DET2_DIR, "sample_image.png")  # committed representative image (read-only)
DET2_TMP_SAMPLE = os.path.join(DET2_DIR, "_anchor_sample.png")  # transient, built + removed here
DET2_ONNX = os.path.join(DET2_DIR, "model.onnx")
DET2_ENGINE = os.path.join(DET2_DIR, "model.engine")

# Anomaly-detection engines build straight from their committed ONNX — no anomalib needed, so both
# variants build in any TensorRT container (../.. path is relative to the OD ``ASSETS`` root).
AD_DIR = os.path.join(ASSETS, "..", "ad")
AD_V1_ONNX = os.path.normpath(os.path.join(AD_DIR, "model_v1", "model.onnx"))
AD_V1_ENGINE = os.path.normpath(os.path.join(AD_DIR, "model_v1", "model.engine"))
AD_V2_ONNX = os.path.normpath(os.path.join(AD_DIR, "model_v2", "model.onnx"))
AD_V2_ENGINE = os.path.normpath(os.path.join(AD_DIR, "model_v2", "model.engine"))


def gpu_available() -> bool:
    """True if a CUDA GPU is visible to torch."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def trt_available() -> bool:
    """True if TensorRT can be imported."""
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


def build_rf_detr(fp16: bool = True, keep_onnx: bool = False) -> None:
    """Export rf-detr-seg-small.pth → ONNX → TensorRT, writing inference_model.engine."""
    if not os.path.isfile(RF_DETR_PTH):
        raise FileNotFoundError(f"Missing weights: {RF_DETR_PTH} (fetch via git-lfs).")
    from rfdetr import RFDETRSegSmall

    from lmi_common.trt_convert import onnx_to_trt
    from object_detectors.rf_detr_lmi.convert import convert_to_onnx

    logger.info("[rf_detr] exporting ONNX from %s ...", RF_DETR_PTH)
    before = set(glob.glob(os.path.join(RF_DETR_DIR, "*.onnx")))
    model = RFDETRSegSmall(pretrain_weights=RF_DETR_PTH, resolution=RF_DETR_RESOLUTION)
    convert_to_onnx(model, RF_DETR_DIR)  # rfdetr names the file itself (e.g. rfdetr-seg-small.onnx)
    produced = sorted(set(glob.glob(os.path.join(RF_DETR_DIR, "*.onnx"))) - before, key=os.path.getmtime)
    if not produced:
        raise RuntimeError(f"rfdetr export produced no .onnx in {RF_DETR_DIR}")
    onnx_path = produced[-1]

    logger.info("[rf_detr] building engine %s → %s ...", os.path.basename(onnx_path), RF_DETR_ENGINE)
    onnx_to_trt(onnx_path, RF_DETR_ENGINE, fp16=fp16)  # static batch: batch kwargs are ignored
    if not keep_onnx and os.path.isfile(onnx_path):
        os.remove(onnx_path)
    logger.info("[rf_detr] done: %s", RF_DETR_ENGINE)


def build_detectron2(fp16: bool = True, keep_onnx: bool = False) -> None:
    """Export the Mask R-CNN weights → ONNX (+ EfficientNMS graph surgery) → TensorRT engine.

    Uses the committed config.yaml and sample_image.png. The sample is resized to a square
    ``MIN_SIZE_TEST`` (divisible by 32) written to a transient file: the ONNX is traced at the
    sample's native size while the surgeon regenerates anchors via ``ResizeShortestEdge(MIN_SIZE_TEST,
    MAX_SIZE_TEST)``, so the two only agree — and the RPN only produces detections — when the sample
    is square at MIN_SIZE_TEST. The committed sample_image.png is read but never modified.
    """
    for path in (DET2_WEIGHTS, DET2_CONFIG_FILE, DET2_SAMPLE):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing detectron2 asset: {path} (fetch via git-lfs).")
    import cv2
    from detectron2.config import get_cfg

    from object_detectors.detectron2_lmi.convert import convert

    cfg = get_cfg()
    cfg.merge_from_file(DET2_CONFIG_FILE)
    size = int(cfg.INPUT.MIN_SIZE_TEST)
    size -= size % 32  # EfficientNMS plugin build requires divisible-by-32 dims
    cv2.imwrite(DET2_TMP_SAMPLE, cv2.resize(cv2.imread(DET2_SAMPLE), (size, size)))

    logger.info("[detectron2] exporting ONNX + building %dx%d engine → %s ...", size, size, DET2_ENGINE)
    try:
        convert(
            {
                "config_file": DET2_CONFIG_FILE,
                "weights": DET2_WEIGHTS,
                "sample_image": DET2_TMP_SAMPLE,
                "output": DET2_DIR,
                "batch_size": 1,
                "fp16": fp16,
                "onnx": True,  # export ONNX + run EfficientNMS graph surgery
                "trt": True,  # then build the engine
            }
        )
    finally:
        os.remove(DET2_TMP_SAMPLE)
    if not keep_onnx and os.path.isfile(DET2_ONNX):
        os.remove(DET2_ONNX)
    logger.info("[detectron2] done: %s", DET2_ENGINE)


def _build_from_onnx(name: str, onnx_path: str, engine_path: str, fp16: bool) -> None:
    """Build a TensorRT engine directly from a committed ONNX file (used by the AD variants)."""
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"Missing ONNX: {onnx_path} (fetch via git-lfs).")
    from lmi_common.trt_convert import onnx_to_trt

    logger.info("[%s] building engine %s → %s ...", name, os.path.basename(onnx_path), engine_path)
    onnx_to_trt(onnx_path, engine_path, fp16=fp16)
    logger.info("[%s] done: %s", name, engine_path)


# AD engines are always FP32 (the incoming fp16 flag is ignored): test_v1/test_v2 compare the TRT
# output against ONNX within atol=0.01, a tolerance FP16 exceeds.
def build_ad_v1(fp16: bool = True, keep_onnx: bool = False) -> None:
    """Build the anomalib v1 test engine (FP32) from its committed model.onnx."""
    _build_from_onnx("ad_v1", AD_V1_ONNX, AD_V1_ENGINE, fp16=False)


def build_ad_v2(fp16: bool = True, keep_onnx: bool = False) -> None:
    """Build the anomalib v2 test engine (FP32) from its committed model.onnx."""
    _build_from_onnx("ad_v2", AD_V2_ONNX, AD_V2_ENGINE, fp16=False)


BUILDERS = {
    "rf_detr": build_rf_detr,
    "detectron2": build_detectron2,
    "ad_v1": build_ad_v1,
    "ad_v2": build_ad_v2,
}

# Output engine path per backend — used to skip backends whose engine already exists.
ENGINE_PATHS = {
    "rf_detr": RF_DETR_ENGINE,
    "detectron2": DET2_ENGINE,
    "ad_v1": AD_V1_ENGINE,
    "ad_v2": AD_V2_ENGINE,
}


def _parse_backends(value: str) -> list:
    """Expand a --backend value ('all' or a comma-separated list) into backend names."""
    if value == "all":
        return list(BUILDERS)
    names = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [n for n in names if n not in BUILDERS]
    if unknown:
        raise SystemExit(f"Unknown backend(s): {', '.join(unknown)}. Choose from: {', '.join(BUILDERS)}, all.")
    return names


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="all", help=f"Backend(s) to build: 'all' or a comma-separated list of {', '.join(BUILDERS)}.")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false", help="Build in FP32 instead of FP16.")
    ap.add_argument("--keep-onnx", action="store_true", help="Keep the intermediate ONNX files.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip backends whose engine file already exists.")
    ap.add_argument(
        "--if-available",
        action="store_true",
        help="No-op (exit 0) when a GPU/TensorRT is not available, "
        "and treat per-backend build failures as warnings. Use when auto-building before tests.",
    )
    args = ap.parse_args()

    if not (gpu_available() and trt_available()):
        msg = "GPU and/or TensorRT not available; skipping engine build."
        if args.if_available:
            logger.warning(msg)
            return
        raise SystemExit(msg)

    failed = []
    for name in _parse_backends(args.backend):
        if args.skip_existing and os.path.isfile(ENGINE_PATHS[name]):
            logger.info("[%s] engine exists, skipping: %s", name, ENGINE_PATHS[name])
            continue
        try:
            BUILDERS[name](fp16=args.fp16, keep_onnx=args.keep_onnx)
        except Exception as e:
            if not args.if_available:
                raise
            logger.warning("[%s] build failed (its TRT tests will skip): %s", name, e)
            failed.append(name)

    if failed:
        logger.warning("Engines not built: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
