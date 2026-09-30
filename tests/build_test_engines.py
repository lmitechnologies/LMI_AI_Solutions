"""Rebuild the OD TensorRT test engines from committed weights, for the current platform.

``.engine`` files are gitignored: TensorRT plan files are platform-specific — tied to the exact
GPU architecture and TensorRT build they were serialized on — and are **not** portable (a foreign
engine fails to deserialize with a "platform tag mismatch"). So they are not committed; regenerate
them locally for your platform before running the engine-gated TRT suites, which otherwise skip:

    tests/object_detectors/detectron2_lmi/test_trt.py    (detectron2)
    tests/object_detectors/rf_detr_lmi/test_trt.py        (rf_detr)
    tests/object_detectors/ultralytics_lmi/yolo/test_trt.py (yolo)
    tests/anomaly_detectors/anomalib_lmi/test_v1.py       (ad_v1)  / test_v2.py (ad_v2)

Run this inside the test container (see ``tests/dockerfile.tests``) from the committed weights/ONNX:

    python -m tests.build_test_engines --backend all
    python -m tests.build_test_engines --backend rf_detr,ad_v1   # a comma-separated subset
    python -m tests.build_test_engines --backend detectron2 --no-fp16

``tests/run_tests.sh`` calls this automatically before each suite with ``--skip-existing
--if-available``, so engines are built on demand:
    --skip-existing : leave engines that already exist untouched.
    --if-available  : exit 0 when no GPU/TensorRT is present (the TRT tests then skip). With a GPU,
                      a failed build still exits non-zero, after trying the remaining backends.

Requires a GPU and TensorRT. Per-backend deps: rf_detr needs ``rfdetr``; detectron2 needs
``detectron2`` plus ``onnx-graphsurgeon`` (EfficientNMS graph surgery); yolo needs ``ultralytics``. The ad_v1/ad_v2 engines
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
DET2_IMAGE_SIZE = (800, 800)  # INPUT.MIN_SIZE_TEST in the committed config.yaml
DET2_ONNX = os.path.join(DET2_DIR, "model.onnx")
DET2_ENGINE = os.path.join(DET2_DIR, "model.engine")

YOLO_DIR = os.path.join(ASSETS, "ultralytics")
# model → image size, matching IMGSZ/OBB_IMGSZ in ultralytics_lmi/yolo/test_model_yolo.py; yolo26 heads are NMS-free, yolo11 use NMS
YOLO_MODELS = {
    "yolo26n.pt": 640,
    "yolo11n-seg.pt": 640,
    "yolo26n-pose.pt": 640,
    "yolo11n-pose.pt": 640,
    "yolo26n-obb.pt": 1024,
    "yolo11n-obb.pt": 1024,
}
YOLO_ENGINES = [os.path.join(YOLO_DIR, os.path.splitext(m)[0] + ".engine") for m in YOLO_MODELS]

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
    try:
        onnx_to_trt(onnx_path, RF_DETR_ENGINE, fp16=fp16)  # static batch: batch kwargs are ignored
    finally:
        if not keep_onnx and os.path.isfile(onnx_path):
            os.remove(onnx_path)
    logger.info("[rf_detr] done: %s", RF_DETR_ENGINE)


def build_detectron2(fp16: bool = True, keep_onnx: bool = False) -> None:
    """Export the Mask R-CNN weights → ONNX (+ EfficientNMS graph surgery) → TensorRT engine.

    Uses the committed config.yaml, at a square MIN_SIZE_TEST engine size.
    """
    for path in (DET2_WEIGHTS, DET2_CONFIG_FILE):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing detectron2 asset: {path} (fetch via git-lfs).")
    from object_detectors.detectron2_lmi.convert import convert

    logger.info("[detectron2] exporting ONNX + building engine → %s ...", DET2_ENGINE)
    try:
        convert(
            {
                "config_file": DET2_CONFIG_FILE,
                "weights": DET2_WEIGHTS,
                "image_size": DET2_IMAGE_SIZE,
                "output": DET2_DIR,
                "batch_size": 1,
                "fp16": fp16,
                "trt": True,
            }
        )
    finally:
        if not keep_onnx and os.path.isfile(DET2_ONNX):
            os.remove(DET2_ONNX)
    logger.info("[detectron2] done: %s", DET2_ENGINE)


def build_yolo(fp16: bool = True, keep_onnx: bool = False, skip_existing: bool = False) -> None:
    """Export the YOLO test models → TensorRT with ultralytics' exporter, writing <name>.engine next to each .pt."""
    # ultralytics' auto-install would put the CPU onnxruntime over the GPU build
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    for (name, imgsz), engine in zip(YOLO_MODELS.items(), YOLO_ENGINES):
        if skip_existing and os.path.isfile(engine):
            logger.info("[yolo] engine exists, skipping: %s", engine)
            continue
        pt = os.path.join(YOLO_DIR, name)
        onnx_path = os.path.splitext(pt)[0] + ".onnx"
        logger.info("[yolo] exporting %s → TensorRT ...", pt)
        try:
            YOLO(pt).export(format="engine", imgsz=imgsz, half=fp16, device=0, verbose=False)
        finally:
            if not keep_onnx and os.path.isfile(onnx_path):
                os.remove(onnx_path)
        logger.info("[yolo] done: %s", engine)


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
    "yolo": build_yolo,
    "ad_v1": build_ad_v1,
    "ad_v2": build_ad_v2,
}

# Output engine paths per backend — used to skip backends whose engines already exist.
ENGINE_PATHS = {
    "rf_detr": [RF_DETR_ENGINE],
    "detectron2": [DET2_ENGINE],
    "yolo": YOLO_ENGINES,
    "ad_v1": [AD_V1_ENGINE],
    "ad_v2": [AD_V2_ENGINE],
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
    ap.add_argument("--skip-existing", action="store_true", help="Skip engines that already exist.")
    ap.add_argument(
        "--if-available",
        action="store_true",
        help="No-op (exit 0) when a GPU/TensorRT is not available. Use when auto-building before tests.",
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
        if args.skip_existing and all(os.path.isfile(p) for p in ENGINE_PATHS[name]):
            logger.info("[%s] engines exist, skipping: %s", name, ", ".join(ENGINE_PATHS[name]))
            continue
        # yolo builds several engines, so it skips the existing ones itself
        extra = {"skip_existing": args.skip_existing} if name == "yolo" else {}
        try:
            BUILDERS[name](fp16=args.fp16, keep_onnx=args.keep_onnx, **extra)
        except Exception:
            logger.exception("[%s] build failed", name)
            failed.append(name)

    if failed:
        raise SystemExit(f"Engines not built: {', '.join(failed)}")


if __name__ == "__main__":
    main()
