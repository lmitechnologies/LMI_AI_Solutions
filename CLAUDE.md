# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMI AI Solutions is a Python monorepo providing unified wrappers for AI/ML model frameworks used in industrial computer vision: object detection, anomaly detection, and classification.

## Commands

### Installation
```bash
pip install -e .
source lmi_ai.env                            # Set PYTHONPATH for scripts
git submodule update --init --recursive      # After cloning
```

### Testing
```bash
pytest tests/lmi_utils
pytest tests/object_detectors
pytest tests/classifiers
pytest tests/anomaly_detectors/anomalib_lmi/test_v0.py
pytest tests/anomaly_detectors/anomalib_lmi/test_v1.py
pytest tests/anomaly_detectors/anomalib_lmi/test_v2.py
bash tests/run_tests.sh v1-all  # Also: od, utils, cls, ad-v1, ad-v2
```

### Linting
```bash
ruff check . && ruff format .
pre-commit run --all-files
```

Line length 140, Python 3.8+, double quotes, rules E/F/I/B. Pre-commit runs Ruff automatically on commit.

## Architecture

### Registry / Factory Pattern

All three domains (`od_core/`, `ad_core/`, `cls_core/`) share the same pattern: framework wrappers register themselves with metadata (`framework`, `model_name`, `task`, `version`), and a top-level factory class (`ObjectDetector`, `AnomalyDetector`, etc.) instantiates the correct backend at runtime.

### Subclass Contract

Every backend implements exactly four abstract methods — `warmup`, `preprocess`, `forward`, `postprocess` — and the base class orchestrates the full inference pipeline.

### ADBase (`ad_core/ad_base.py`)

- `predict(image, batch_size=None)` — accepts a single image, list, or BHWC numpy/tensor batch. When `batch_size` is set, delegates to `_run_batched_predict` which chunks inputs **before** preprocessing. Set `self.fixed_batch_size` on TRT subclasses for automatic zero-padding.
- `annotate(img, ad_scores, ad_threshold, ad_max)` — GPU-accelerated turbo-colormap heatmap overlay; returns `uint8` HWC numpy.
- `colormap_tensor` — lazily initialized `[256, 3]` turbo LUT on `self.device`.

### Anomalib_Base (`anomalib_lmi/base.py`)

Subclasses `ADBase`; shared by v1 and v2 backends. Adds:
- `_load_tensorrt_model` — loads TRT engine, sets `fixed_batch_size`, shape, fp16.
- `convert(model_path, export_path, fp16=True, convert_type="trt")` — `.pt` → ONNX → TRT.
- `test(images_path, ...)` — evaluation with gamma-fit threshold suggestions, CSV stats, optional tiling, annotated outputs.

### ODBase (`od_core/od_base.py`)

- `predict(image, configs, operators=None, batch_size=None)` — returns `(results_dict, time_info)`. `results_dict` keys: `boxes`, `scores`, `classes`, `masks`, `segments`, `points` (each a per-image list). Fixed-batch TRT engines are zero-padded.
- `annotate_image(results, image, ...)` — draws boxes, masks, segments, keypoints; handles OBB.
- Helpers: `_parse_confidence_config`, `_apply_confidence_filter`, `_normalize_operators`, `_revert_coordinates`, `_aggregate_results`.

`Results` (`od_core/results.py`) — all numeric fields stored as `torch.Tensor`; defaults to empty tensors so `to_dict()` is safe with zero detections.

### CI/CD

Three Docker test scenarios (Python 3.10, linux/amd64 + arm64): `no_ad` (utils/od/cls), `ad_v1` (Anomalib v1.1.1), `ad_v2` (Anomalib v2.2.0). Images tagged `py310-{arch}-{version}` on GHCR.

### Legacy / Do Not Modify

- `yolov5_lmi/` (git submodule), `anomalib_lmi/v0/`, `deprecated/` — do not modify unless explicitly instructed.
- `tf_objdet/` folder may remain; TensorFlow OD support has been dropped.

### Versioning

Automated semantic versioning via `.releaserc.json`. Default branch is `ais`.
