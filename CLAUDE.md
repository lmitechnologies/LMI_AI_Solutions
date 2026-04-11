# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMI AI Solutions is a Python monorepo providing unified wrappers and utilities for AI/ML model frameworks used in industrial computer vision: object detection, anomaly detection, and classification.

## Commands

### Installation
```bash
pip install -e .                # Development install
source lmi_ai.env               # Set PYTHONPATH for running scripts directly
```

### Testing
```bash
pytest tests/lmi_utils
pytest tests/object_detectors
pytest tests/classifiers
pytest tests/anomaly_detectors/anomalib_lmi/test_anomaly_model_v1.py
pytest tests/anomaly_detectors/anomalib_lmi/test_anomaly_model_v2.py
bash tests/run_tests.sh         # Run all tests via script
```

### Linting & Formatting
```bash
ruff check .
ruff format .
pre-commit run --all-files      # Run all pre-commit hooks (Ruff + large file check)
```

Pre-commit hooks run Ruff automatically on commit. Line length is 140, target Python 3.8+, double quotes, rules E/F/I/B enforced.

## Architecture

### Registry / Factory Pattern

The core design across all model domains (object detection, anomaly detection, classification) uses a **registry + factory** pattern:

- `od_core/`, `ad_core/`, `cls_core/` define base classes and a registry
- Each framework wrapper (e.g., `ultralytics_lmi/`, `detectron2_lmi/`) registers itself with the registry using metadata: `framework`, `model_name`, `task`, `version`
- The top-level factory class (`ObjectDetector`, etc.) uses the registry to instantiate the correct backend at runtime based on these keys

### Module Structure

| Module | Purpose |
|--------|---------|
| `lmi_utils/` | Shared utilities: image, data, dataset, label (incl. `json_to_factory`), eval, pre/post-processing, point cloud, pipeline base, system utils |
| `lmi_common/` | Cross-domain shared code (`YoloCore` wrapping Ultralytics AutoBackend, shared by both detectors and classifiers) |
| `object_detectors/` | Object detection: Ultralytics YOLO (v8–v12 / yolo26), Detectron2, RF-DETR; YOLOv5 (legacy, do not modify) |
| `anomaly_detectors/` | Anomaly detection: Anomalib v1.1.1 and v2.2.0 wrappers |
| `classifiers/` | Classification: Ultralytics YOLO classifier (`ultralytics_lmi/yolo`); legacy `yolov8_cls` in `deprecated/` |

### Object Detection Base Class (`od_core/`)

The `ODBase` class in `od_core/od_base.py` implements the **shared inference pipeline** used by all object detection backends. Subclasses only need to implement `warmup`, `preprocess`, `forward`, and `postprocess` — the rest is handled by the base:

- `predict(image, configs, operators)` — full pipeline: preprocess → forward → postprocess, with support for single images, lists, and BHWC numpy arrays. Fixed batch sizes (e.g. TRT engines) are handled via zero-padding.
- `annotate_image(results, image, ...)` — draws bounding boxes / masks on an image.
- `_apply_confidence_filter` / `_compute_thresholds` — per-class confidence filtering.
- `_normalize_operators` / `_revert_coordinates` — coordinate reversion via operator chains.
- `_parse_confidence_config` — normalizes a float or dict config into a per-class threshold dict.
- `_aggregate_results` — collects a list of `Results` objects into a batched output dict.

`Results` (`od_core/results.py`) stores all numeric fields (`boxes`, `scores`, `masks`, `points`, `segments`) as **`torch.Tensor`** internally. `boxes`, `scores`, and `classes` always default to empty tensors/lists so `to_dict()` is safe even with zero detections.

### Git Submodules

External dependencies (YOLOv5, Anomalib forks) are included as git submodules under `*/submodules/`. After cloning, run `git submodule update --init --recursive`.

> Note: The EfficientNet, TF OD API models, and PaddleOCR submodules have been removed. The `tf_objdet/` folder may remain but TensorFlow Object Detection API support has been dropped.

### Supported Model Backends

- **Ultralytics YOLO** (v8–v12, "yolo26" generation): detection, segmentation, pose, OBB, classification — v0 support dropped; `yolov8_lmi` and `yolov8_cls` moved to `deprecated/`
- **YOLOv5** (custom fork submodule) — **LEGACY: do not modify unless explicitly instructed**
- **Detectron2** (Facebook Research)
- **RF-DETR**
- ~~**TensorFlow Object Detection API**~~ — removed
- **Anomalib** (v1.x and v2.x — separate class hierarchies due to API differences)
- **SAM2** (Segment Anything Model 2)
- ~~**PaddleOCR**~~ — removed

### Versioning & Releases

Automated semantic versioning via `.releaserc.json`. Default branch is `ais`. CI/CD in `.github/workflows/`.