# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMI AI Solutions is a Python monorepo providing unified wrappers and utilities for AI/ML model frameworks used in industrial computer vision: object detection, anomaly detection, classification, and OCR.

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
| `lmi_utils/` | Shared utilities: image, data, dataset, label, eval, pre/post-processing, point cloud, pipeline base |
| `lmi_common/` | Cross-domain shared code (e.g., `YoloCore` wrapping Ultralytics AutoBackend) |
| `object_detectors/` | Object detection: Ultralytics YOLO (v8+), YOLOv5, Detectron2, RF-DETR, TF OD API |
| `anomaly_detectors/` | Anomaly detection: Anomalib v1.1.1 and v2.2.0 wrappers |
| `classifiers/` | Classification: Ultralytics YOLO classification wrapper |
| `ocr_models/` | OCR: PaddleOCR wrapper |

### Git Submodules

External dependencies (YOLOv5, Anomalib forks) are included as git submodules under `*/submodules/`. After cloning, run `git submodule update --init --recursive`.

### Supported Model Backends

- **Ultralytics YOLO** (v8, v9, v11, v12 — "yolo26" generation): detection, segmentation, pose, classification
- **YOLOv5** (custom fork)
- **Detectron2** (Facebook Research)
- **RF-DETR**
- **TensorFlow Object Detection API**
- **Anomalib** (v1.x and v2.x — separate class hierarchies due to API differences)
- **SAM2** (Segment Anything Model 2)

### Versioning & Releases

Automated semantic versioning via `.releaserc.json`. Default branch is `ais`. CI/CD in `.github/workflows/`.