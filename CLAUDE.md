# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMI AI Solutions is a Python monorepo providing unified wrappers for AI/ML model frameworks used in industrial computer vision: object detection, anomaly detection, and classification.

## Working Rules

- **No assumptions.** When something is unclear, ask before implementing. Do not invent based on guesses.
- **Keep this file concise.** Prefer pointers to source files over duplicated detail. Remove anything derivable from the code itself.
- **Update on significant changes.** When introducing a new domain, base class, top-level pattern, or breaking change to existing architecture, update this file in the same change.
- **Keep docstrings concise.** State the public contract — what it does, args, returns, non-obvious caveats — and stop. Skip internal mechanism and restated implementation. Merge related notes into the relevant arg description rather than adding separate sections.
- **Preserve valid comments.** Do not delete existing comments that still accurately describe the code. Only remove a comment if it is wrong, stale, or made redundant by the change you are making.

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

### Runtime requirements

- TensorRT ≥ 8.5

## Architecture

### Registry / Factory Pattern

All three domains (`object_detectors/od_core/`, `anomaly_detectors/ad_core/`, `classifiers/cls_core/`) share the same pattern: framework wrappers register themselves with metadata (`framework`, `model_name`, `task`, `version`), and a top-level factory class (`ObjectDetector`, `AnomalyDetector`, etc.) instantiates the correct backend at runtime.

### Subclass Contract

Every backend implements exactly four abstract methods — `warmup`, `preprocess`, `forward`, `postprocess` — and the base class orchestrates the full inference pipeline.

### Domain base classes

- AD: `anomaly_detectors/ad_core/ad_base.py` — orchestrates AD inference, GPU heatmap annotation.
- AD (Anomalib): `anomaly_detectors/anomalib_lmi/base.py` — extends `ADBase` with TRT loading, `.pt` → ONNX → TRT export, and evaluation.
- OD: `object_detectors/od_core/od_base.py` — orchestrates OD inference; `Results` (`object_detectors/od_core/results.py`) stores numeric fields as `torch.Tensor` (empty by default, so zero-detection cases are safe).
- CLS: `classifiers/cls_core/cls_base.py` — orchestrates classification inference.

### CI/CD

Three Docker test scenarios (Python 3.10, linux/amd64 + arm64): `no_ad` (utils/od/cls), `ad_v1` (Anomalib v1.1.1), `ad_v2` (Anomalib v2.2.0). Images tagged `py310-{arch}-{version}` on GHCR.

### Legacy / Do Not Modify

- `yolov5_lmi/` (git submodule), `anomalib_lmi/v0/`, `deprecated/` — do not modify unless explicitly instructed.
- `tf_objdet/` folder may remain; TensorFlow OD support has been dropped.

### Versioning

Automated semantic versioning via `.releaserc.json`. Default branch is `ais`.
