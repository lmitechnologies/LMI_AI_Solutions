<div style="text-align: center;"> 
    <picture>
        <source srcset="assets/images/FactorySmartAI_Logo_dark.png" media="(prefers-color-scheme: dark)">
        <source srcset="assets/images/FactorySmartAI_Logo_light.png" media="(prefers-color-scheme: light)">
        <img src="assets/images/FactorySmartAI_Logo_light.png" alt="FactorySmartAI Logo">
    </picture>
</div>

# <img src="assets/images/lmi.png" width="20"/> LMI AI Solutions

This repo contains the utils scripts, and several submodules for LMI Technologies Inc. AI modeling development.

Currently, the following models are supported in the repo:
+ object detection
    - [Ultralytics YOLO models](https://github.com/ultralytics/ultralytics)
    - [yolov5](https://github.com/lmitechnologies/yolov5)
    - [efficientnet](https://github.com/lmitechnologies/EfficientNet-PyTorch)
    - [detectron2](https://github.com/facebookresearch/detectron2)
    - [tensorflow object detection API](https://github.com/lmitechnologies/models)
- anomaly detection
    - [anomalib](https://github.com/lmitechnologies/anomalib)
- OCR
    - [paddleOCR](https://github.com/lmitechnologies/models)


## Use this repo

Installing latest from git:

```bash
pip install -e "git+https://github.com/lmitechnologies/LMI_AI_Solutions.git@ais#egg=lmi_utils&subdirectory=lmi_utils"
pip install -e "git+https://github.com/lmitechnologies/LMI_AI_Solutions.git@ais#egg=object_detectors&subdirectory=object_detectors"
pip install -e "git+https://github.com/lmitechnologies/LMI_AI_Solutions.git@ais#egg=anomaly_detectors&subdirectory=anomaly_detectors"
```

Installing from source:

```bash
cd LMI_AI_Solutions && pip install -e lmi_utils
cd LMI_AI_Solutions && pip install -e object_detectors
cd LMI_AI_Solutions && pip install -e anomaly_detectors
```

### Running Scripts

Run any scripts in this repo, for example:

```bash
python -m label_utils.plot_labels -h
```

## Development Guidelines

### Code Quality with Pre-commit

This repository uses pre-commit hooks with Ruff to ensure code quality and consistency. All contributions must pass these checks before being merged.

For detailed setup and usage instructions, see the [Pre-commit with Ruff Guide](docs/PRECOMMIT_GUIDE.md).

Quick setup:
```bash
pip install pre-commit
pre-commit install
```

The hooks will automatically run on `git commit` to check your code for style issues and formatting.
