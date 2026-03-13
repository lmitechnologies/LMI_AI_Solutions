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


## Installation

There are two options to use this repository:

### Option 1: Install from Git (Recommended for users)
Use this option if you only want to use the tools without modifying the code.

```bash
pip install "git+https://github.com/lmitechnologies/LMI_AI_Solutions.git@ais"
```

### Option 2: Install from Source (Recommended for developers)
Use this option if you plan to modify the code or contribute to the repository.

1. Clone the repository:
```bash
git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git
cd LMI_AI_Solutions
```

2. Install package in editable mode:
```bash
pip install -e .
```

### Running Scripts

Run any scripts in this repo, for example:

```bash
python -m lmi_utils.label_utils.plot_with_json -h
```

## Development Guidelines

### Code Quality with Pre-commit

This repository uses pre-commit hooks with Ruff to ensure code quality and consistency. All contributions must pass these checks before being merged.

For detailed setup and usage instructions, see the [Pre-commit with Ruff Guide](docs/PRECOMMIT_GUIDE.md).

Quick setup:
```bash
pip install pre-commit
pre-commit install
# run pre-commit on all files (optional)
pre-commit run --all-files
```

The hooks will automatically run on `git commit` to check your code for style issues and formatting.

**If your commit fails due to pre-commit hooks:** Ruff automatically fixes most lint and formatting issues in place. Simply re-stage the modified files and commit again:
```bash
git add -u
git commit -m "Your commit message"
```
If any issues cannot be auto-fixed, Ruff will print the errors — fix them manually, then stage and commit.

### Contributing

Please read the [Contributing Guide](CONTRIBUTING.md) before submitting a pull request. It covers our branch workflow, commit conventions, and PR requirements.
