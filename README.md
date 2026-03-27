<p align="center"> 
    <picture>
        <source srcset="assets/images/FactorySmartAI_Logo_dark.png" media="(prefers-color-scheme: dark)">
        <source srcset="assets/images/FactorySmartAI_Logo_light.png" media="(prefers-color-scheme: light)">
        <img src="assets/images/FactorySmartAI_Logo_light.png" alt="FactorySmartAI Logo">
    </picture>
</p>

# <img src="assets/images/lmi.png" width="20" alt="LMI logo"/> LMI AI Solutions

This repo contains the utils scripts, and several submodules for LMI Technologies Inc. AI modeling development.

Currently, the following models are supported in the repo:
+ Object Detection
    - [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
    - [YOLOv5](https://github.com/ultralytics/yolov5)
    - [Detectron2](https://github.com/facebookresearch/detectron2)
    - [RF-DETR](https://github.com/roboflow/rf-detr)
- Anomaly Detection
    - [Anomalib](https://github.com/open-edge-platform/anomalib)
- Classification
    - [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)


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
```

2. Install package in editable mode:
```bash
pip install -e LMI_AI_Solutions
```

### Running Scripts

Run any scripts in this repo, for example:

```bash
python -m lmi_utils.label_utils.plot_with_json -h
```

## Contributing

Please read the [Contributing Guide](CONTRIBUTING.md) before submitting a pull request. It covers branch workflow, commit conventions, pre-commit / Ruff setup, and PR requirements.
