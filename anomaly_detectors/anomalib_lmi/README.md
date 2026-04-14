# Anomalib Integration

This document demonstrates the usage of the latest version of [Anomalib v1.1.1](https://github.com/openvinotoolkit/anomalib/releases/tag/v1.1.1) for anomaly detection. If you are using older version, refer to [this link](https://github.com/lmitechnologies/LMI_AI_Solutions/blob/ais/anomaly_detectors/anomalib_lmi/README_old.md).

## Requirements

- Nvidia Driver installed
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Model training

- X86 system
- ubuntu >= 22.04
- python >= 3.10

### Convert to TensorRT on a GoMax

- Jetpack >= 6.0

## Usage

The current implementation requires the following workflow:

1. Organize data
2. Use tiling (optional)
3. Train model on a x86 system
4. convert the model to .pt and convert to tensorRT engine
5. Test tensorRT engine using the histogram method for anomaly threshold selection
6. Deploy tensorRT engine on a GoMax

## 1. Organize Data

Training Data Directory Structure

```bash
├── data
│   ├── train
│   ├── ├── good
│   ├ - test [optional]
│   ├ - ├ - good
│   ├ - ├ - defect_category_1
│   ├ - ├ - defect_category_2
│   ├ - ground_truth [optional and corresponding to test]
│   ├ - ├ - defect_category_1
│   ├ - ├ - defect_category_2
```

test and the ground_truth are optional. Follow the steps below to create these folders:

- Simply polygon label your test samples with [VGG](https://www.robots.ox.ac.uk/~vgg/software/via/via.html)
- Convert the labels to ground_truth format with [json_to_ground_truth.py](https://github.com/lmitechnologies/LMI_AI_Solutions/blob/ais/lmi_utils/label_utils/deprecate/json_to_ground_truth.py)
- Put the test images into `data/test`, corresponding ground_truth into `data/ground_truth`

## 2. Train Model on X86

Basic steps to train an Anomalib model:

1. Initialize/modify dockerfile
2. Initialize/modify docker-compose.yaml
3. Train model
4. convert the model to a pt file

### 2.1 Initialize/modify dockerfile for X86

x86.dockerfile:

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.04-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install libgl1 -y
WORKDIR /app

RUN pip install opencv-python -U --user
RUN pip install tabulate

# Installing from anomalib src
RUN git clone -b v1.1.1 https://github.com/openvinotoolkit/anomalib.git && cd anomalib && pip install -e .
RUN anomalib install --option core

RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git
```

### 2.2 Initialize/modify docker-compose.yaml

Install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The following sample yaml file trains a PaDiM model and outputs the model at `./training/2024-09-06`. The [patchcore.yaml](https://github.com/lmitechnologies/LMI_AI_Solutions/blob/ais/anomaly_detectors/anomalib_lmi/configs/patchcore.yaml) should exist in `./configs`.

```yaml
services:
  anomalib_train:
    build:
      context: .
      dockerfile: ./x86.dockerfile
    volumes:
      - ./data:/app/data/
      - ./configs/:/app/configs/
      - ./training/2024-09-06/:/app/out/
    ipc: host
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    command: >
      anomalib train --config /app/configs/patchcore.yaml
```

### 2.3 Train

Build the docker image:

```bash
docker compose build --no-cache
```

Run the container:

```bash
docker compose up 
```

### 2.4 convert the model to a pt file

The training outputs a lightning model. Use the following docker-compose file to convert to a pt file.

```yaml
services:
  anomalib_convert:
    build:
      context: .
      dockerfile: ./x86.dockerfile
    volumes:
      - ./training/2024-09-06/Patchcore/dataset/v0:/app/out/
    ipc: host
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    command: >
      anomalib export --model Patchcore --export_type torch --ckpt_path /app/out/weights/lightning/model.ckpt --default_root_dir /app/out
```

## 3. Generate TensorRT Engine

1. Initialize/modify docker file
   1. dockerfile for X86
   2. dockferfile for ARM
2. Initialize/modify docker-compose.yaml
3. Convert model

### 3.1.1 Initialize/modify docker file for X86

The same docker file as defined in [2.1 Initialize/modify dockerfile](#21-initializemodify-dockerfile-for-x86).

### 3.1.2 Initialize/modify docker file for ARM

arm.dockerfile:

```dockerfile
FROM nvcr.io/nvidia/l4t-tensorrt:r8.6.2-devel
ARG DEBIAN_FRONTEND=noninteractive
ENV PATH=$PATH:/usr/src/tensorrt/bin/

RUN apt-get update
RUN apt-get install libgl1 wget -y
WORKDIR /app

# dependencies for anomalib
RUN pip install pip setuptools packaging scikit-learn onnx tabulate -U
RUN pip install opencv-python -U --user

# Installing from anomalib src
RUN git clone -b v1.1.1 https://github.com/openvinotoolkit/anomalib.git && cd anomalib && pip install -e .
RUN anomalib install --option core

# Install torch dependencies
RUN wget https://developer.download.nvidia.com/compute/cusparselt/0.6.3/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
RUN dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
RUN cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.6.3/cusparselt-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install libcusparselt0 libcusparselt-dev libopenblas-dev

# install torch
RUN pip install https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+f70bd71a48.nv24.06.15634931-cp310-cp310-linux_aarch64.whl

# install torchvision
RUN apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
RUN git clone --branch 0.19.0 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
RUN cd torchvision && export BUILD_VERSION=0.19.0 && python3 setup.py install --user

RUN git clone -b FAIE-1673 https://github.com/lmitechnologies/LMI_AI_Solutions.git
```

### 3.2 Initialize/modify docker-compose file

```yaml
services:
  anomalib_trt:
    build:
      context: .
      dockerfile: ./arm.dockerfile
    volumes:
      - ./training/2024-09-06/Patchcore/dataset/v0/weights:/app/weights
    ipc: host
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    command: >
      bash -c "source /app/LMI_AI_Solutions/lmi_ai.env && python -m anomaly_detectors.anomalib_lmi.anomaly_model_v1 convert -i /app/weights/torch/model.pt -e /app/weights/engine"

```

### 3.3 Convert model

Build the docker image:

```bash
docker compose build --no-cache
```

Run the container:

```bash
docker compose up 
```

## 4. Validate Model

1. Initialize/modify docker-compose.yaml
2. Validate model
3. Choose Threshold

### 4.1 Initialize/modify docker-compose.yaml

```yaml
services:
  anomalib_infer:
    build:
      context: .
      dockerfile: ./arm.dockerfile
    volumes:
      - ./data:/app/data/
      - ./outputs:/app/outputs/
      - ./training/2024-09-06/Padim/dataset/v0/weights:/app/weights
    ipc: host
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    command: >
      bash -c "source /app/LMI_AI_Solutions/lmi_ai.env && python -m anomaly_detectors.anomalib_lmi.anomaly_model_v1 test -i /app/weights/engine/model.engine -d /app/data -o /app/outputs -p"

```

### 4.2 Validate model

Build the docker image:

```bash
docker compose build --no-cache
```

Run the container:

```bash
docker compose up 
```

### 4.3 Determine Optimum Threshold

![pdf](gamma_pdf_fit.png)

| Threshold | 2 | 7 | 11 | 16 | 21 | 25 | 30 | 35 | 39 | 44 |
|:-------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Prob of Patch Defect  |  99.862  |  45.0425 |  2.6595 |  0.047  |  0.0004 |  0      |  0     |  0     |  0     |  0 |
| Prob of Sample Defect | 100      | 100      | 70.8    | 25.8    |  7.6    |  3.8    |  1.4   |  0.4   |  0.2   |  0 |

In this example, setting a threshold at 11 would lead to a 2.6595% failure rate at the patch level, and a 70.8% failure rate at the part level.  Setting the threshold to 25 will lead to a 3.8% part/sample failure rate.

## 5 Tiling (optional)

It splits the original images into tiles. The current AIS implementation supports two modes for tiling validation and tensorRT conversion: batch mode and single mode. Batch mode makes predictions on all tiles of a image in one forward pass, whereas single mode makes predictions on a single tile.

Steps:

1. Generate tiles
2. Train a model using tiles
3. Validate/test
4. Convert to TensorRT

The [5.2 Validate/Test](#52-validatetest) and [5.3 Convert to TensorRT](#53-convert-to-tensorrt) are for batch mode.
If using single mode, please refer to the steps in [4. Validate Model](#4-validate-model) and [3. Generate TensorRT Engine](#3-generate-tensorrt-engine) for validation and tensorRT conversion.

### 5.1 Generate Tiles

Generate tiles as the training dataset, where `TILE_SZ` and `STRIDE` are integers for the tile size and stride step respectively. **Note:** if original images have varying sizes, make the image size consistent before generating tiles.

```bash
python -m lmi_utils.image_utils.img_tile --option tile -i PATH_DATA -o PATH_OUT --tile TILE_SZ TILE_SZ --stride STRIDE STRIDE
```

### 5.2 Validate/Test

Use original images (not the tiles), where `PATH_DATA` is the path to the original images.

```bash
python -m anomaly_detectors.anomalib_lmi.anomaly_model_v1 test -i PATH_MODEL -d PATH_DATA -o PATH_OUT -p --tile TILE_SZ TILE_SZ --stride STRIDE STRIDE
```

### 5.3 Convert to TensorRT

The script requires the original image size, tile size and stride for TensorRT conversion, where `H` and `W` are the height and width from original images (not the tiles).

```bash
python -m anomaly_detectors.anomalib_lmi.anomaly_model_v1 convert -i MODEL_PATH -o EXPORT_PATH --hw H W --tile TILE TILE --stride STRIDE STRIDE
```
