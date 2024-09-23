# Anomalib Integration
This document demonstrates the usage of the latest version of [Anomalib](https://github.com/openvinotoolkit/anomalib) for anomaly detection. If you are using older version, refer to [this link](https://github.com/lmitechnologies/LMI_AI_Solutions/blob/ais/anomaly_detectors/anomalib_lmi/README_old.md).


## Requirements
- Nvidia Driver installed
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### Model training
- X86 system
- ubuntu >= 22.04
- python >= 3.10

#### TensorRT on GoMax
TODO

## Usage

The current implementation requires the following workflow:

1. Organize data
2. Train model on x86 
3. convert the model to .pt and tensorRT engine
4. Test tensorRT engine using the histogram method for anomaly threshold selection
5. Deploy tensorRT engine on GoMax


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
- Simply polygon label your test samples with [VGG](https://www.robots.ox.ac.uk/~vgg/software/via/via.html), 
- Convert the labels to ground_truth format with [json_to_ground_truth.py](https://github.com/lmitechnologies/LMI_AI_Solutions/blob/ais/lmi_utils/label_utils/deprecate/json_to_ground_truth.py)
- Put the test images into `data/test`, corresponding ground_truth into `data/ground_truth`

## 2. Train Model on X86

Basic steps to train an Anomalib model:

1. Initialize/modify dockerfile
2. Initialize/modify docker-compose.yaml
3. Train model
4. convert the model to a pt file

### 2.1 Initialize/modify dockerfile for X86

```docker
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

# TODO: merge this branch to AIS
RUN git clone -b FAIE-1673 https://github.com/lmitechnologies/LMI_AI_Solutions.git
```

### 2.2 Initialize/modify docker-compose.yaml
Install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).   
The following sample yaml file trains a PaDiM model and outputs the model at `./training/2024-09-23`. The [padim.yaml](https://github.com/lmitechnologies/LMI_AI_Solutions/blob/ais/anomaly_detectors/anomalib_lmi/configs/padim.yaml) should exist in `./configs`. 

```yaml
services:
  anomalib_train:
    build:
      context: .
      dockerfile: ./dockerfile
    volumes:
      - ./data:/app/data/
      - ./configs/:/app/configs/
      - ./scripts/:/app/scripts/
      - ./training/2024-09-23/:/app/out/
    shm_size: '20gb'
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    command: >
      anomalib train --config /app/configs/padim.yaml

```
### 2.3 Train

1. Build the docker image: 
```bash
docker compose build --no-cache
```
2. Run the container:
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
      dockerfile: ./dockerfile
    volumes:
      - ./training/2024-09-23/Padim/dataset/v0:/app/out/
    shm_size: '20gb'
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    command: >
      anomalib export --model Padim --export_type torch --ckpt_path /app/out/weights/lightning/model.ckpt --default_root_dir /app/out

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
TODO

### 3.2 Initialize/modify docker-compose file
```yaml
services:
  anomalib_trt:
    build:
      context: .
      dockerfile: ./dockerfile
    volumes:
      - ./training/2024-09-23/Padim/dataset/v0/weights:/app/weights
    shm_size: '20gb'
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    command: >
      bash -c "source /app/LMI_AI_Solutions/lmi_ai.env && python -m anomalib_lmi.anomaly_model2 -a convert -i /app/weights/torch/model.pt -e /app/weights/engine"

```
### 3.3 Convert model
1. Build the docker image: 
```bash
docker compose build --no-cache
```
2. Run the container:
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
      dockerfile: ./dockerfile
    volumes:
      - ./data:/app/data/
      - ./outputs:/app/outputs/
      - ./training/2024-09-23/Padim/dataset/v0/weights:/app/weights
    shm_size: '20gb'
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    command: >
      bash -c "source /app/LMI_AI_Solutions/lmi_ai.env && python -m anomalib_lmi.anomaly_model2 -i /app/weights/engine/model.engine -d /app/data -o /app/outputs -p"

```
### 4.2 Validate model
1. Build the docker image: 
```bash
docker compose build --no-cache
```
2. Run the container:
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


