# Anomalib Integration
This folder contains the [Anomalib](https://github.com/openvinotoolkit/anomalib) integration script for anomaly detection, supported models:
- [patchcore](https://arxiv.org/abs/2106.08265)
- [padim](https://arxiv.org/abs/2011.08785)

# Usage

This repo is used to train, test, and run anomalib anomaly detector models.

The current implementation requires the following workflow:

1. Organize data
2. Train model and output onnx file for tensorRT conversion
3. Convert onnx model to tensorRT engine
4. Test tensorRT engine using the histogram method for anomaly threshold selection
5. Deploy tensorRT engine

Limitations includes:
1. The current model does not support native torch testing and deployment.  Therefor, there is no easy way to perform the histogram testing for threshold selection.

## 1. Organize Data

#### Training Data Directory Structure
```bash
├── root_dir
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
Although test and the ground_truth are optional, it enables the training to generate insightful metrics
> * Simply polygon label your test samples with [VGG](https://www.robots.ox.ac.uk/~vgg/software/via/via.html), 
> * Convert the labels to ground_truth format with lmi_utils/label_utils/json_to_ground_truth.py
> * Put the test images into root_dir/test, corresponding ground_truth into root_dir/ground_truth as #1
> * At the end of the training, it will generate metrics like this:
```bash
        image_AUROC          0.8500000238418579
       image_F1Score         0.9268293380737305
        pixel_AUROC          0.9906309843063354
       pixel_F1Score         0.4874393045902252
```
## 2. Train Model

Basic steps to train an Anomalib model:

1. Initialize/modify dockerfile
2. Initialize/modify docker-compose.yaml
3. Train model
4. Validate Model

### 1. Initialize/modify dockerfile

```Dockerfile
FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update
RUN apt-get install python3 python3-pip -y
RUN apt-get install git libgl1 -y
WORKDIR /app

RUN pip install pycuda
RUN pip install opencv-python==4.6.0.66 --user 

RUN pip install openvino-dev==2023.0.0 openvino-telemetry==2022.3.0 nncf==2.4.0
RUN pip install nvidia-pyindex onnx-graphsurgeon

# Installing from anomalib src requires latest pip 
RUN python3 -m pip install --upgrade pip
RUN git clone -b ais https://github.com/lmitechnologies/LMI_AI_Solutions.git && cd LMI_AI_Solutions/anomaly_detectors && git submodule update --init submodules/anomalib
RUN cd LMI_AI_Solutions/anomaly_detectors/submodules/anomalib && pip install -e .
```

### 2. Initialize/modify docker-compose.yaml

The following sample yaml file references training data at ./training/2023-08-16 and trains a PaDiM model.

```yaml
version: "3.9"
services:
  anomalib_train:
    build:
      context: .
      dockerfile: ./dockerfile.x86_64
    volumes:
      - ../data/train/:/app/data/train/
      - ./configs/:/app/configs/
      - ./training/2023-08-16/:/app/out/
    environment:
      - model=padim
    shm_size: '20gb' 
    deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
    command: >
      python3 /app/LMI_AI_Solutions/anomaly_detectors/submodules/anomalib/tools/train.py
      --model padim
      --config /app/configs/padim.yaml
    # stdin_open: true # docker run -i
    # tty: true        # docker run -t
```
### 3. Train

1. Build the docker image: 
```bash
docker compose build --no-cache
```
2. Run the container:
```bash
docker compose up 
```
## 3. Generate TensorRT Engine

1. Initialize/modify docker-compose.yaml
2. Convert model

### 1. Initialize/modify docker-compose.yaml

```yaml
version: "3.9"
services:
  anomalib_convert:
    build:
      context: .
      dockerfile: ./dockerfile.x86_64
    volumes:
      - ./training/2023-08-29/results/padim/model/run/weights/onnx/:/app/onnx/
      - ./training/2023-08-29/results/padim/model/run/weights/onnx/engine/:/app/onnx/engine/
    environment:
      - error_threshold=0
    shm_size: '20gb' 
    deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
    command: >
      python3 /app/LMI_AI_Solutions/anomaly_detectors/anomalib_lmi/anomaly_model.py 
      --action convert
```
### 2. Convert model
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

### 1. Initialize/modify docker-compose.yaml

```yaml
version: "3.9"
services:
  anomalib_test:
    build:
      context: .
      dockerfile: ./dockerfile.x86_64
    volumes:
      - ./model.engine:/app/onnx/engine/model.engine
      - ../data/train/good:/app/data/
      - ../annotation_results/:/app/annotation_results/
    environment:
      - error_threshold=0
    shm_size: '20gb' 
    deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
    command: >
      bash -c "
      source /app/LMI_AI_Solutions/lmi_ai.env &&
      python3 /app/LMI_AI_Solutions/anomaly_detectors/anomalib_lmi/anomaly_model.py 
      --action test
      "
```
### 2. Validate model
1. Build the docker image: 
```bash
docker compose build --no-cache
```
2. Run the container:
```bash
docker compose up 
```

### 3. Determine Optimum Threshold
![pdf](gamma_pdf_fit.png)
![cdf](gamma_cdf_fit.png)

+-----------------------+----------+----------+---------+---------+---------+---------+--------+--------+--------+----+
| Threshold             |   2.0059 |   6.6719 | 11.3379 | 16.0039 | 20.6699 | 25.3359 | 30.002 | 34.668 | 39.334 | 44 |
+-----------------------+----------+----------+---------+---------+---------+---------+--------+--------+--------+----+
| Prob of Patch Defect  |  99.862  |  45.0425 |  2.6595 |  0.047  |  0.0004 |  0      |  0     |  0     |  0     |  0 |
+-----------------------+----------+----------+---------+---------+---------+---------+--------+--------+--------+----+
| Prob of Sample Defect | 100      | 100      | 70.8    | 25.8    |  7.6    |  3.8    |  1.4   |  0.4   |  0.2   |  0 |
+-----------------------+----------+----------+---------+---------+---------+---------+--------+--------+--------+----+