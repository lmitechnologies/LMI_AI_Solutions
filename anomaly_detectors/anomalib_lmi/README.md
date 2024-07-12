# Anomalib Integration
This folder contains the [Anomalib](https://github.com/openvinotoolkit/anomalib) integration script for anomaly detection, supported models:
- [patchcore](https://arxiv.org/abs/2106.08265)
- [padim](https://arxiv.org/abs/2011.08785)

## Requirements
- Nvidia Drivers
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### Model training
- X86 system
- ubuntu OS

#### TensorRT on GoMax
- JetPack 5.0 or 5.1

## Usage

This repo is used to train, test, and run anomalib anomaly detector models.

The current implementation requires the following workflow:

1. Organize data
2. Train model and output onnx file for tensorRT conversion
3. Convert onnx model to tensorRT engine
4. Test tensorRT engine using the histogram method for anomaly threshold selection
5. Deploy tensorRT engine


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
Although test and the ground_truth are optional, it enables the training to generate insightful metrics
> * Simply polygon label your test samples with [VGG](https://www.robots.ox.ac.uk/~vgg/software/via/via.html), 
> * Convert the labels to ground_truth format with [json_to_ground_truth.py](https://github.com/lmitechnologies/LMI_AI_Solutions/blob/ais/lmi_utils/label_utils/json_to_ground_truth.py)
> * Put the test images into `data/test`, corresponding ground_truth into `data/ground_truth`
> * At the end of the training, it will generate metrics like this:
```bash
        image_AUROC          0.8500000238418579
       image_F1Score         0.9268293380737305
        pixel_AUROC          0.9906309843063354
       pixel_F1Score         0.4874393045902252
```
## 2. Train Model on X86

Basic steps to train an Anomalib model:

1. Initialize/modify dockerfile
2. Initialize/modify docker-compose.yaml
3. Train model
4. Validate Model

### 2.1 Initialize/modify dockerfile for X86

```Dockerfile
FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN apt-get update
RUN apt-get install python3 python3-pip -y
RUN apt-get install git libgl1 -y
WORKDIR /app

RUN pip install pycuda
RUN pip install opencv-python -U --user 

RUN pip install openvino-dev==2023.0.0 openvino-telemetry==2022.3.0 nncf==2.4.0
RUN pip install nvidia-pyindex onnx-graphsurgeon
RUN pip install tabulate
RUN pip install albumentations

# Installing from anomalib src requires latest pip 
RUN python3 -m pip install --upgrade pip
RUN git clone -b ais https://github.com/lmitechnologies/LMI_AI_Solutions.git && cd LMI_AI_Solutions/anomaly_detectors && git submodule update --init submodules/anomalib
RUN cd LMI_AI_Solutions/anomaly_detectors/submodules/anomalib && pip install -e .
```

### 2.2 Initialize/modify docker-compose.yaml
Install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).   
The following sample yaml file references training data at `./training/2024-02-28` and trains a PaDiM model. The [padim.yaml](https://github.com/lmitechnologies/LMI_AI_Solutions/blob/ais/anomaly_detectors/anomalib_lmi/configs/padim.yaml) should exist in `./configs`. 

```yaml
version: "3.9"
services:
  anomalib_train:
    build:
      context: .
      dockerfile: ./dockerfile
    volumes:
      # mount location_in_host:location_in_container
      - ./data/train/:/app/data/train/
      - ./configs/:/app/configs/
      - ./training/2024-02-28/:/app/out/
    shm_size: '20gb' 
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    command: >
      python3 /app/LMI_AI_Solutions/anomaly_detectors/submodules/anomalib/tools/train.py
      --config /app/configs/padim.yaml
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


## 3. Generate TensorRT Engine

1. Initialize/modify docker file
   1. dockerfile for X86
   2. dockferfile for ARM
2. Initialize/modify docker-compose.yaml
3. Convert model

### 3.1.1 Initialize/modify docker file for X86
The same docker file as defined in [2.1 Initialize/modify dockerfile](#21-initializemodify-dockerfile-for-x86).

### 3.1.2 Initialize/modify docker file for ARM
```dockerfile
# this works for JetPack 5.0 and 5.1
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

RUN apt-get update && apt-get install git -y
RUN pip install --upgrade pip
RUN pip3 install --ignore-installed PyYAML>=5.3.1
RUN pip3 install opencv-python --user

RUN git clone -b ais https://github.com/lmitechnologies/LMI_AI_Solutions.git && cd LMI_AI_Solutions/anomaly_detectors && git submodule update --init submodules/anomalib
RUN cd LMI_AI_Solutions/anomaly_detectors/submodules/anomalib && pip install -e .

# trtexec
ENV PATH=$PATH:/usr/src/tensorrt/bin/

RUN pip3 install nvidia-pyindex
RUN pip3 install onnx-graphsurgeon
RUN pip3 install pycuda

# downgrade from 1.24.x to 1.23.1 to fix "np.bool` was a deprecated alias for the builtin `bool`"
RUN pip3 install numpy==1.23.1
```

### 3.2 Initialize/modify docker-compose file
```yaml
version: "3.9"
services:
  anomalib_convert:
    build:
      context: .
      dockerfile: ./dockerfile
    volumes:
      - ./training/2024-02-28/results/padim/model/run/weights/onnx/:/app/onnx/
      - ./training/2024-02-28/results/padim/model/run/weights/engine/:/app/engine/
    shm_size: '20gb' 
    runtime: nvidia
    command: >
      bash -c "source /app/LMI_AI_Solutions/lmi_ai.env && 
      python3 -m anomalib_lmi.anomaly_model
      --action convert -i /app/onnx/model.onnx -e /app/engine"
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
version: "3.9"
services:
  anomalib_test:
    build:
      context: .
      dockerfile: ./dockerfile
    volumes:
      - ./training/2024-02-28/results/padim/model/run/weights/engine/model.engine:/app/model/model.engine
      - ./data/train/good:/app/data/
      - ./annotation_results/:/app/annotation_results/
    shm_size: '20gb' 
    runtime: nvidia
    command: >
      bash -c "source /app/LMI_AI_Solutions/lmi_ai.env && 
      python3 -m anomalib_lmi.anomaly_model
      --action test -i /app/model/model.engine --plot --generate_stats"
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
![cdf](gamma_cdf_fit.png)


| Threshold | 2 | 7 | 11 | 16 | 21 | 25 | 30 | 35 | 39 | 44 |
|:-------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Prob of Patch Defect  |  99.862  |  45.0425 |  2.6595 |  0.047  |  0.0004 |  0      |  0     |  0     |  0     |  0 |
| Prob of Sample Defect | 100      | 100      | 70.8    | 25.8    |  7.6    |  3.8    |  1.4   |  0.4   |  0.2   |  0 |

In this example, setting a threshold at 11 would lead to a 2.6595% failure rate at the patch level, and a 70.8% failure rate at the part level.  Setting the threshold to 25 will lead to a 3.8% part/sample failure rate. 


