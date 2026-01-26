## Anomalib v2 Models

### Configuration

Setup the config.yaml for training a model in the following example we use padim. Please refer to anomalib v2.2.0 API documentation for aditional parameters. 

```yaml
model:
  class_name: "Padim"
  params:
    backbone: "resnet18"
    layers: ["layer1", "layer2", "layer3"]
    pre_trained: true
    image_size: [224, 224]

data:
  name: "dataset"
  root: "/app/data/toothbrush" # path to the root dir for the dataset
  normal_dir: "train" # the folder to use for training
  extensions: [".png"]
  train_batch_size: 32
  eval_batch_size: 32
  num_workers: 8
  test_split_mode: "synthetic"
  test_split_ratio: 0.2
  val_split_mode: "same_as_test"
  val_split_ratio: 0.5
  train_augmentations: # feel free to remove or modify augmentations (null for no traning augmentations)
    - class_name: "RandomHorizontalFlip"
      params:
        p: 0.5
  
    - class_name: "RandomVerticalFlip"
      params:
        p: 0.2
      
    - class_name: "RandomResizedCrop"
      params:
        size: [224, 224]
        scale: [0.8, 1.0]

    - class_name: "ColorJitter"
      params:
        brightness: 0.4
        contrast: 0.4
        saturation: 0.4
        hue: 0.2

    - class_name: "RandomGrayscale"
      params:
        p: 0.1

engine:
  max_epochs: 1
  accelerator: "auto" # gpu or cpu auto uses gpu if available
  devices: 1
  default_root_dir: "/app/out/"
```
### Training
The following is the dockerfile that is requried for training and inference.

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.04-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install libgl1 -y
WORKDIR /app

RUN pip install opencv-python -U --user
RUN pip install tabulate

# Installing from anomalib src
RUN git clone https://github.com/openvinotoolkit/anomalib.git && cd anomalib && pip install -e .
RUN anomalib install

RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git
RUN cd LMI_AI_Solutions && git submodule update --init object_detectors/submodules/yolov5
RUN cd LMI_AI_Solutions && pip3 install -e object_detectors && pip3 install -e anomaly_detectors && pip3 install -e lmi_utils && pip3 install -e classifiers
```

The docker-compose.yaml for training is the following:

```yaml
services:
  anomalib_train:
    build:
      context: .
      dockerfile: ./dockerfile
    volumes:
      - ../data/:/app/data/
      - ./configs/:/app/configs/
      - ./training/mvtech:/app/out/
    ipc: host
    runtime: nvidia 
    command: >
      python3 -m anomalib_lmi.anomalib_v2.train --config /app/configs/padim.yaml 
```
### Inference

The docker compose file for inference:

```yaml
services:
  anomalib_predict:
    build:
      context: .
      dockerfile: ./dockerfile
    volumes:
      - ../data/:/app/data/
      - ./configs/:/app/configs/
      - ./predictions/:/app/output/predictions/
      - ./training/models/:/app/weights/
    ipc: host
    runtime: nvidia 
    command: >
      python3 -m anomalib_lmi.anomalib_model_v2 test -i /app/weights/Patchcore/dataset/v0/weights/torch/model.pt -d /app/data/toothbrush/ -o /app/output/predictions/ -p
```

### Convert to TensorRT

```dockerfile
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r35.2.1-py3
FROM ${BASE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive

ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

WORKDIR /app
# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Download and install PyTorch wheels for Jetpack 5
RUN wget -q https://github.com/lmitechnologies/lmi-ais-assets/releases/download/jp5/torch-2.2.0-cp38-cp38-linux_aarch64.whl && \
    wget -q https://github.com/lmitechnologies/lmi-ais-assets/releases/download/jp5/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl && \
    pip3 install --no-cache-dir torch-2.2.0-cp38-cp38-linux_aarch64.whl torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl && \
    rm -f *.whl

# Install core AI/ML packages (install PyYAML first with --ignore-installed to avoid conflicts)
RUN pip3 install --no-cache-dir --ignore-installed "PyYAML>=5.3.1" && \
    pip3 install --no-cache-dir ultralytics scikit-guess scipy pycocotools

RUN git clone -b ais https://github.com/lmitechnologies/LMI_AI_Solutions.git && cd LMI_AI_Solutions && \
    pip3 install --no-cache-dir -e lmi_utils -e object_detectors -e anomaly_detectors -e classifiers
RUN pip3 install --user opencv-python==4.6.0.66
RUN pip3 install onnx onnx_graphsurgeon onnxruntime
ENV PATH="/usr/src/tensorrt/bin:${PATH}"
RUN pip3 install --no-cache-dir numpy==1.23.5 tabulate
```

docker-compose.yaml to convert onnx to TensorRT engine
```yaml
services:
  anomalib_convert_to_trt:
    build:
      context: .
      dockerfile: ./arm.dockerfile
    volumes:
      - ../data/:/app/data/
      - ./configs/:/app/configs/
      - ./training/2026-01-15/:/app/weights/
    ipc: host
    runtime: nvidia 
    command: >
        trtexec --onnx=/app/weights/Patchcore/dataset/v0/weights/onnx/model.onnx --saveEngine=/app/weights/Patchcore/dataset/v0/weights/model.engine --memPoolSize=workspace:4096
```

