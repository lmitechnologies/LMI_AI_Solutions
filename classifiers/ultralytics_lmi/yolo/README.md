# Train and test YOLO classification models
This tutorial walks through training and testing YOLO classification models.

## System requirements
- Nvidia Drivers
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Model training
- x86 system
- Ubuntu OS

### TensorRT on GoMax
- JetPack >= 5.0

## Directory structure
The folder structure below will be created when we go through the tutorial. By convention, we use today's date (i.e. 2026-03-11) as the folder and file name.
```
├── config
│   ├── 2026-03-11_train.yaml
│   ├── 2026-03-11_val.yaml
│   ├── 2026-03-11_trt.yaml
├── preprocess
│   ├── 2026-03-11.sh
├── data
│   ├── train
│   ├── val (optional)
│   ├── test 
├── training
│   ├── 2026-03-11
├── validation
│   ├── 2026-03-11
├── prediction
│   ├── 2026-03-11
├── docker-compose_preprocess.yaml
├── docker-compose_train.yaml
├── docker-compose_val.yaml
├── docker-compose_predict.yaml
├── docker-compose_trt.x86.yaml
├── dockerfile
├── docker-compose_trt.arm.yaml   # arm system
├── arm.dockerfile                # arm system
```


## Create a dockerfile
Create `./dockerfile`:
```docker
FROM nvcr.io/nvidia/pytorch:25.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user opencv-python
RUN pip install ultralytics

# clone LMI AI Solutions repository
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && \
    pip install -e LMI_AI_Solutions

```

## Prepare the dataset
### Dataset Structure
```
├── data
│   ├── raw
│   │   ├── train
│   │   |   ├── class_1
│   │   |   ├── class_2
│   │   |   ├── ...
│   │   |   ├── class_N
│   │   ├── val (optional)
│   │   |   ├── class_1
│   │   |   ├── class_2
│   │   |   ├── ...
│   │   |   ├── class_N
│   │   ├── test
│   │   |   ├── class_1
│   │   |   ├── class_2
│   │   |   ├── ...
│   │   |   ├── class_N
```
Yolo classification models use the subfolder names as the class names. Replace `class_1`, `class_2`, `class_N` with real class names.

### Create a script for image preprocessing
Since **YOLO models require the dimensions of images to be divisible by 32**, in this tutorial, we prepare the dataset by the following:
- resize images to 224 in height while keeping the aspect ratio
- pad images to 224 in width


Create `./preprocess/2026-03-11.sh`:
```bash
# preprocess training dataset
python -m lmi_utils.image_utils.img_resize -i /app/data/train -o /temp --height 224 --recursive
python -m lmi_utils.image_utils.img_pad -i /temp -o /app/out/train --wh 224,224 --recursive

# preprocess validation dataset
python -m lmi_utils.image_utils.img_resize -i /app/data/val -o /temp --height 224 --recursive
python -m lmi_utils.image_utils.img_pad -i /temp -o /app/out/val --wh 224,224 --recursive
```

### Create a docker-compose file
Create `./docker-compose_preprocess.yaml`:
```yaml
services:
  yolo-cls:
    container_name: yolo-cls_prep
    build:
      context: .
      dockerfile: ./dockerfile
    ipc: host
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    volumes:
      # mount location_in_host:location_in_container
      - ./data/raw:/app/data
      - ./data/out:/app/out
      - ./preprocess/2026-03-11.sh:/app/preprocess/preprocess.sh
    command: >
      bash /app/preprocess/preprocess.sh
```

### Spin up the container
Run the following commands:
```bash
# build the container
docker compose -f docker-compose_preprocess.yaml build

# spin up the container
docker compose -f docker-compose_preprocess.yaml up
```
The preprocessed datasets will be generated in `./data/out`.


## Train the model

### Create a hyperparameter file
Create `./config/2026-03-11_train.yaml`. Below shows an example of training a **small-size yolo classification model** with the image size of 224x224:
```yaml
task: classify # (str) YOLO task, i.e. detect, segment, classify, pose
mode: train # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark

# Train settings -------------------------------------------------------------------------------------------------------
model: yolo26s-cls.pt # (str, optional) path to model file, i.e. yolo26n-cls.pt, yolo26s-cls.pt, yolo26m-cls.pt, yolo26l-cls.pt, yolo26x-cls.pt
epochs: 100 # (int) number of epochs to train for
patience: 100 # (int) epochs to wait for no observable improvement for early stopping of training
batch: 32 # (int) number of images per batch (-1 for AutoBatch)
imgsz: 224 # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
optimizer: auto # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
single_cls: False # (bool) train multi-class data as single-class
rect: False # (bool) rectangular training if mode='train' or rectangular validation if mode='val'
cos_lr: False # (bool) use cosine learning rate scheduler
close_mosaic: 10 # (int) disable mosaic augmentation for final epochs (0 to disable)
resume: False # (bool) resume training from last checkpoint
fraction: 1.0 # (float) dataset fraction to train on (default is 1.0, all images in train set)
freeze: None # (int | list, optional) freeze first n layers, or freeze list of layer indices during training

# Classification
dropout: 0.0 # (float) use dropout regularization (classify train only)

# data augmentation hyperparameters
hsv_h: 0.015 # (float) image HSV-Hue augmentation (fraction)
hsv_s: 0.7 # (float) image HSV-Saturation augmentation (fraction)
hsv_v: 0.4 # (float) image HSV-Value augmentation (fraction)
degrees: 0.0  # (float) image rotation (+/- deg)
translate: 0.1  # (float) image translation (+/- fraction)
scale: 0.5  # (float) image scale (+/- gain)
shear: 0.0  # (float) image shear (+/- deg)
perspective: 0.0  # (float) image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # (float) image flip up-down (probability)
fliplr: 0.5  # (float) image flip left-right (probability)
mosaic: 1.0  # (float) image mosaic (probability)
mixup: 0.0  # (float) image mixup (probability)
copy_paste: 0.0  # (float) segment copy-paste (probability)
auto_augment: randaugment # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
erasing: 0.4 # (float) probability of random erasing during classification training (0-1)

# more settings: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

### Create a docker-compose file
Create `./docker-compose_train.yaml`:
```yaml
services:
  yolo-cls:
    container_name: yolo-cls_train
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training:/app/training   # training output
      - ./data/out:/app/dataset  # training data, which should include a "train" subfolder and a "val"/"test" subfolder
      - ./config/2026-03-11_train.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 -m classifiers.ultralytics_lmi.run_cmd

```
> [!WARNING]
> Do **not** modify the target container paths (e.g., `/app/training`, `/app/dataset`, `/app/config/hyp.yaml`). The internal scripts expect these exact locations.


### Start training
Run the container (see [Spin up the container](#spin-up-the-container)) using `docker-compose_train.yaml`. The output will be generated in `./training/2026-03-11`.

### Monitor the training progress (optional)
While the training process is running, open another terminal and enter the commands:
```bash
# find the CONTAINER_ID
docker ps

# Log into the container which hosts the training process
docker exec -it CONTAINER_ID bash 

# track the training progress using tensorboard
tensorboard --logdir /app/training/2026-03-11 --port 6006
```

Monitor the training at http://localhost:6006.


## Validation
Create `./config/2026-03-11_val.yaml`:
```yaml
task: classify # (str) YOLO task, i.e. detect, segment, classify, pose
mode: val # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark
batch: 32 # (int) number of images per batch (-1 for AutoBatch)
imgsz: 224,224 # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
rect: False # (bool) rectangular training if mode='train' or rectangular validation if mode='val'
split: val # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Create `./docker-compose_val.yaml`:
```yaml
services:
  yolo-cls:
    container_name: yolo-cls_val
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./validation:/app/validation  # output path
      - ./training/2026-03-11/weights:/app/trained-inference-models   # trained model path, where it has best.pt
      - ./data/out:/app/dataset  # input data path
      - ./config/2026-03-11_val.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 -m classifiers.ultralytics_lmi.run_cmd

```

### Start validation
Run the container (see [Spin up the container](#spin-up-the-container)) using `docker-compose_val.yaml`. The output will be saved in `./validation/2026-03-11`.


## Prediction
Create `./config/2026-03-11_predict.yaml`. The `imgsz` should be a list of [h,w]:
```yaml
task: classify # (str) YOLO task, i.e. detect, segment, classify, pose
mode: predict # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark
batch: 16 # (int) number of images per batch (-1 for AutoBatch)
imgsz: 224,224 # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Create `./docker-compose_predict.yaml`:
```yaml
services:
  yolo-cls:
    container_name: yolo-cls_predict
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./prediction:/app/prediction  # output path
      - ./training/2026-03-11/weights:/app/trained-inference-models   # trained model path, where it has best.pt
      - ./data/out/test/distil:/app/data  # input data path
      - ./config/2026-03-11_test.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 -m classifiers.ultralytics_lmi.run_cmd
```

### Start prediction
Run the container (see [Spin up the container](#spin-up-the-container)) using `docker-compose_predict.yaml`. The output will be saved in `./prediction/2026-03-11`.


## Generate TensorRT engines
The TensorRT engines can be generated in two systems: x86 and ARM. Both systems share the same hyperparameter file, while the dockerfile and docker-compose files are different.

### Create a hyperparameter file
Create `./config/2026-03-11_trt.yaml` that works for both systems:
```yaml
task: classify  # (str) YOLO task, i.e. detect, segment, classify, pose, where classify, pose are NOT tested
mode: export  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark, where track, benchmark are NOT tested

# Export settings 
format: engine  # (str) format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats
half: True  # (bool) use half precision (FP16)
imgsz: 224,224 # (list) input images size as list[h,w] for predict and export modes
device: 0 # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu

# less likely used settings 
dynamic: False  # (bool) ONNX/TF/TensorRT: dynamic axes
simplify: False  # (bool) ONNX: simplify model
opset:  # (int, optional) ONNX: opset version
workspace: 4  # (int) TensorRT: workspace size (GB)

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Create `./docker-compose_trt.yaml`:
```yaml
services:
  yolo-cls:
    container_name: yolo-cls_trt
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./training/2026-03-11/weights:/app/trained-inference-models   # trained model path, which includes a best.pt
      - ./config/2026-03-11_trt.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 -m classifiers.ultralytics_lmi.run_cmd
```

### Engine Generation on x86 systems

Run the container (see [Spin up the container](#spin-up-the-container)) using `docker-compose_trt.yaml`. The output engines will be saved in `./training/2026-03-11/weights`.


### Engine Generation on ARM systems
Create `./arm.dockerfile`:
```docker
# jetpack 5.1
FROM --platform=linux/arm64/v8 nvcr.io/nvidia/l4t-ml:r35.2.1-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN python3 -m pip install pip --upgrade
RUN pip3 install --upgrade setuptools wheel
RUN pip3 install opencv-python --user
RUN pip3 install ultralytics

# clone AIS
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && \
  pip install -e LMI_AI_Solutions
```

Replace the line `dockerfile: dockerfile` in `./docker-compose_trt.yaml` with `dockerfile: arm.dockerfile`.

Run the container (see [Spin up the container](#spin-up-the-container)) using `docker-compose_trt.yaml`. The output engines will be saved in `./training/2026-03-11/weights`.
