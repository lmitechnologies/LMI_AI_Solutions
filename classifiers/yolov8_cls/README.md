# Train and test YOLOv8 models
This is the tutorial walking through how to train and test YOLOv8 classification models.

## System requirements
- [Docker Engine](https://docs.docker.com/engine/install)

### Model training
- x86 system
- CUDA >= 12.1
- ubuntu

### TensorRT on GoMax
- JetPack >= 5.0.2

## Directory structure
The folder structure below will be created when we go through the tutorial. By convention, we use today's date (i.e. 2023-07-19) as the file name.
```
├── config
│   ├── 2023-07-19_train.yaml
│   ├── 2023-07-19_val.yaml
│   ├── 2023-07-19_trt.yaml
├── preprocess
│   ├── 2023-07-19.sh
├── data
│   ├── train
│   ├── val
│   ├── test (optional)
├── training
│   ├── 2023-07-19
├── validation
│   ├── 2023-07-19
├── prediction
│   ├── 2023-07-19
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
Create a file `./dockerfile`. It installs the dependencies and clone LMI_AI_Solutions repository inside the container.
```docker
# last version running on ubuntu 20.04, require CUDA 12.1 
FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user opencv-python
RUN pip install ultralytics -U

# clone LMI AI Solutions repository
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git

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
│   │   ├── val
│   │   |   ├── class_1
│   │   |   ├── class_2
│   │   |   ├── ...
│   │   |   ├── class_N
│   │   ├── test (optional)
│   │   |   ├── class_1
│   │   |   ├── class_2
│   │   |   ├── ...
│   │   |   ├── class_N
```
Yolo classification models use the subfolder names as the class names. Replace `class_1`, `class_2`, `class_N` with real class names.

### Create a script for image processing
Since **YOLO models require the dimensions of images to be dividable by 32**, in this tutorial, we prepare the dataset by the followings:
- resize images to 224 in height while keep the aspect ratio
- pad images to 224 in width


Create a script `./preprocess/2023-07-19.sh` as follows:
```bash
# import the repo paths
source /repos/LMI_AI_Solutions/lmi_ai.env

# preprocess training dataset
python -m image_utils.img_resize -i /app/data/train -o /temp --height 224 --recursive
python -m image_utils.img_pad -i /temp -o /app/out/train --wh 224,224 --recursive

# preprocess validation dataset
python -m image_utils.img_resize -i /app/data/val -o /temp --height 224 --recursive
python -m image_utils.img_pad -i /temp -o /app/out/val --wh 224,224 --recursive
```

### Create a docker-compose file
To run the script in the container, we need to create a file `./docker-compose_preprocess.yaml`. We mount the location in host to a location in container so that the files/folders changes in container are reflected in host. Below, we mount `./data/raw` in the host to `/app/data` in the container. Also, mount the bash script to `/app/preprocess/preprocess.sh`. 
```yaml
version: "3.9"
services:
  yolov8_cls:
    container_name: yolov8-cls_prep
    build:
      context: .
      dockerfile: ./dockerfile
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data/raw:/app/data
      - ./data/out:/app/out
      - ./preprocess/2023-07-19.sh:/app/preprocess/preprocess.sh
    command: >
      bash /app/preprocess/preprocess.sh
```

### Spin up the container
Spin up the container using the following commands: 
```bash
# build the container
# "-f" specifies the yaml file to load
docker compose -f docker-compose_preprocess.yaml build

# spin up the container
docker compose -f docker-compose_preprocess.yaml up
```
Once it finishs, the train and val datasets will be created in `./data/out`.


## Train the model
To train the model, we need to create a hyperparameter file and a docker-compose file.

### Create a hyperparameter file
Crete a file `./config/2023-07-19_train.yaml`. Below shows an example of training a **small-size yolov8 classification model** with the image size of 224x224. If the training images are square, set `rect` to `False`.
```yaml
task: classify # (str) YOLO task, i.e. detect, segment, classify, pose
mode: train # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark

# Train settings -------------------------------------------------------------------------------------------------------
model: yolov8s-cls.pt # (str, optional) path to model file, i.e. yolov8n.pt, yolov8n.yaml
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
Create a file `./docker-compose_train.yaml`. It mounts the host locations to the required directories in the container and run the script `run_cmd.py`, which load the hyperparameters and run the task that was specified in the file `./config/2023-07-19_train.yaml`.
```yaml
version: "3.9"
services:
  yolov8-cls:
    container_name: yolov8-cls_train
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training:/app/training   # training output
      - ./data/out:/app/dataset  # training data, which should include a "train" subfolder and a "val"/"test" subbfolder
      - ./config/2023-07-19_train.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/classifiers/yolov8_cls/run_cmd.py

```
Note: Do **NOT** modify the required locations in the container, such as `/app/training`, `/app/data`, `/app/config/dataset.yaml`, `/app/config/hyp.yaml`.


### Start training
Spin up the docker containers to train the model as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_train.yaml`.** Once the training is done, a folder named by today's date will be generated in `training` folder, i.e. `training/2023-07-19`.

### Monitor the training progress (optional)
While the training process is running, open another terminal. 
```bash
# find the CONTAINER_ID
docker ps

# Log in the container which hosts the training process
docker exec -it CONTAINER_ID bash 

# track the training progress using tensorboard
tensorboard --logdir /app/training/2023-07-19 --port 6006
```

Execuate the command above and go to http://localhost:6006 to monitor the training.


## Validation
Create a hyperparameter file `./config/2023-07-19_val.yaml`.
```yaml
task: classify # (str) YOLO task, i.e. detect, segment, classify, pose
mode: val # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark
batch: 32 # (int) number of images per batch (-1 for AutoBatch)
imgsz: 224,224 # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
rect: False # (bool) rectangular training if mode='train' or rectangular validation if mode='val'
split: val # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Create a file `./docker-compose_val.yaml` as below.
```yaml
version: "3.9"
services:
  yolov8-cls:
    container_name: yolov8-cls_val
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./validation:/app/validation  # output path
      - ./training/2023-07-19/weights:/app/trained-inference-models   # trained model path, where it has best.pt
      - ./data/out:/app/dataset  # input data path
      - ./config/2023-07-19_val.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/classifiers/yolov8_cls/run_cmd.py

```

### Start validation
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_val.yaml.`** Then, the output results are saved in `./validation/2023-07-19`.


## Prediction
Create a hyperparameter file `./config/2023-07-19_predict.yaml`. The `imgsz` should be a list of [h,w].
```yaml
task: classify # (str) YOLO task, i.e. detect, segment, classify, pose
mode: predict # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark
batch: 16 # (int) number of images per batch (-1 for AutoBatch)
imgsz: 224,224 # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Create a file `./docker-compose_predict.yaml` as below.
```yaml
version: "3.9"
services:
  yolov8-cls:
    container_name: yolov8-cls_predict
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./prediction:/app/prediction  # output path
      - ./training/2023-07-19/weights:/app/trained-inference-models   # trained model path, where it has best.pt
      - ./data/out/test/distil:/app/data  # input data path
      - ./config/2023-07-19_test.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/classifiers/yolov8_cls/run_cmd.py
```

### Start prediction
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_predict.yaml.`** Then, the output results are saved in `./prediction/2023-07-19`.


## Generate TensorRT engines
The TensorRT egnines can be generated in two systems: x86 and ARM. Both systems share the same hyperparameter file, while the dockerfile and docker-compose file are different.

### Create a hyperparameter file
Create a hyperparamter yaml file `./config/2023-07-19_trt.yaml` that works for both systems:
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

### Engine Generation on x86 systems

Create a docker-compose file `./docker-compose_trt.x86.yaml`:
```yaml
version: "3.9"
services:
  yolov8-cls:
    container_name: yolov8-cls_trt
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./training/2023-07-19/weights:/app/trained-inference-models   # trained model path, which includes a best.pt
      - ./config/2023-07-19_trt.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/classifiers/yolov8_cls/run_cmd.py
```

#### Start generation
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_trt.x86.yaml`.** Then, the tensorRT engine is generated in `./training/2023-07-19/weights`.


### Engine Generation on ARM systems
Create a file `./arm.dockerfile`.
```docker
# jetpack 5.1
FROM --platform=linux/arm64/v8 nvcr.io/nvidia/l4t-ml:r35.2.1-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN python3 -m pip install pip --upgrade
RUN pip3 install --upgrade setuptools wheel
RUN pip3 install opencv-python --user
RUN pip3 install ultralytics -U

# clone AIS
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git
```

Create a file `./docker-compose_trt.arm.yaml`,
```yaml
version: "3.9"
services:
  yolov8-cls:
    container_name: yolov8-cls_trt
    build:
      context: .
      dockerfile: arm.dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./training/2023-07-19/weights:/app/trained-inference-models   # contains a best.pt
      - ./config/2023-07-19_trt.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/classifiers/yolov8_cls/run_cmd.py
```

#### Start generation
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). Ensure to load the `./docker-compose_trt.arm.yaml`. The output engines are saved in `./training/2023-07-19/weights`.