# 📖 YOLO Models Tutorial
This tutorial walks through how to train and test Ultralytics YOLO models. 

## ⛭ System requirements

### General
- [Nvidia Drivers](https://www.nvidia.com/en-us/drivers/)
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Model training
- x86
- ubuntu OS or Windows
- labeling tool
  - [Label Studio](https://labelstud.io/)

### TensorRT on GoMax
- JetPack >= 5.0


## 📁 Directory Structure
The folder structure below will be created when we go through the tutorial. By convention, we use today's date (e.g., `2026-03-11`) for file and folder names.
```
├── configs
│   ├── 2026-03-11_train.yaml
│   ├── 2026-03-11_predict.yaml
│   ├── 2026-03-11_trt.yaml
├── preprocess
│   ├── 2026-03-11.sh
├── data
│   ├── allImages
│   │   ├── *.png
│   │   ├── *.json
├── training
│   ├── 2026-03-11
├── prediction
│   ├── 2026-03-11
├── docker-compose_preprocess.yaml
├── docker-compose_train.yaml
├── docker-compose_predict.yaml
├── docker-compose_trt.yaml
├── dockerfile
├── arm.dockerfile                # arm system
```


## Create a dockerfile
Create `./dockerfile`:
```docker
FROM nvcr.io/nvidia/pytorch:25.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --user opencv-python
RUN pip install ultralytics

# clone LMI AI Solutions repository
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && \
  pip install -e LMI_AI_Solutions

```

## Prepare the dataset
Prepare the dataset by doing the following:
- convert json to csv
- resize images and labels in csv
- convert labeling data to YOLO format

**YOLO models require the dimensions of images to be dividable by 32**. In this tutorial, we resize images to 640x640.

### Create a script for data processing
First, create a script `./preprocess/2026-03-11.sh`, which converts labels from Label Studio JSON to CSV, resizes images, and converts data to YOLO format.
```bash
# modify to your data path
input_path=/app/data/allImages
# modify the width and height according to your data
W=640
H=640

# convert labels from VGG json to csv
# python -m lmi_utils.label_utils.via_json_to_csv -d $input_path --output_fname labels.csv

# convert labels from label studio to csv
python -m lmi_utils.label_utils.lst_to_csv -i $input_path -o $input_path

# resize images with labels
python -m lmi_utils.label_utils.resize_with_csv -i $input_path -o /app/data/resized --width $W --height $H

# convert to YOLO format
# remove the --seg flag if you want to train an object detection model
python -m lmi_utils.label_utils.csv_to_yolo -i /app/data/resized -o /app/data/resized_yolo --seg
```

### Create a docker-compose file
Create `./docker-compose_preprocess.yaml`:
```yaml
services:
  yolo_preprocess:
    container_name: yolo_preprocess
    build:
      context: .
      dockerfile: ./dockerfile
    ipc: host
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    volumes:
      - ./data:/app/data
      - ./preprocess/2026-03-11.sh:/app/preprocess/preprocess.sh
    command: >
      bash /app/preprocess/preprocess.sh
```

### Spin up the container
```bash
# build the container
docker compose -f docker-compose_preprocess.yaml build

# spin up the container
docker compose -f docker-compose_preprocess.yaml up
```
Once it finishes, the YOLO format dataset will be created at `./data/resized_yolo/dataset.yaml`.



## Train the model

### Create a hyperparameter file
Create `./configs/2026-03-11_train.yaml`. (To train object detection models, set `task` to `detect`): 

```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose
mode: train  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark

# training settings
model: yolo26m-seg.pt # (str) supported models: yolo26n-seg.pt, yolo26s-seg.pt, yolo26m-seg.pt, yolo26l-seg.pt, yolo26x-seg.pt
epochs: 300  # (int) number of epochs
time: # (float, optional) max hours to train; overrides epochs if set
patience: 100 # (int) early stop after N epochs without val improvement
batch: 16  # (int) number of images per batch (-1 for AutoBatch)
imgsz: 640  # (int) input images size, use the larger dimension if rectangular image
exist_ok: False  # (bool) whether to overwrite existing training folder
rect: False  # (bool) use rectangular images for training if mode='train' or rectangular validation if mode='val'
resume: False  # (bool) resume training from last checkpoint

# Segmentation
overlap_mask: True # (bool) merge instance masks into one mask during training (segment only)
mask_ratio: 4 # (int) mask downsample ratio (segment only)

# data augmentation hyperparameters
degrees: 0.0  # (float) image rotation (+/- deg)
translate: 0.1  # (float) image translation (+/- fraction)
scale: 0.5  # (float) image scale (+/- gain)
shear: 0.0  # (float) image shear (+/- deg)
perspective: 0.0  # (float) image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # (float) image flip up-down (probability)
fliplr: 0.5  # (float) image flip left-right (probability)
bgr: 0.0 # (float) RGB↔BGR channel swap probability
mosaic: 1.0  # (float) image mosaic (probability)
mixup: 0.0  # (float) image mixup (probability)
cutmix: 0.0 # (float) CutMix augmentation probability
copy_paste: 0.0  # (float) segment copy-paste (probability)
copy_paste_mode: flip # (str) copy-paste strategy for segmentation: flip or mixup
auto_augment: randaugment # (str) classification auto augmentation policy: randaugment, autoaugment, augmix

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

### Create a docker-compose file
Create `./docker-compose_train.yaml`:
```yaml
services:
  yolo_train:
    container_name: yolo_train
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training:/app/training   # training output
      - ./data/resized_yolo:/app/data  # training data
      - ./data/resized_yolo/dataset.yaml:/app/config/dataset.yaml  # dataset settings
      - ./configs/2026-03-11_train.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 -m object_detectors.ultralytics_lmi.run_cmd

```
> [!IMPORTANT]
> Do not modify the target paths inside the container (e.g., `/app/training`, `/app/data`, `/app/config/dataset.yaml`, `/app/config/hyp.yaml`).


### Start training
Spin up the docker containers as shown in [Spin up the container](#spin-up-the-container). **Ensure you load `docker-compose_train.yaml`.**

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


## Prediction
Create `./configs/2026-03-11_predict.yaml`. (`imgsz` should be a list of [h,w]):
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose
mode: predict  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark

# Prediction settings 
imgsz: 640,640 # (list) input images size as list[h,w] for predict and export modes
conf:  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
iou: 0.7 # (float) IoU threshold used for NMS
max_det: 300  # (int) maximum number of detections per image

# less likely to be used 
show: False  # (bool) show results if possible
save_txt: False  # (bool) save results as .txt file
save_conf: False  # (bool) save results with confidence scores
save_crop: False  # (bool) save cropped images with results
show_labels: True  # (bool) show object labels in plots
show_conf: True  # (bool) show object confidence scores in plots
vid_stride: 1  # (int) video frame-rate stride
line_width:   # (int, optional) line width of the bounding boxes, auto if missing
visualize: False  # (bool) visualize model features
augment: False  # (bool) apply image augmentation to prediction sources
agnostic_nms: False  # (bool) class-agnostic NMS
classes:  # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
retina_masks: False  # (bool) use high-resolution segmentation masks
show_boxes: True  # (bool) Show boxes in segmentation predictions

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Create `./docker-compose_predict.yaml`:
```yaml
services:
  yolo_predict:
    container_name: yolo_predict
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./prediction:/app/prediction  # output path
      - ./training/2026-03-11/weights:/app/trained-inference-models   # trained model path, where it has best.pt
      - ./data/resized_yolo/images:/app/data  # input data path
      - ./configs/2026-03-11_predict.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 -m object_detectors.ultralytics_lmi.run_cmd

```

### Start prediction
Spin up the container as shown in [Spin up the container](#spin-up-the-container). **Ensure you load `docker-compose_predict.yaml`.**


## Generate TensorRT engines

Create `./configs/2026-03-11_trt.yaml`:
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose
mode: export  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark

# Export settings 
format: engine  # (str) format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats
half: True  # (bool) use half precision (FP16)
imgsz: 640,640 # (list) input images size as list[h,w] for predict and export modes
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
  yolo_trt:
    container_name: yolo_trt
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./training/2026-03-11/weights:/app/trained-inference-models   # trained model path, which includes a best.pt
      - ./configs/2026-03-11_trt.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 -m object_detectors.ultralytics_lmi.run_cmd
```

### Engine Generation on x86 systems
Spin up the container as shown in [Spin up the container](#spin-up-the-container). **Ensure you load `docker-compose_trt.yaml`.**


### Engine Generation on arm systems
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

Spin up the container as shown in [Spin up the container](#spin-up-the-container). Ensure you load `./docker-compose_trt.yaml`.
