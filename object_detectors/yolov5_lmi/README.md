# Train and test YOLOv5 models
This is the tutorial how to train and test the yolov5 object detection models.

## System requirements
- [Docker Engine](https://docs.docker.com/engine/install)
- [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)

### Model training
- x86 CPU
- CUDA >= 12.1
- ubuntu

### TensorRT on GoMax
- JetPack 5.1


## Directory structure
The folder structure below will be created when we go through the tutorial. By convention, we use today's date (i.e. 2023-07-19) as the file name.
```
├── config
│   ├── 2023-07-19_dataset.yaml
│   ├── 2023-07-19_train.yaml
│   ├── 2023-07-19_val.yaml
│   ├── 2023-07-19_trt.yaml
├── preprocess
│   ├── 2023-07-19.sh
├── data
│   ├── allImages
│   │   ├── *.png
│   │   ├── *.json
├── training
│   ├── 2023-07-19
├── validation
│   ├── 2023-07-19
├── docker-compose_preprocess.yaml
├── docker-compose_train.yaml
├── docker-compose_val.yaml
├── docker-compose_trt.x86.yaml
├── dockerfile
├── docker-compose_trt.arm.yaml   # arm system
├── arm.dockerfile                # arm system
```


## Setup the container for training
Create a file `./dockerfile`. It installs the dependencies and clone LMI_AI_Solutions repository inside the docker container.
```docker
# last version running on ubuntu 20.04, require CUDA 12.1 
FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user opencv-python
RUN pip install pillow -U
RUN pip install ultralytics -U

# clone LMI AI Solutions repository
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git
RUN cd LMI_AI_Solutions && git submodule update --init object_detectors/submodules/yolov5
```

## Prepare the dataset
Prepare the dataset by the followings:
- resize images and labels in csv (optional)
- convert labeling data to YOLO format

**YOLO models require the dimensions of images to be dividable by 32**. In this tutorial, we resize images to 640x320.

### Create a script for data processing
First, create a script `./preprocess/2023-07-19.sh`, which convert labels from VGG json to csv, resizes images, and converts data to yolo format. 
```bash
# modify to your data path
input_path=/app/data/allImages
# modify the width and height according to your data
W=640
H=320

# import the repo paths
source /repos/LMI_AI_Solutions/lmi_ai.env

# convert labels from VGG json to csv
python -m label_utils.via_json_to_csv -d $input_path --output_fname labels.csv

# resize images with labels
python -m label_utils.resize_with_csv --path_imgs $input_path --width $W --height $H --path_out /app/data/resized

# convert to yolo format
# remote the --seg flag if you want to train a object detection model
python -m label_utils.convert_data_to_yolo --path_imgs /app/data/resized --path_out /app/data/resized_yolo --seg
```

### Create a docker-compose file
To run the bash script in the container, we need to create a file `./docker-compose_preprocess.yaml`. We mount the location in host to a location in container so that the file/folder changes in container are reflected in host. Assume that the path to the original data in host is `./data/allImages`. Below, mount `./data` in the host to `/app/data` in the container. Also, mount the bash script to `/app/preprocess/preprocess.sh`.
```yaml
version: "3.9"
services:
  yolov5_preprocess:
    container_name: yolov5_preprocess
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
      # mount location_in_host:location_in_container
      - ./data:/app/data 
      - ./preprocess/2023-07-19.sh:/app/preprocess/preprocess.sh
    command: >
      bash /app/preprocess/preprocess.sh
```

### Spin up the container
To run the command in the docker-compose file, spin up the container using the following commands: 
```bash
# build the container
# "-f" specifies the yaml file to load
docker compose -f docker-compose_preprocess.yaml build

# spin up the container
# "-f" specifies the yaml file to load
docker compose -f docker-compose_preprocess.yaml up
```
Once it finishs, the yolo format dataset will be created in `./data/resized_yolo`.


## Train the model
To train the model, we need to create the following files: a dataset file, a hyperparameter file and a docker-compose file.


### Create a dataset file
Create a dataset file to tell the model where is the dataset and what are the classes. The classes information can be found in `./data/resized_yolo/class_map.json`. 

Below is what is in the class_map.json:
```json
{"peeling": 0, "scuff": 1, "white":2}
```

Below is the yaml file that need to be created. Save it as `./config/2023-07-19_dataset.yaml`.
```yaml
path: /app/data  # dataset root dir (must use absolute path!)
train: images  # train images (relative to 'path')
val: images  # val images (relative to 'path')
test:  # test images (optional)
 
# Classes
names: # class names must match with the names in class_map.json
  0: peeling
  1: scuff
  2: white
```
The order of class names in the yaml file **must match with** the order of names in the json file. 


### Create a hyperparameter file
Create a file `./config/2023-07-19_train.yaml`. Below shows an example of training a **medium-size yolov5 instance segmentation model** with the image size of 640. To train object detection models, set `task` to `detect`. If the training images are square, set `rect` to `False`.

```yaml
task: segment  # (str) YOLO task, i.e. detect, segment
mode: train  # (str) YOLO mode, i.e. train, predict, export

# training settings
epochs: 300  # (int) number of epochs
batch: 16  # (int) number of images per batch (-1 for AutoBatch)
model: yolov5m-seg.pt # (str) one of yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt, yolov5s-seg.pt, yolov5m-seg.pt, yolov5l-seg.pt, yolov5x-seg.pt
imgsz: 640  # (int) input images size, use the larger dimension if rectangular image
patience: 50  # (int) epochs to wait for no observable improvement for early stopping of training
rect: True  # (bool) use rectangular images for training if mode='train' or rectangular validation if mode='val'
exist_ok: False  # (bool) whether to overwrite existing training folder
resume: False  # (bool) resume training from last checkpoint

# data augmentation hyperparameters
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

# more hyperparameters: https://github.com/lmitechnologies/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml
```


### Create a docker-compose file
Create a file `./docker-compose_train.yaml`. It mounts the host locations to the required locations in the container and run the script `cmd.py`, which load the hyperparameters and perform the specified task. 

```yaml
version: "3.9"
services:
  yolov5_train:
    container_name: yolov5_train
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
      - ./data/resized_yolo:/app/data  # training data
      - ./config/2023-07-19_dataset.yaml:/app/config/dataset.yaml  # dataset info
      - ./config/2023-07-19_train.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/object_detectors/yolov5_lmi/cmd.py
```
Note: Do **NOT** modify the required locations in the container, such as `/app/training`, `/app/data`, `/app/config/dataset.yaml`, `/app/config/hyp.yaml`.


### Start training
Spin up the docker containers as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_train.yaml`.** Once the training is done, A output folder will be generated in `./training/2023-07-19`.


### Monitor the training progress (optional)
While training process is running, open another terminal. 
```bash
# Log in the container which hosts the training process
docker exec -it CONTAINER_ID bash 

# track the training progress using tensorboard
tensorboard --logdir /app/training/2023-07-19 --port 6006
```
Execuate the command above and go to http://localhost:6006 to monitor the training.


## Validation
Create a hyperparameter file `./config/2023-07-19_val.yaml`. The `imgsz` should be a list of [h,w].

```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose, where classify, pose are NOT tested
mode: predict  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark, where val, track, benchmark are NOT tested

# Prediction settings 
imgsz: 320,640 # (list) input images size as list[h,w] for predict and export modes
conf:  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
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
```

Create a file `./docker-compose_val.yaml`

```yaml
version: "3.9"
services:
  yolov5_val:
    container_name: yolov5_val
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
      - ./validation:/app/validation  # validation output
      - ./data/test:/app/data  # input data
      - ./training/2023-07-19/weights:/app/trained-inference-models   # contains a best.pt
      - ./config/2023-07-19_val.yaml:/app/config/hyp.yaml  # customized hyperparameters
      - ./config/2023-07-19_dataset.yaml:/app/config/dataset.yaml  # contains class names
    command: >
      python3 /repos/LMI_AI_Solutions/object_detectors/yolov5_lmi/cmd.py
```


### Start validation
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). Ensure to load the `./docker-compose_val.yaml`. The output results are saved in `./validation/2023-07-19`.


## Generate TensorRT engines
The TensorRT egnines can be generated in two systems: x86 and arm. Both systems share the same hyperparameter file, while the dockerfile and docker-compose file are different.

Create a file `./config/2023-07-19_trt.yaml` that works for both systems.

```yaml
task: segment  # (str) YOLO task, i.e. detect, segment
mode: export  # (str) YOLO mode, i.e. train, predict, export

# Export settings 
format: engine  # (str) format to export to
imgsz: 320,640 # (list) input images size as list[h,w] for predict and export modes
half: True  # (bool) use half precision (FP16)
device: 0 # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu

# less likely used settings 
dynamic: False  # (bool) ONNX/TF/TensorRT: dynamic axes
simplify: False  # (bool) ONNX: simplify model
opset:  # (int, optional) ONNX: opset version
workspace: 4  # (int) TensorRT: workspace size (GB)
```


### Engine generation on x86 systems
Create a file `./docker-compose_trt.x86.yaml`
```yaml
version: "3.9"
services:
  yolov5_trt:
    container_name: yolov5_trt
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
      - ./training/2023-07-19/weights:/app/trained-inference-models   # contains a best.pt
      - ./config/2023-07-19_trt.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/object_detectors/yolov5_lmi/cmd.py
```

#### Start Generation
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). Ensure to load the `./docker-compose_trt.x86.yaml`. The output engines are saved in `./training/2023-07-19/weights`.


### Engine generation on arm systems
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
RUN cd LMI_AI_Solutions && git submodule update --init object_detectors/submodules/yolov5
```

Create a file `./docker-compose_trt.arm.yaml`,
```yaml
version: "3.9"
services:
  yolov5_trt:
    container_name: yolov5_trt
    build:
      context: .
      dockerfile: arm.dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./training/2023-07-19/weights:/app/trained-inference-models   # contains a best.pt
      - ./config/2023-07-19_trt.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/object_detectors/yolov5_lmi/cmd.py
```

#### Start Generation
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). Ensure to load the `./docker-compose_trt.arm.yaml`. The output engines are saved in `./training/2023-07-19/weights`.
