# Train and test YOLOv8 models
This is the tutorial how to train and test the YOLOv8 models. This tutorial assume that the training and testing happen in a x86 system.

## Directory structure
```
├── config
│   ├── dockerfile
│   ├── docker-compose.yaml
│   ├── 2023-07-19.yaml
│   ├── 2023-07-19_hyp.yaml
├── preprocess
│   ├── 2023-07-19.sh
├── data
│   ├── allImages
│   │   ├── *.png
├── training
├── validation
```


## Create a dockerfile
Let's create a file named `dockerfile` in `config`.
In the `dockerfile`, it installs the dependencies and clone LMI AI Solutions repository. 
```docker
# last version running on ubuntu 20.04, require CUDA 12.1 
FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user  opencv-python
RUN pip install ultralytics -U

# clone LMI AI Solutions repository
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git

```

## Prepare the datasets
Prepare the datasets by the followings:
- resize images [optional]
- convert labeling data to YOLO format

**YOLO models require the dimensions of images to be dividable by 32**. The resizing is optional if the dimensions already meet this requirement. In this tutorial, we resize images to 640x320 as an example.

### Create a bash script
First, create a bash script named `./preprocess/2023-07-19.sh` to do the resize and data conversion to yolo format. We usually use today's date as the file name.
```bash
# modify the width and height according to your data
input_path=/app/data/allImages
W=640
H=320

source /repos/LMI_AI_Solutions/lmi_ai.env

python -m label_utils.resize_with_csv --path_imgs $input_path --out_imsz $W,$H --path_out /app/data/resized

python -m label_utils.convert_data_to_yolo --path_imgs /app/data/resized --path_out /app/data/resized_yolo
```

### Create a docker-compose.yaml file
In order to run the bash script in the container, we need to create a `./config/docker-compose.yaml` file.
Assume that the path to the original data in host is `../data/allImages`. Below, we mount the folder `../data` to `/app/data` in the docker container. Also, mount the bash script to `/app/preprocess/preprocess.sh`.
```yaml
version: "3.9"
services:
  yolov8_data:
    container_name: yolov8_data
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
      - ../data:/app/data
      - ../preprocess/2023-07-19.sh:/app/preprocess/preprocess.sh
    command: >
      python /app/preprocess/preprocess.sh
```

### Spin up the container
Go to `./config` and spin up the container with the following commands: 
```bash
# build the container
docker compose build

# spin up the container
docker compose up
```
Once it finishs executing, the yolo formatted dataset will be created in the host: `../data/resized_yolo`.


## Create a yaml file indicating the location of the training dataset
After converting data to yolo format, a json file can be found in `../data/resized_yolo/class_map.json`. The order of class names in the yaml file **must match with** the order of names in the json file. 

Below is what is in the class_map.json:
```json
{"peeling": 0, "scuff": 1, "white":2}
```

Below is the yaml file that need to be created. Let's save it as `./config/2023-07-19.yaml`.
```yaml
path: /app/data # dataset root dir (must use absolute path!)
train: images  # train images (relative to 'path')
val: images  # val images (relative to 'path')
test:  # test images (optional)
 
# Classes
names: # class names must match with the names in class_map.json
  0: peeling
  1: scuff
  2: white
```


## Train the model
To train the model, we need to create a hyperparameter yaml file and modify the existing `docker-compose.yaml` file.

### Create a hyperparameter yaml file
 We need to crete another file `./config/2023-07-19_hyp.yaml`. Below shows you an example of training a **medium-size yolov8 instance segmentation model** with the image size of 640. To train object detection models, set `task` to `detect`. If the training images are square, set `rect` to `False`.
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose
mode: train  # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark

# training settings
epochs: 300  # (int) number of epochs
batch: 16  # (int) number of images per batch (-1 for AutoBatch)
model: yolov8m-seg.pt # (str) one of yolov8n.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolov8n-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt
imgsz: 640  # (int) input images size, use the larger dimension if h!=w
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

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

### Modify the docker-compose file
Let's modify the `docker-compose.yaml` file as below. It mount the host locations to the required directories in the container and run the script in the container: `/repos/LMI_AI_Solutions/object_detectors/yolov8_lmi/cmd.py`.
```yaml
version: "3.9"
services:
  yolov8_train:
    container_name: yolov8_train
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
      - ../training:/app/training   # training output
      - ../data/640x256/yolo:/app/data  # training data
      - ./2023-07-19.yaml:/app/config/dataset.yaml  # dataset settings
      - ./2023-07-19_hyp.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python /repos/LMI_AI_Solutions/object_detectors/yolov8_lmi/cmd.py

```

Once, modification is done. Spin up the docker containers to train the model as shown in [spin-up-the-container](#spin-up-the-container).

## Monitor the training progress
```bash
tensorboard --logdir ./training/2023-07-19
```
While training process is running, open another terminal.
Execuate the command above and go to http://localhost:6006 to monitor the training.


# Testing
Create another hyperparameter file `./config/2023-07-19_val.yaml`. The `imgsz` should be a list of [h,w].
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose
mode: predict  # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark

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
boxes: True  # (bool) Show boxes in segmentation predictions

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Modify the docker-compose file as below.
```yaml
version: "3.9"
services:
  yolov8_val:
    container_name: yolov8_val
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
      - ../validation:/app/validation  # validation output path
      - ../training/2023-07-19/weights:/app/trained-inference-models   # trained model path, where it has best.pt
      - ../data/640x256/test:/app/data  # input data path
      - ./2023-07-19_val.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python /repos/LMI_AI_Solutions/object_detectors/yolov8_lmi/cmd.py

```
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). Then, the output results are saved in `./validation/2023-07-19`.

# Generate TensorRT engines
Similarly, create another hyperparamter yaml file named `2023-07-19_trt.yaml` as below:
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose
mode: export  # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark

# Export settings 
format: engine  # (str) format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats
half: True  # (bool) use half precision (FP16)
imgsz: 320,640 # (list) input images size as list[h,w] for predict and export modes
device: 0 # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu

# less likely used settings 
keras: False  # (bool) use Kera=s
optimize: False  # (bool) TorchScript: optimize for mobile
int8: False  # (bool) CoreML/TF INT8 quantization
dynamic: False  # (bool) ONNX/TF/TensorRT: dynamic axes
simplify: False  # (bool) ONNX: simplify model
opset:  # (int, optional) ONNX: opset version
workspace: 4  # (int) TensorRT: workspace size (GB)
nms: False  # (bool) CoreML: add NMS

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Modify the docker-compose file:
```yaml
version: "3.9"
services:
  yolov8_trt:
    container_name: yolov8_trt
    build:
      context: .
      dockerfile: x86.dockerfile
    ipc: host
    deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
    volumes:
      - ../training/2023-08-16/weights:/app/trained-inference-models   # trained model path, which includes a best.pt
      - ./2023-07-19_trt.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python /repos/LMI_AI_Solutions/object_detectors/yolov8_lmi/cmd.py

```

Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). Then, the output tensorRT engine is saved in `./training/2023-07-19/weights`.
