# Train and test YOLOv8 models
This is the tutorial walking through how to train and test YOLOv8 models. This tutorial assumes that the training and testing processes are running in a **x86** system.

## System requirements
### Model training
- x86 system
- CUDA 12.1
- ubuntu 20.04

### TensorRT engine generation if the model needs to be deployed on Nvidia Jetson devices
- arm system
- JetPack 5.0.2

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
├── validation
├── docker-compose_preprocess.yaml
├── docker-compose_train.yaml
├── docker-compose_val.yaml
├── docker-compose_trt.yaml
├── dockerfile
```


## Create a dockerfile
Let's create a file `./dockerfile`. It installs the dependencies and clone LMI_AI_Solutions repository inside the docker container.
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

## Prepare the dataset
Prepare the dataset by the followings:
- resize images and labels in csv (optional)
- convert labeling data to YOLO format

**YOLO models require the dimensions of images to be dividable by 32**. The resizing is optional if the dimensions already meet this requirement. In this tutorial, we resize images to 640x320 as an example.

### Create a bash script for data processing
First, create a bash script `./preprocess/2023-07-19.sh`, which resizes images and converts data to yolo format. In the end, it will generate a yolo-formatted dataset in `/app/data/resized_yolo`.
```bash
# modify to your data path
input_path=/app/data/allImages
# modify the width and height according to your data
W=640
H=320

# import the repo paths
source /repos/LMI_AI_Solutions/lmi_ai.env

# extract labels from json
python -m label_utils.via_json_to_csv -d $input_path --output_fname labels.csv

# resize images with labels
python -m label_utils.resize_with_csv --path_imgs $input_path --out_imsz $W,$H --path_out /app/data/resized

# convert to yolo format
# remote the --seg flag if you want to train a object detection model
python -m label_utils.convert_data_to_yolo --path_imgs /app/data/resized --path_out /app/data/resized_yolo --seg
```

### Create a docker-compose file
To run the bash script in the container, we need to create a file `./docker-compose_preprocess.yaml`. We want to mount the location in host to a location in container so that the file/folder changes in container are picked up in host. Assume that the path to the original data in host is `./data/allImages`. Below, we mount `./data` in the host to `/app/data` in the container. Also, mount the bash script to `/app/preprocess/preprocess.sh`. Let's define the commands in the command section in the docker-compose file, where it runs the mounted preprocess script in the container.
```yaml
version: "3.9"
services:
  yolov8_preprocess:
    container_name: yolov8_preprocess
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
      - ./data:/app/data # format is location_in_host:location_in_container
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
# "-f" specifies the yaml file to load
docker compose -f docker-compose_preprocess.yaml up
```
Once it finishs executing, the yolo formatted dataset will be created in the host: `./data/resized_yolo`.


## Create a dataset yaml file indicating the location of the dataset and its classes
After converting data to yolo format, a json file can be found in `./data/resized_yolo/class_map.json`. The order of class names in the yaml file **must match with** the order of names in the json file. 

Below is what is in the class_map.json:
```json
{"peeling": 0, "scuff": 1, "white":2}
```

Below is the yaml file that need to be created. Let's save it as `./config/2023-07-19_dataset.yaml`.
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
To train the model, we need to create a hyperparameter yaml file and create a `./docker-compose_train.yaml` file.

### Create a hyperparameter file
 We need to crete a file `./config/2023-07-19_train.yaml`. Below shows an example of training a **medium-size yolov8 instance segmentation model** with the image size of 640. To train object detection models, set `task` to `detect`. If the training images are square, set `rect` to `False`.
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose, where classify, pose are NOT tested
mode: train  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark, where val, track, benchmark are NOT tested

# training settings
epochs: 300  # (int) number of epochs
batch: 16  # (int) number of images per batch (-1 for AutoBatch)
model: yolov8m-seg.pt # (str) one of yolov8n.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolov8n-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt
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

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

### Create a docker-compose file
Let's create a file `./docker-compose_train.yaml`. It mount the host locations to the required directories in the container and run the script `cmd.py`, which load the hyperparameters and do the task that was specified in the file `./config/2023-07-19_train.yaml`.
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
    ports:
      - 6008:6008 # tensorboard
    volumes:
      - ./training:/app/training   # training output
      - ./data/resized_yolo:/app/data  # training data
      - ./config/2023-07-19_dataset.yaml:/app/config/dataset.yaml  # dataset settings
      - ./config/2023-07-19_train.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/object_detectors/yolov8_lmi/cmd.py

```

Spin up the docker containers to train the model as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_train.yaml` instead.** By default, once the training is done, the cmd.py script will create a folder named by today's date in `training` folder, i.e. `training/2023-07-19`.

## Monitor the training progress (optional)
While training process is running, open another terminal. 
```bash
# Log in the container which hosts the training process
docker exec -it CONTAINER_ID bash 

# track the training progress using tensorboard
tensorboard --logdir /app/training/2023-07-19 --port 6008
```

Execuate the command above and go to http://localhost:6008 to monitor the training.


# Validation
Create another hyperparameter file `./config/2023-07-19_val.yaml`. The `imgsz` should be a list of [h,w].
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
boxes: True  # (bool) Show boxes in segmentation predictions

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Create a file `./docker-compose_val.yaml` as below.
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
      - ./validation:/app/validation  # validation output path
      - ./training/2023-07-19/weights:/app/trained-inference-models   # trained model path, where it has best.pt
      - ./data/resized_yolo/images:/app/data  # input data path
      - ./config/2023-07-19_val.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/object_detectors/yolov8_lmi/cmd.py

```
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_val.yaml` instead.** Then, the output results are saved in `./validation/2023-07-19`.

# Generate TensorRT engines
Similarly, create another hyperparamter yaml file named `./config/2023-07-19_trt.yaml` as below:
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose, where classify, pose are NOT tested
mode: export  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark, where val, track, benchmark are NOT tested

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

Create a docker-compose file `./docker-compose_trt.yaml`:
```yaml
version: "3.9"
services:
  yolov8_trt:
    container_name: yolov8_trt
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
      - ./training/2023-08-16/weights:/app/trained-inference-models   # trained model path, which includes a best.pt
      - ./config/2023-07-19_trt.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      python3 /repos/LMI_AI_Solutions/object_detectors/yolov8_lmi/cmd.py

```

If the deployment platform is arm based, use the arm dockerfile instead. Here is the dockerfile that works in Nvidia Jetson JP 5.0.2:
https://github.com/lmitechnologies/LMI_AI_Solutions/blob/ais/object_detectors/yolov8_lmi/docker/arm.dockerfile


Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_trt.yaml` instead.** Then, the tensorRT engine is generated in `./training/2023-07-19/weights`.
