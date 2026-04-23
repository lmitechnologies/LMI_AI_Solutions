# Train MaskRCNN

## Dataset

The required dataset format is COCO, in the following structure:


```text
<DATASETS DIR>/
    -<DATASET_NAME>/
        --images/
            ...
        --annotations.json
```
example:

```text
coco/
    train/
        images/
            ...
        annotations.json
```

*Dataset name should be the same name as the name declared in the config yaml file*

## Training

- [Configuration](#configuration)
- [Dockerfile](#dockerfile)
- [Training](#train)

### Configuration

The following is an example configuration file for training a maskrcnn model

```yaml
DATASETS:
  TEST: [] # define the test dataset similar to the train dataset or leave it as empty list.
  TRAIN:
  - train # has the match name

DATALOADER:
  NUM_WORKERS: 4
  FILTER_EMPTY_ANNOTATIONS: true

MODEL:
  ROI_HEADS:
    NUM_CLASSES: 1
  
INPUT:
  MIN_SIZE_TRAIN: 
  - 512
  MAX_SIZE_TRAIN: 512
  MIN_SIZE_TEST: 512
  MAX_SIZE_TEST: 512
  FORMAT: RGB
  MASK_FORMAT: # "polygon"  # alternative: "bitmask"

# Training parameters
SOLVER:
  AMP:
    ENABLED: false
  BASE_LR: 0.02
  BIAS_LR_FACTOR: 1.0
  CHECKPOINT_PERIOD: 5000 # Save a checkpoint after every N number of iterations
  GAMMA: 0.1
  IMS_PER_BATCH: 2 # Number of images per batch
  LR_SCHEDULER_NAME: WarmupMultiStepLR
  MAX_ITER: 3000 # Maximum number of iterations default is 270000 for 3x schedule please change it accordingly

TEST:
  # Feel free to add more augmentations or change the min/max sizes for the images
  DETECTIONS_PER_IMAGE: 1000 # Number of detections per image
  EVAL_PERIOD: 0 # Runs evaluation every N iterations

# Update the training augmentations here, if you would like to not use one of the augmentations one can comment the out. All of the augmentations are random
AUGMENTATIONS:
  BRIGHTNESS:
    MIN: 0.9
    MAX: 1.1
  # LIGHTING:
  #   SCALE: 0.1
  # FLIP_HORIZONTAL:
  #   PROB: 0.5
  # FLIP_VERTICAL:
  #   PROB: 0.5
  # ROTATION:
  #   MIN: 90
  #   MAX: 90
  # SATURATION:
  #   MIN: 0.9
  #   MAX: 1.1
  # CONTRAST:
  #   MIN: 0.9
  #   MAX: 1.1

```

### Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user opencv-python

# clone LMI AI Solutions repository
WORKDIR /home
RUN git clone https://github.com/facebookresearch/detectron2
RUN pip install --user -e detectron2
RUN pip install tensorboard
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && pip install -e LMI_AI_Solutions
RUN pip install onnx-graphsurgeon onnxruntime
RUN pip install numba

```

### Train

Example docker-compose.yaml file to start a training job

```yaml
services:
  detectron2_lmi_train:
    container_name: detectron2_lmi_train
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training/maskrcnn/:/home/weights   # training output where trained models are saved
      - ./data/coco/:/home/data  # training data please attach to the root folder
      - ./configs/maskrcnn.yaml:/home/config.yaml  # customized hyperparameters
    command: >
      python3 -m detectron2_lmi.cli train
```
*The training process automatically starts tensorboard*

### Tensorboard

Served up at the following address [localhost:6006](http://localhost:6006)
*6006 is the default port*

## Convert to PT

Detectron2 outputs a `.pth` file. To convert it to a regular Pytorch `pt` file please convert it the following way:

```yaml
services:
  detectron2_lmi_convert:
    container_name: detectron2_lmi_convert
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training/maskrcnn/2024-12-26-v3/:/home/weights   # weights (where the pt file is saved)
    stdin_open: true # docker run -i
    tty: true        # docker run -t
    command: >
      python3 -m detectron2_lmi.cli convert --pt
```

## Inference

To run inference please use the following docker-compose.yaml file:

```yaml
services:
  detectron2_lmi_infer:
    container_name: detectron2_lmi_infer
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training/maskrcnn/2024-12-26-v3/:/home/weights   # weights
      - ./data/coco/train/images/:/home/input  # inference data
      - ./prediction/maskrcnn/test/:/home/output  # inference output
      - ./configs/class_map.json:/home/class_map.json  # class map
    stdin_open: true # docker run -i
    tty: true        # docker run -t
    command: >
      python3 -m detectron2_lmi.cli test -w /home/weights/model.pt
```
### Outputs

A LMI formated csv file with all predictions is automatically saved in the the the output folder defined in the docker compose file.

## Convert to TensorRT

To convert to tensorrt a `sample_image.png` is required to be in folder where the weights are stored. The image should be of size thats divizeable by 32. The imagesize should be defined in the config.yaml file shown above for training.

*Default batch size is 1 although batch size can be changed to any batch size using -b*

```yaml
services:
  detectron2_lmi_trt:
    container_name: detectron2_lmi_trt
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training/maskrcnn/2024-12-26-v3/:/home/weights   # weights
    stdin_open: true # docker run -i
    tty: true        # docker run -t
    command: >
      python3 -m detectron2_lmi.cli convert --trt --fp16
```
