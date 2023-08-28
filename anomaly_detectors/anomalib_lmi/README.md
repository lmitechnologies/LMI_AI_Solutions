# Anomalib Integration
This folder contains the [Anomalib](https://github.com/openvinotoolkit/anomalib) integration script for anomaly detection, supported models:
- [patchcore](https://arxiv.org/abs/2106.08265)
- [padim](https://arxiv.org/abs/2011.08785)

# Usage

## Prepare dataset
1. Structure training data like this
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
2. Although test and the ground_truth are optional, it enables the training to generate insightful metrics
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
## Training Overview
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
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && cd LMI_AI_Solutions/anomaly_detectors && git submodule update --init submodules/anomalib
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
### 4. Validate the Model






1. bash main.sh -a train -d /path/to/train/data/root_dir -o /path/to/outdir
2. For more detailed usage, execute bash main.sh -h
2. Check the corresponding [config file](https://openvinotoolkit.github.io/anomalib/reference_guide/algorithms/patchcore.html), important fields:
```bash
# general
dataset
|---- image_size

model
|---- backbone

# patchcore specific
model
|---- layers
|---- coreset_sampling_ratio
|---- num_neighbors

# padim specific
model
|---- n_features
```
4. Find the trained model in outdir/results

## Generate TRT engine
1. bash main.sh -a convert -x /path/to/onnx  -o /path/to/outdir
2. Find the generated engine at outdir/engines
3. Run without -x /path/to/onnx, it defaults to the latest onnx folder in the outdir/results/{modeltype}/trained-models

## Unit Test
1. bash main.sh -a test -e /path/to/trt/engine -t /path/to/test/data -o /path/to/outdir
2. Find annotated prediction result at outdir/predictions
3. Run without -e /path/to/trt/engine, it defaults to the latest trt engine folder in the outdir/engines

## Code Example
```python
am = AnomalyModel(padim_or_patchcore_trt_engine)

am.warmup()

decision, annotation, more_details = am.predict(
        image, 
        err_threshold   # reference the adaptive err_threshold from model/run/weights/onnx/metadata.json 
)
```