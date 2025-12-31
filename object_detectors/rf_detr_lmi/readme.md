## RF-Detr

### Training

##### Dataset

The dataset for training should be a coco formated dataset, with the category id starting from 0 instead of 1. The dataset should be in the following format.

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```
*All three folders are required for training and should consist of _annotations.coco.json*

##### Dockerfile

```Dockerfile
FROM nvcr.io/nvidia/pytorch:25.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user opencv-python-headless
RUN pip install ultralytics label-studio-sdk shapely roboflow python-dotenv
RUN git clone https://github.com/roboflow/rf-detr.git
RUN pip install -e rf-detr
RUN pip install onnxsim onnx-graphsurgeon pycuda seaborn



# clone LMI AI Solutions repository
WORKDIR /repos
RUN git clone -b FAIE-2765 https://github.com/lmitechnologies/LMI_AI_Solutions.git
RUN cd LMI_AI_Solutions && pip install -e lmi_utils
RUN cd LMI_AI_Solutions && pip install -e object_detectors
RUN cd LMI_AI_Solutions && pip install -e anomaly_detectors
RUN cd LMI_AI_Solutions && pip install -e classifiers
```

##### Configuration

The following is an example yaml configuration to train an object detector.

```yaml
model_type: medium
operation: train # or convert
training:
    dataset_dir: /app/data/coco/dataset # the path to the dataset directory
    epochs: 10 # number of epochs to train
    batch_size: 4 # batch size
    grad_accum_steps: 4 # gradient accumulation steps
    lr: 1e-4 # learning rate
    output_dir: /app/output/ # output directory
    resolution: 256 # image size (square image only)
```
Other traning parameters that can be passed in:
| Parameter | Description |
| :--- | :--- |
| **dataset_dir** | Specifies the COCO-formatted dataset location with `train`, `valid`, and `test` folders, each containing `_annotations.coco.json`. Ensures the model can properly read and parse data. |
| **output_dir** | Directory where training artifacts (checkpoints, logs, etc.) are saved. Important for experiment tracking and resuming training. |
| **epochs** | Number of full passes over the dataset. Increasing this can improve performance but extends total training time. |
| **batch_size** | Number of samples processed per iteration. Higher values require more GPU memory but can speed up training. Must be balanced with `grad_accum_steps` to maintain the intended total batch size. |
| **grad_accum_steps** | Accumulates gradients over multiple mini-batches, effectively raising the total batch size without requiring as much memory at once. Helps train on smaller GPUs at the cost of slightly more time per update. |
| **lr** | Learning rate for most parts of the model. Influences how quickly or cautiously the model adjusts its parameters. |
| **lr_encoder** | Learning rate specifically for the encoder portion of the model. Useful for fine-tuning encoder layers at a different pace. |
| **resolution** | Sets the input image dimensions. Higher values can improve accuracy but require more memory and can slow training. Must be divisible by 56. |
| **weight_decay** | Coefficient for L2 regularization. Helps prevent overfitting by penalizing large weights, often improving generalization. |
| **device** | Specifies the hardware (e.g., `cpu` or `cuda`) to run training on. GPU significantly speeds up training. |
| **use_ema** | Enables Exponential Moving Average of weights, producing a smoothed checkpoint. Often improves final performance with slight overhead. |
| **gradient_checkpointing** | Re-computes parts of the forward pass during backpropagation to reduce memory usage. Lowers memory needs but increases training time. |
| **checkpoint_interval** | Frequency (in epochs) at which model checkpoints are saved. More frequent saves provide better coverage but consume more storage. |
| **resume** | Path to a saved checkpoint for continuing training. Restores both model weights and optimizer state. |
| **tensorboard** | Enables logging of training metrics to TensorBoard for monitoring progress and performance. |
| **wandb** | Activates logging to Weights & Biases, facilitating cloud-based experiment tracking and visualization. |
| **project** | Project name for Weights & Biases logging. Groups multiple runs under a single heading. |
| **run** | Run name for Weights & Biases logging, helping differentiate individual training sessions within a project. |
| **early_stopping** | Enables an early stopping callback that monitors mAP improvements to decide if training should be stopped. Helps avoid needless epochs when mAP plateaus. |
| **early_stopping_patience** | Number of consecutive epochs without mAP improvement before stopping. Prevents wasting resources on minimal gains. |
| **early_stopping_min_delta** | Minimum change in mAP to qualify as an improvement. Ensures that trivial gains don’t reset the early stopping counter. |
| **early_stopping_use_ema** | Whether to track improvements using the EMA version of the model. Uses EMA metrics if available, otherwise falls back to regular mAP. |

The following are the available models for training:

* RF-DETR Nano
* RF-DETR Small
* RF-DETR Base
* RF-DETR Medium
* RF-DETR Large

##### Initiating Training

The following is an example docker compose file for training:

```yaml
version: "3.9"
services:
  postprocess:
    container_name: train
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    # ports:
    # ports:
    #   - 6006:6006 # tensorboard
    volumes:
      - ./configs/:/app/configs/
      - ./preprocessed/:/app/data/
      - ./output:/app/output
    command: >
      python3 -m rf_detr_lmi.cli -c /app/configs/rf-detr.yaml
```

##### Converting to TensorRT Engine

conversion config file:

```yaml
model_type: medium
operation: convert 
conversion:
    pretrain_weights: /app/output/v1/checkpoint_best_total.pth # the pth file
    resolution: 256 # image size
```

docker-compose file
```yaml
version: "3.9"
services:
  postprocess:
    container_name: train
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    # ports:
    # ports:
    #   - 6006:6006 # tensorboard
    volumes:
      - ./configs/:/app/configs/
      - ./preprocessed/:/app/data/
      - ./output:/app/output
    command: >
      python3 -m rf_detr_lmi.cli -c /app/configs/rf-detr.convert.yaml
```

