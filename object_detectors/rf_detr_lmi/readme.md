# RF-Detr

Last updated: 2026-08-19

## Training

### Dataset

The dataset for training should be a coco formated dataset. The dataset should be in the following format:

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

### Dockerfile

```Dockerfile
FROM nvcr.io/nvidia/pytorch:25.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install libgl1 -y
RUN pip install onnxruntime-gpu
RUN sed -i '/setuptools/d' /etc/pip/constraint.txt && \
    pip install --upgrade setuptools
RUN sed -i '/lightning-utilities/d;/pycocotools/d' /etc/pip/constraint.txt && \
    pip install label_studio_sdk shapely "rfdetr[train]==1.8.3"

# clone repos
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && pip install -e LMI_AI_Solutions

```

### Configuration

The following is an example yaml configuration to train an object detector. **Seg model resolutions must be divisible by 24 and detection models by 32.**

```yaml
model_type: small
operation: train  # train, convert, or export
task: seg         # od or seg
# pretrain_weights: /path/to/checkpoint.pth  # optional; see below
training:
    dataset_dir: /app/data/coco/dataset # the path to the dataset directory
    epochs: 20                          # number of epochs to train
    batch_size: 4                       # batch size
    grad_accum_steps: 4                 # gradient accumulation steps
    lr: 1e-4                            # learning rate
    output_dir: /app/training           # output directory
    resolution: 384                     # image size (square image only)
```

The optional top-level `pretrain_weights` controls the weights training starts from:

- key absent: the variant's default COCO-pretrained weights (downloaded on first use)
- a checkpoint path: warm-start training from that checkpoint's weights
- explicit `null`: skip base-weight loading entirely; use together with `training.resume`, which restores model weights and optimizer state from the resumable checkpoint

<details>
<summary>Other training parameters that can be passed in</summary>

| Parameter | Description |
| :--- | :--- |
| **dataset_dir** | Specifies the COCO-formatted dataset location with `train`, `valid`, and `test` folders, each containing `_annotations.coco.json`. Ensures the model can properly read and parse data. |
| **output_dir** | Directory where training artifacts (checkpoints, logs, etc.) are saved. Important for experiment tracking and resuming training. |
| **versioned_output_dir** | When `true` (the default), each run writes into a fresh date-versioned subdirectory of `output_dir` (e.g. `2026-07-09-v1`). Set to `false` to write directly into `output_dir` when a predictable path is needed. |
| **epochs** | Number of full passes over the dataset. Increasing this can improve performance but extends total training time. |
| **batch_size** | Number of samples processed per iteration. Higher values require more GPU memory but can speed up training. Must be balanced with `grad_accum_steps` to maintain the intended total batch size. |
| **grad_accum_steps** | Accumulates gradients over multiple mini-batches, effectively raising the total batch size without requiring as much memory at once. Helps train on smaller GPUs at the cost of slightly more time per update. |
| **lr** | Learning rate for most parts of the model. Influences how quickly or cautiously the model adjusts its parameters. |
| **lr_encoder** | Learning rate specifically for the encoder portion of the model. Useful for fine-tuning encoder layers at a different pace. |
| **resolution** | Sets the input image dimensions. Higher values can improve accuracy but require more memory and can slow training. |
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

</details>

### Initiating Training

The following is an example docker compose file for training:

```yaml
services:
  postprocess:
    container_name: train
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./configs/:/app/configs/
      - ./preprocessed/:/app/data/
      - ./training:/app/training
    command: >
      python3 -m object_detectors.rf_detr_lmi.cli -c /app/configs/train.yaml
```

## Converting to TensorRT Engine

conversion config file:

```yaml
model_type: small
operation: convert
task: seg         # od or seg
format: tensorrt  # onnx tensorrt
conversion:
    pretrain_weights: /app/training/checkpoint_best_total.pth   # must be .pth file
    resolution: 384
    device: cuda
    output_dir: /app/training
```

docker-compose file
```yaml
services:
  postprocess:
    container_name: convert
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./configs/:/app/configs/
      - ./training/2026-04-22-v1:/app/training
    command: >
      python3 -m object_detectors.rf_detr_lmi.cli -c /app/configs/convert.yaml
```

## Exporting to ONNX

The export operation uses rfdetr's native exporter and writes the fixed filename `model.onnx`, with the class names embedded in the file's own metadata — the `RfdetrModel` ONNX/TensorRT backends read them from there, so the model deploys as a single file.

```yaml
model_type: small
operation: export
task: seg         # od or seg
pretrain_weights: /app/training/checkpoint_best_total.pth   # required; must be a .pth file
export:
    output_dir: /app/training   # receives model.onnx
    resolution: 384
    opset_version: 17           # optional, default 17
```

Any additional keys under `export` (e.g. `device`) are passed to the model constructor.

## Inference

`RfdetrModel` picks a backend from the file extension:

| Extension | Backend | Device |
| :--- | :--- | :--- |
| `.pth` | rfdetr checkpoint, JIT-traced at load | cuda or cpu |
| `.onnx` | ONNX Runtime | cuda or cpu |
| `.engine` | TensorRT | cuda |

The `.pth` backend takes `model_type` and an optional `batch_size` (fixed at load time, since the traced graph bakes it in).
The `.onnx` and `.engine` backends read the batch size and resolution from the model, and take class names from `class_map`
or from the metadata embedded at export (`metadata_props` in an ONNX, a JSON header on an engine). Passing `class_map`
is required for a model exported elsewhere, which carries no embedded names.

docker-compose file
```yaml
services:
  postprocess:
    container_name: inference
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./preprocessed/test:/app/data/
      - ./training:/app/training
    command: >
      python3 -m object_detectors.rf_detr_lmi.infer --weights /app/training/checkpoint_best_total.pth --input /app/data/test --output /app/training/predictions --model_type seg-small
```
