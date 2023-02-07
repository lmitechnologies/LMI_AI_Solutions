# LMI AI Solutions
This repo contains the automation script for tf-trt conversion:

TF to TensorRT
- PaDiM
- Faster RCNN
- Mask RCNN
- Efficient Net

Pytorch to TensorRT
- Yolov5

## Usage
1. set variables in .env
2. launch the docker compose to generate the trt engine for given model, the output directory is ${BASELINE_SAVED_MODEL_DIR/WEIGHT_PATH}_out
3. Further for TF to TensorRT, benchmark the saved model and generated trt saved model with provided data_dir
