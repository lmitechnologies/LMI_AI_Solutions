# LMI AI Solutions
This repo contains the automation script for tf-trt conversion:
- PaDiM
- Faster RCNN
- Mask RCNN
- Efficient Net

## Usage
1. set variables in .env
2. launch the docker compose to 
    1. generate the trt saved model for given saved model, the output directory is ${BASELINE_SAVED_MODEL_DIR}_out
    2. benchmark the saved model with provided data_dir
    3. benchmark the generated trt saved model as well
