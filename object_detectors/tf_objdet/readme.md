# Install Dependencies
## Clone LMI AI Solutions Repo
```bash
git clone git@github.com:lmitechnologies/LMI_AI_Solutions.git
git submodule update --init --recursive
```
## Activate the Object Detection API
Run following from within ./LMI_AI_Solutions/object_detectors/tf_objdet/models/research
```bash
protoc object_detection/protos/*.proto --python_out=.
```
## Install Python Modules
    pip3 install tensorflow --upgrade
    pip3 install tf_slim
    pip3 install pycocotools
    pip3 install lvis
    pip3 install scipy
    pip3 install tensorflow_io
    pip3 install tf-models-official

# Training Steps

## Activate LMI_AI environment
The following commands assume that the LMI_AI_Solution repo is cloned in `~/LMI_AI_Solutions`.

```bash
source ~/LMI_AI_Solutions/lmi_ai.env 
```

## Label Images

[VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)

## Export Label Data

## Initialize Project Directory Structure

```
|-- config
|
|-- data
|
|-- pretrained-models
|
|-- records
|
|-- trained-inference-models
|
|-- training
|
`-- validation
```

## Generate .csv from .json

``` bash
python3 -m label_utils.via_json_to_csv -d ./data --output_fname labels.csv --label_name=Name --render True --is_mask False --mask_to_bbox False
```

## Set Training Parameters in config.py
``` python
import os

#%% Define paths for data
DATA_PATH='./data'
ANNOT_PATH=os.path.sep.join([DATA_PATH,'labels_v1/labels_v1.csv'])

#%% Define paths for records
TRAIN_RECORD='./records/v1/training.record'
TEST_RECORD='./records/v1/testing.record'
CLASSES_FILE='./records/v1/classes.pbtxt'

#%% Feature options 
MASK_OPTION=False
KEYPOINT_OPTION=True

#%% Resizing options
RESIZE_OPTION=True
MAX_W=512

#%% Initialize training/test split
TEST_SIZE=0.1

#%% initialize the class labels dictionary
CLASSES={'class_1':1}
KEYPOINTS={'class_1':{'keypoint_0':0,'keypoint_1':1,'keypoint_2':2}}

```


## Build TF Records

``` bash
python3 -m tf_objdet.lmi_utils.build_records_v2 -c ./config/config.py
```

## Create your_application.config File
Download pretrained model from TensorFlow 2 Detection Model Zoo \n
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Modify Accordingly

## Train 

``` bash
python3 -m tf_objdet.lmi_utils.model_main_tf2 --model_dir=training --pipeline_config_path=config/pipeline.config

```

## Launch TensorBoard

``` bash
tensorboard --logdir=training --bind_all
```

## Export Trained Model

``` bash
python3 -m tf_objdet.lmi_utils.exporter_main_v2 --input_type image_tensor --pipeline_config_path ./config/pipeline.config --trained_checkpoint_dir ./training --output_directory ./trained-inference-models
```

## Test & Validate

``` bash
python3 -m tf_objdet.lmi_utils.predict_v2 --savedmodel ./trained-inference-models/saved_model --labels ./records/classes.pbtxt --image ./data/path_to_images --min_dim 1024 --max_dim 1024 --num-classes 2 --min-confidence 0.5 --draw True -s ./data/validation_images -o ./data/validation_images/prediction_results.csv
```
