# Train and test YOLOv5 object detection models
This is the tutorial how to train and test the yolov5 object detection models. If you want to train the yolov5 instance segmentation models, please refer to [segment.md](https://github.com/lmitechnologies/yolov5_lmi/segment.md).

## Activate the virtual environments
The command below assumes that the virtual python environment is installed in `~/virtual_env`.

```bash
source ~/virtual_env/bin/activate
```

## Install the dependencies
```bash
git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && cd LMI_AI_Solutions && git submodule update --init object_detectors/submodules/yolov5

pip install -r object_detectors/submodules/yolov5/requirements.txt
```

## Activate LMI_AI environment
The command below assumes that the LMI_AI_Solution repo is cloned in `~/LMI_AI_Solutions`.

```bash
source ~/LMI_AI_Solutions/lmi_ai.env
```

## Prepare the datasets
Prepare the datasets by the followings:
- resize images [optional]
- convert labeling data to YOLO format

**The yolo model requires the dimensions of images to be dividable by 32**. The resizing is optional if the dimensions already meet this requirement. In this tutorial, we resize images to 640x320.

Assume that the original data is downloaded in `./data/allImages_1024`. After execting the exmaple commands below, it will generate a yolo formatted dataset folder in `./data/resized_yolo`.

```bash
# modify the width and height to your data
W=640
H=320

python -m label_utils.resize_with_csv --path_imgs ./data/allImages_1024 --out_imsz $W,$H --path_out ./data/resized

python -m label_utils.convert_data_to_yolo --path_imgs ./data/resized --path_out ./data/resized_yolo
```

## Create a yaml file indicating the locations of datasets
After converting data to yolo format, a json file will be created in `./data/resized_yolo/class_map.json`. The order of class names in the yaml file **must match with** the order of names in the json file. 

Below is what is in the class_map.json:
```json
{"peeling": 0, "scuff": 1, "white":2}
```

Below is the yaml file that need to be created:
```yaml
path: /home/user/project/data/resized_yolo  # dataset root dir (must use absolute path!)
train: images  # train images (relative to 'path')
val: images  # val images (relative to 'path')
test:  # test images (optional)
 
# Classes
names: # class names must match with the names in class_map.json
  0: peeling
  1: scuff
  2: white
```
We usually use today's date as the config file name. If today is 07/19/2023, then save the file as `./config/2023-07-19.yaml`.


## Train the model
The yolov5 training script has the following arguments:
- img: image size (use the **longer** edge if images are rectangular)
- batch: batch size
- data: the path to the yaml file
- **rect: if the images are rectangular**
- weights: the path to the pre-trained weights. The weights will be automatically downloaded if not exist. **This argument controls what model will be trained.** The available pre-trained models are yolov5n, yolov5s, yolov5m, yolov5l, yolov5x.
- project: the output folder
- name: the subfolder inside the output folder. We usually use today's date as the subfolder name.
- exist-ok(optional): overwrite the existing output subfolder

Below is an example of training a yolov5s model: 
```bash
python -m yolov5.train --img 640 --rect --batch 16 --epoch 300 --data ./config/2023-07-19.yaml --weights ./pretrained-models/yolov5s.pt --project ./training --name 2023-07-19 --exist-ok
```

## Monitor the training progress
```bash
tensorboard --logdir ./training/2023-07-19
```
While training process is running, open another terminal.
Execuate the command above and go to http://localhost:6006 to monitor the training.


# Testing
## Save trained model
After training, the weights are saved in `./training/2023-07-19/weights/best.pt`. Copy the best.pt to `./trained-inference-models/2023-07-19`.

```bash
mkdir -p ./trained-inference-models/2023-07-19
cp ./training/2023-07-19/weights/best.pt ./trained-inference-models/2023-07-19
```

## Run inference
The command below run the inference using the following arguments:
- source: the path to the test images
- weights: the path to the trained model weights file
- img: a list of the image size (h,w)
- project: the output folder
- conf-thres(optional): the confidence level, default is 0.25
- name: the subfolder to be created inside the output folder
- save-csv: save the outputs as a csv file

```bash
python -m yolov5.detect --source ./data/resized_yolo/images --weights ./trained-inference-models/2023-07-19/best.pt --img 320 640 --project ./validation --name 2023-07-19
```
The output results are saved in `./validation/2023-07-19`.

# Generate TensorRT Engine
Refer to here: https://github.com/lmitechnologies/yolov5_lmi/trt.
