# Train and test YOLOv5 instance segmentation models
## Activate the python virtual environments
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
- pad images [optinal]
- resize images [optinal]
- convert labelings to YOLO format

**The yolo model requires the dimension of images to be dividable by 32**. The padding and resizing are optional if the dimensions already meet this requirement. In this tutorial, we pad and resize images to 640x288.

Assume that the original data is downloaded in `./data/Canadian-SPF`. After execting the exmaple commands below, it will generate a yolo formatted dataset folder in `./data/resized_yolo`.

```bash
# modify the followings according to your data
pad_w=1600
pad_h=720
resize_w=640
resize_h=288

python -m label_utils.pad_with_csv --path_imgs ./data/Canadian-SPF --path_out ./data/pad --out_imsz $pad_w,$pad_h

python -m label_utils.resize_with_csv --path_imgs ./data/pad --path_out ./data/resized --out_imsz $resize_w,$resize_h

python -m label_utils.convert_data_to_yolo --path_imgs ./data/resized --path_out ./data/resized_yolo --seg
```

## Create a yaml file indicating the locations of datasets
After converting data to yolo format, a json file will be created in `./data/resized_yolo/class_map.json`. The order of class names in the yaml file **must match with** the order of names in the json file. 

Below is what is in the class_map.json:
```json
{"knot": 0, "wane": 1}
```

Below is the yaml file to be created:
```yaml
path: /home/user/data/resized_yolo  # dataset root dir
train: images  # train images (relative to 'path') 128 images
val: images  # val images (relative to 'path') 128 images
test:  # test images (optional)

# the order of names must match with the names in class_map.json!
names: 
  0: knot
  1: wane
```
We usually use today's date as the file name. If today is 07/19/2023, save it as `./config/2023-07-19.yaml`:


## Train the model
The yolov5 training script has the following arguments:
- img: image size (the largest edge if images are rectangular)
- batch: batch size
- data: the path to the yaml file
- **rect: if the images are rectangular**
- weights: the path to the pre-trained weights. The weights will be automatically downloaded if not exist. **This argument controls what model will be trained.** The available pre-trained models are yolov5n-seg, yolov5s-seg, yolov5m-seg, yolov5l-seg, yolov5x-seg.
- project: the output folder
- name: the subfolder to be created inside the output folder
- exist-ok(optional): overwrite the existing output subfolder

Below is an example of training a yolov5s-seg model:
```bash
python -m yolov5.segment.train --img 640 --batch 8 --rect --epoch 300 --data ./config/2023-07-19.yaml --weights ./pretrained-models/yolov5s-seg.pt --project training --name 2023-07-19 --exist-ok
```

## Monitor the training progress
While training process is running, open another terminal. Execuate the command below and go to http://localhost:6006 to monitor the training.
```bash
tensorboard --logdir ./training/2023-07-19
```


# Testing
## Save trained model
After training, the weights are saved in `./training/2023-07-19/weights/best.pt`. Copy the best.pt to `./trained-inference-models/2023-07-19`.

```bash
mkdir -p ./trained-inference-models/2023-07-19
cp ./training/2023-07-19/weights/best.pt ./trained-inference-models/2023-07-19
```

## Run inference
The command below generates perdictions using the following arguments:
- source: the path to the test images
- weights: the path to the trained model weights file
- img: a list of image size (h,w)
- project: the output folder
- conf-thres(optional): the confidence level, default is 0.25
- name: the subfolder to be created inside the output folder
- save-txt: save the outputs as a txt file

```bash
python -m yolov5.segment.predict --img 288 640 --source data/predict_images --weights trained-inference-models/2023-07-19/best.pt --project validation --name 2023-07-19 --exist-ok
```
The output results are saved in `./validation/2023-07-19`.

# Generate TensorRT Engine
Refer to here: https://github.com/lmitechnologies/yolov5_lmi/trt.
