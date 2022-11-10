# LMI AI Solutions
This repo contains the utils scripts, and several submodules for LMI AI modeling development.
Currently, the following submodules are included in the repo:
- [yolov5](https://github.com/lmitechnologies/yolov5)
- [efficientnet](https://github.com/lmitechnologies/EfficientNet-PyTorch)
- [tensorflow object detection API](https://github.com/lmitechnologies/models)
- [paddleOCR](https://github.com/lmitechnologies/models)
- [tf-trt](https://github.com/tensorflow/tensorrt.git)

## Clone this master repo
For users who haven't set up the ssh keys
```bash
git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git
```
For users who have the ssh keys
```bash
git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git
# npm using git for https
git config --global url."git@github.com:".insteadOf https://github.com/
git config --global url."git://".insteadOf https://
```

## Clone submodules
Go to the master repo
```bash
cd LMI_AI_Solutions
```
Each submodule is pointing to a specific commit in its `ais` branch. Clone the submodules to the commit that is specified in this repo 
```bash
git submodule update --init --recursive
```
(optional) if you want to update all submodules to the `lastest` commit in the `ais` branch, use the `--remote` argument
```bash
git submodule update --init --recursive --remote
```

## Make contributions to this repo
The `ais` branch of this repo and that branch of submodules are protected, which means you can't directly commit to that branch. You could create a new branch and open the pull request in order to merge into `ais` branch.
