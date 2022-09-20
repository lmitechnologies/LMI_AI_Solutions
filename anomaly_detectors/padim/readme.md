# Training Steps

## Activate LMI_AI environment
The following commands assume that the LMI_AI_Solution repo is cloned in `~/LMI_AI_Solutions`.

```bash
source ~/LMI_AI_Solutions/lmi_ai.env 
```

## Preapre the yaml config file
Currently, the padim model supports two backbones: efficientnet and resnet.
The layers to select for resnet: 
- pool1_pool
- conv{i}_block1_preact_relu, where i = 2,3,4,5
The layers to select for efficientnet: 
- stem_activation
- block{i}a_activation, where i = 2,3,4,5,6

This is an example:
```yaml
backbone:
  name: eff # eff or res
  layer1: stem_activation
  layer2: block2a_activation
  layer3: block3a_activation
```

## Train the model
Use the `train.py` to train the model and it has the following arguments:
- path_data: the path to training data
- path_out: the output path where it save the trained models
- config_yaml: the yaml file specifies the type of model and its layers
- imsz: (optional) comma separated image dimension: w,h. default=224,224
- n: (optional) the number of vectors to randomly draw, default=200
- batch_sz: (optional) batch size, default=8
- gpu_mem: (optional) gpu memory limit, default=16384

Here is an example:
```bash
python3 -m padim.train --path_data ./data/resized_train_augmented --path_out ./outputs --config_yaml ./efficientnet_layers.yaml --gpu_mem 16000 --batch_sz 8
```

# Testing
Use the `test.py` to test and it has the following arguments:
- path_data: the path to the test images
- path_model: the path to the saved model
- path_out: the output path
- thres_err: (optional) the error dist threshold, default=20
- gpu_mem: (optional) the gpu memory limit, default=2048


Here is an example:
```bash
python3 -m padim.test --path_data ./data/resized_test --path_model ./outputs/saved_model --path_out ./outputs --thres_err 20
```

