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
2. Although optional test and the ground_truth are, it enables the training to generate insightful metrics
> * Simply polygon label your ground_truth with [VGG](https://www.robots.ox.ac.uk/~vgg/software/via/via.html), 
> * Convert the labels to ground_truth format with lmi_utils/label_utils/json_to_ground_truth.py
> * Update 'abnormal_dir', 'normal_test_dir' and 'mask' fileds in the config
> * At the end of the training, metrics show like this:
```bash
        image_AUROC          0.8500000238418579
       image_F1Score         0.9268293380737305
        pixel_AUROC          0.9906309843063354
       pixel_F1Score         0.4874393045902252
```
## Train
1. Update .env, with ACTION to train
2. Check the corresponding [config file](https://openvinotoolkit.github.io/anomalib/reference_guide/algorithms/patchcore.html), important fields:
```bash
# general
dataset
|---- path
|---- normal_dir
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
3. Launch the docker-compose.yaml to run
4. Find the trained model in untracked-data/results
5. Find evaluation result in untracked-data/evaluations if test data given

## Generate TRT engine
1. Update ACTION to convert in .env
2. Launch the docker-compose.yaml to run
3. Find the generated engine at untracked-data/engines

## Unit Test
1. Update ACTION to unittest in .env
2. Ensure the arguments of unit_test in anomaly_model.py reflect the truth for you
3. Launch the docker-compose.yaml to run
4. Find annotated prediction result at untracked-data/predictions

## Code Example
```python
am = AnomalyModel(padim_or_patchcore_trt_engine)

am.warmup()

decision, annotation, more_details = am.predict(
        image, 
        err_threshold   # reference the adaptive err_threshold from model/run/weights/onnx/metadata.json 
)
```