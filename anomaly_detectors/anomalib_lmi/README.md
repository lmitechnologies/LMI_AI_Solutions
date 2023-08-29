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
2. Although test and the ground_truth are optional, it enables the training to generate insightful metrics
> * Simply polygon label your test samples with [VGG](https://www.robots.ox.ac.uk/~vgg/software/via/via.html), 
> * Convert the labels to ground_truth format with lmi_utils/label_utils/json_to_ground_truth.py
> * Put the test images into root_dir/test, corresponding ground_truth into root_dir/ground_truth as #1
> * At the end of the training, it will generate metrics like this:
```bash
        image_AUROC          0.8500000238418579
       image_F1Score         0.9268293380737305
        pixel_AUROC          0.9906309843063354
       pixel_F1Score         0.4874393045902252
```
## Train
1. bash main.sh -a train -d /path/to/train/data/root_dir -o /path/to/outdir
2. For more detailed usage, execute bash main.sh -h
2. Check the corresponding [config file](https://openvinotoolkit.github.io/anomalib/reference_guide/algorithms/patchcore.html), important fields:
```bash
# general
dataset
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
4. Find the trained model in outdir/results

## Generate TRT engine
1. bash main.sh -a convert -x /path/to/onnx  -o /path/to/outdir
2. Find the generated engine at outdir/engines
3. Run without -x /path/to/onnx, it defaults to the latest onnx folder in the outdir/results/{modeltype}/trained-models

## Unit Test
1. bash main.sh -a test -e /path/to/trt/engine -t /path/to/test/data -o /path/to/outdir
2. Find annotated prediction result at outdir/predictions
3. Run without -e /path/to/trt/engine, it defaults to the latest trt engine folder in the outdir/engines

## Code Example
```python
am = AnomalyModel(padim_or_patchcore_trt_engine)

am.warmup()

decision, annotation, more_details = am.predict(
        image, 
        err_threshold   # reference the adaptive err_threshold from model/run/weights/onnx/metadata.json 
)
```