# Pre-Release Notes — v2.0.0 → next

Breaking changes and new features since `v2.0.0`. Each breaking change says what to update.

## Breaking changes

### `gofactory_AD` removed

**PR:** [#452](https://github.com/lmitechnologies/LMI_AI_Solutions/pull/452)

`anomaly_detectors.anomalib_lmi.gofactory_AD`, the script GoFactory's Anomalib v1 trainer ran on its validation and testing images, is gone. GoFactory now keeps that step in its own repository.

Its `--tile` and `--stride` options had no effect since [#263](https://github.com/lmitechnologies/LMI_AI_Solutions/pull/263), which moved AD tiling out of the model. To score a folder of images, with or without tiling, use the `test` action:

```bash
python -m anomaly_detectors.anomalib_lmi.v1.model test -i model.pt -d images -o out --tile 256 256 --stride 224 224
```

## New features

### `ONNXEngine` sizes unrecorded outputs at load

**PR:** [#455](https://github.com/lmitechnologies/LMI_AI_Solutions/pull/455)

`ONNXEngine` now sizes the output dims an ONNX file leaves symbolic with one inference at load, so EfficientAd ONNX exports load. Exports no longer record those dims, so a variable-batch ONNX exported from now on needs this AIS on the pipeline.
