# AD Error Map Masking Functions

This document describes the masking functions used to mask Anomaly Detection (AD) error maps based on Object Detection (OD) predictions.

## Overview

The masking module provides functionality to suppress AD error map responses in regions detected by an Object Detection model. This is useful for filtering out false positives in specific object regions and improving anomaly detection accuracy.

## Core Functions

### `blur_mask(mask: np.ndarray, kernel_size: int, distance_based: bool = False) -> np.ndarray`

Blurs a mask using either Gaussian blur or distance transform-based soft weighting.

**Parameters:**
- `mask` (np.ndarray): Input binary or soft mask (values typically in [0, 1])
- `kernel_size` (int): Size of the blur kernel
  - For Gaussian blur: must be odd, values like 3, 5, 7, 11, etc.
  - For distance-based blur: must be in [0, 3, 5]
- `distance_based` (bool): If False (default), uses Gaussian blur. If True, uses distance transform-based soft weights.

**Returns:**
- np.ndarray: Blurred mask with same shape as input

**Notes:**
- When `distance_based=False`, uses OpenCV's `GaussianBlur`
- When `distance_based=True`, applies distance transform and converts distances to soft weights using exponential decay (σ=5.0)
- Invalid kernel sizes are automatically corrected with a warning
- Gaussian blur kernel size will be incremented by 1 if even

**Example:**
```python
import numpy as np
from pipeline_utils import blur_mask

# Gaussian blur
mask = np.random.rand(256, 256)
blurred = blur_mask(mask, kernel_size=11, distance_based=False)

# Distance-based soft weights
soft_mask = blur_mask(mask, kernel_size=5, distance_based=True)
```

---

### `apply_ad_mask(err_map: np.ndarray, od_predictions: dict, mask_config: dict, class_names=None) -> tuple`

Applies OD-based masks to suppress regions in an AD error map.

**Parameters:**
- `err_map` (np.ndarray): Anomaly Detection error map (H, W) or (H, W, C)
- `od_predictions` (dict): Object Detection predictions containing:
  - `masks`: Array of detection masks
  - `classes`: List of class names for each detection
  - `scores`: Confidence scores for each detection
- `mask_config` (dict): Configuration dictionary with structure:
  ```python
  {
      "global_weight": float,  # Global multiplier for all masks
      "masking_params": {
          "class_id": {
              "weight": float,  # Default: 1.0
              "weight_by_confidence": bool,  # Default: True
              "erode_kernel_size": int,  # Default: 0 (no erosion)
              "blur_kernel_size": int,  # Default: 11
              "multiply_by_mask": bool,  # Default: True
              "simple_blur": bool,  # Default: True
          }
      },
  }
  ```
- `class_names` (list, optional): List of class names to process. If None, processes all classes.

**Returns:**
- tuple: (masked_err_map, total_mask)
  - `masked_err_map` (np.ndarray): Error map with suppressed regions
  - `total_mask` (np.ndarray): Cumulative mask applied (clipped to [0, 1])

**Configuration Details:**
- `weight`: Per-class masking strength
- `weight_by_confidence`: If True, multiplies weight by detection confidence score
- `erode_kernel_size`: If > 0, applies morphological erosion to mask before blurring
- `blur_kernel_size`: Controls blur intensity (larger = more blur)
- `multiply_by_mask`: If True, scales error map by mask, else subtract mask from error map
- `simple_blur`: If True, uses Gaussian blur; if False, uses distance-based soft weighting

**Example:**
```python
from pipeline_utils import apply_ad_mask

# Configuration for masking
mask_config = {
    "global_weight": 0.8,
    "masking_params": {
        "person": {
            "weight": 1.0,
            "weight_by_confidence": True,
            "erode_kernel_size": 3,
            "blur_kernel_size": 11,
            "multiply_by_mask": True,
            "simple_blur": True,
        },
        "vehicle": {"weight": 0.5, "weight_by_confidence": True},
    },
}

# Apply masking
masked_err_map, total_mask = apply_ad_mask(err_map, od_predictions, mask_config, class_names=["person", "vehicle"])
```

---

### `masked_ad_predict(pipe, ad_inp, ad_model_role: str | np.ndarray, od_model_role: str, configs: dict, class_names=None) -> tuple`

Performs AD prediction with OD-based masking applied.

**Parameters:**
- `pipe`: Inference pipeline object with preprocessing/model prediction methods
- `ad_inp` (np.ndarray): Input image for anomaly detection
- `ad_model_role` (str or np.ndarray): Either model role name or pre-computed error map
- `od_model_role` (str): Object detection model role name
- `configs` (dict): Configuration dictionary containing model configs (passed to pipeline) and masking parameters
- `class_names` (list, optional): Specific classes to use for masking

**Returns:**
- tuple: (masked_err_map, od_predictions, mask_img)
  - `masked_err_map` (np.ndarray): Masked anomaly error map
  - `od_predictions` (dict): Object detection predictions in original image space
  - `mask_img` (np.ndarray): Visualization of applied mask as RGB image

**Example:**
```python
from pipeline_utils import masked_ad_predict

masked_err, od_preds, mask_vis = masked_ad_predict(
    pipe, input_image, ad_model_role="anomaly_detector", od_model_role="object_detector", configs=configs, class_names=["person"]
)
```

---

### `masked_ad_annotate(pipe, img, ad_model_role, od_model_role, err_map, od_predictions, configs, color=(152, 251, 152)) -> np.ndarray`

Annotates an image with both AD error map heatmap and OD bounding boxes.

**Parameters:**
- `pipe`: Inference pipeline object
- `img` (np.ndarray): Input image for annotation (BGR format)
- `ad_model_role` (str): AD model role name
- `od_model_role` (str): OD model role name
- `err_map` (np.ndarray): Anomaly error map to visualize
- `od_predictions` (dict): Object detection predictions (from OD .predict)
- `configs` (dict): Configuration dictionary with thresholds and parameters
- `color` (tuple, optional): RGB color for OD bounding boxes. Default: light green (152, 251, 152)

**Returns:**
- np.ndarray: Annotated image with AD heatmap and OD boxes

**Configuration Expected:**
```python
configs["models"][ad_model_role]["configs"] should contain:
  - "min_threshold": float  # Lower threshold for heatmap
  - "max_threshold": float  # Upper threshold for heatmap
```

**Example:**
```python
from pipeline_utils import masked_ad_annotate

annotated = masked_ad_annotate(
    pipe,
    img,
    ad_model_role="anomaly_detector",
    od_model_role="object_detector",
    err_map=masked_err_map,
    od_predictions=od_preds,
    configs=configs,
    color=(0, 255, 0),  # Green
)
```

---

## Workflow Example

```python
import numpy as np
from pipeline_utils import masked_ad_predict, masked_ad_annotate

# Define masking configuration
mask_config = {
    "global_weight": 0.9,
    "masking_params": {"defect": {"weight": 1.0, "blur_kernel_size": 15}, "scratch": {"weight": 0.7, "blur_kernel_size": 9}},
}

# Configuration with model thresholds
configs = {
    "models": {"ad_model": {"configs": {"min_threshold": 0.2, "max_threshold": 0.8}}, "od_model": {"configs": {"confidence": 0.5}}},
    "od_model": mask_config,
}

# Step 1: Run masked AD prediction
masked_err_map, od_preds, mask_vis = masked_ad_predict(
    pipe, input_image, ad_model_role="ad_model", od_model_role="od_model", configs=configs
)

# Step 2: Annotate results
result_img = masked_ad_annotate(
    pipe, input_image, ad_model_role="ad_model", od_model_role="od_model", err_map=masked_err_map, od_predictions=od_preds, configs=configs
)
```

## Key Design Principles

1. **Hierarchical Masking**: Per-class configurations allow different suppression strategies for different object types
2. **Confidence Weighting**: Detection confidence scores can modulate masking strength
3. **Soft Boundaries**: Blur and distance-based smoothing prevent hard edges in masked regions
4. **Flexible Suppression**: Masks can be applied additively or multiplicatively
5. **Morphological Operations**: Erosion reduces over-masking of object boundaries

## Performance Considerations

- Distance-based blurring is more computationally expensive than Gaussian blur
- Larger blur kernels increase processing time
- Morphological erosion adds overhead proportional to kernel size

## Common Configuration Patterns

### Conservative Masking (Minimal Suppression)
```python
{"global_weight": 0.3, "masking_params": {"all_classes": {"weight": 0.5}}}
```

### Aggressive Masking (Strong Suppression)
```python
{"global_weight": 1.0, "masking_params": {"all_classes": {"weight": 1.0, "blur_kernel_size": 21}}}
```

### Distance-Based Soft Masking
```python
{"masking_params": {"all_classes": {"simple_blur": False, "blur_kernel_size": 5}}}
```
